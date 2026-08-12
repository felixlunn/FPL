"""Training and inference for the fantasy-points prediction model.

A separate LightGBM regressor is trained per playing position (GKP / DEF /
MID / FWD). Scoring dynamics differ a lot by position -- clean sheets and
saves matter for goalkeepers and defenders, goals/assists dominate for
attackers -- so letting each position learn its own feature interactions
consistently out-performs one pooled model in backtests, at the cost of
needing more per-position data (mitigated by capping model complexity when
a position has few rows).

Model selection uses walk-forward (``TimeSeriesSplit``) cross-validation so
hyperparameters are never chosen using future gameweeks, which is the
single biggest source of over-optimistic offline metrics in FPL models.

The regression objective is Tweedie rather than the default L2/MAE.
FPL points are a skewed, mostly-small-non-negative-integer, zero-inflated
quantity (most player-gameweeks score 0-4, with an occasional double-digit
haul from a goal/assist/bonus combination) -- much closer to a compound
Poisson-Gamma process than to something Gaussian-distributed, and Tweedie
(1 < variance power < 2) is the standard choice for exactly that shape. A
handful of gameweeks score negative points (red card, own goal); Tweedie
requires a non-negative target, so those rows are clipped to 0 only for
the *training* label -- evaluation MAE is still computed against the true
(unclipped) score, so reported accuracy isn't flattered by the clipping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from lightgbm.callback import early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from fpl_predictor.config import POS_MAP, SEED

# Tweedie fits FPL's skewed, zero-inflated points distribution much better
# than a Gaussian (L2) objective -- see module docstring. 1.5 is the
# standard mid-range variance power (compound Poisson-Gamma) and a common,
# robust default when there's no strong reason to prefer either extreme.
OBJECTIVE_PARAMS = {"objective": "tweedie", "tweedie_variance_power": 1.5}

PARAM_GRID = [
    {"n_estimators": 600, "learning_rate": 0.05, "max_depth": 5, "num_leaves": 31, "min_child_samples": 20},
    {"n_estimators": 900, "learning_rate": 0.03, "max_depth": 6, "num_leaves": 47, "min_child_samples": 15},
    {"n_estimators": 400, "learning_rate": 0.08, "max_depth": 4, "num_leaves": 20, "min_child_samples": 25},
]


def _make_regressor(params: dict) -> LGBMRegressor:
    return LGBMRegressor(random_state=SEED, n_jobs=-1, verbosity=-1, **OBJECTIVE_PARAMS, **params)


def _non_negative(y: pd.Series) -> pd.Series:
    """Clip the rare negative-points rows for Tweedie training only (it
    requires y >= 0); evaluation should always use the true, unclipped y.
    """
    return y.clip(lower=0)


@dataclass
class PositionModel:
    position_id: int
    model: LGBMRegressor
    mae: float
    n_train_rows: int
    best_params: dict
    feature_importance: pd.DataFrame


@dataclass
class TrainedModel:
    """Bundle of per-position models plus the metadata needed to use them."""

    feature_cols: list[str]
    positions: dict = field(default_factory=dict)  # position_id -> PositionModel

    @property
    def overall_mae(self) -> float:
        weighted = [(p.mae, p.n_train_rows) for p in self.positions.values() if p.n_train_rows > 0]
        if not weighted:
            return float("nan")
        total_rows = sum(n for _, n in weighted)
        return sum(mae * n for mae, n in weighted) / total_rows

    def predict(self, X: pd.DataFrame, position_col: str = "element_type") -> np.ndarray:
        preds = np.zeros(len(X))
        for pos_id, pm in self.positions.items():
            mask = (X[position_col] == pos_id).to_numpy()
            if mask.any():
                preds[mask] = pm.model.predict(X.loc[mask, self.feature_cols].fillna(0))
        return preds


def _fit_one(X_train, y_train, X_val, y_val, params) -> LGBMRegressor:
    model = _make_regressor(params)
    model.fit(
        X_train, _non_negative(y_train),
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(period=0)],
    )
    return model


def _tune_position(X: pd.DataFrame, y: pd.Series, n_splits: int) -> tuple[dict, float, list[LGBMRegressor]]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_params, best_mae, best_models = None, np.inf, []
    for params in PARAM_GRID:
        fold_maes, fold_models = [], []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            model = _fit_one(X_tr, y_tr, X_te, y_te, params)
            fold_maes.append(mean_absolute_error(y_te, model.predict(X_te)))  # unclipped y_te: honest MAE
            fold_models.append(model)
        avg_mae = float(np.mean(fold_maes))
        if avg_mae < best_mae:
            best_mae, best_params, best_models = avg_mae, params, fold_models
    return best_params, best_mae, best_models


def train_position_model(
    feature_df: pd.DataFrame, feature_cols: list[str], position_id: int, label_col: str = "label_points", tune: bool = True,
) -> PositionModel | None:
    """Train one position's model. ``tune=False`` skips the hyperparameter
    grid/CV and fits a single reasonable configuration directly -- much
    faster, used by the backtest harness which retrains many times over.
    """
    pos_df = feature_df[feature_df["element_type"] == position_id]
    if pos_df.empty:
        return None
    X = pos_df[feature_cols].fillna(0)
    y = pos_df[label_col]

    n_samples = len(X)
    n_splits = min(5, max(2, n_samples // 500))
    if not tune or n_samples < 2 * n_splits:
        # Too little data to cross-validate meaningfully (or tuning was
        # skipped for speed): fit a single reasonable config directly.
        params = PARAM_GRID[0]
        model = _make_regressor(params)
        model.fit(X, _non_negative(y))
        return PositionModel(position_id, model, float("nan"), n_samples, params, _importance_df(model, feature_cols))

    best_params, best_mae, fold_models = _tune_position(X, y, n_splits)

    final_model = _make_regressor(best_params)
    final_model.fit(X, _non_negative(y))

    avg_importance = np.mean([m.feature_importances_ for m in fold_models], axis=0) if fold_models else final_model.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": avg_importance}).sort_values("importance", ascending=False)

    return PositionModel(position_id, final_model, best_mae, n_samples, best_params, imp_df)


def train_model(feature_df: pd.DataFrame, feature_cols: list[str], label_col: str = "label_points", tune: bool = True) -> TrainedModel:
    trained = TrainedModel(feature_cols=feature_cols)
    for pos_id in POS_MAP:
        pm = train_position_model(feature_df, feature_cols, pos_id, label_col, tune=tune)
        if pm is not None:
            trained.positions[pos_id] = pm
    return trained


def _importance_df(model: LGBMRegressor, feature_cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
