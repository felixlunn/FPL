"""Prediction intervals (floor/ceiling) for each player's points estimate,
alongside the main Tweedie point-estimate model.

Captaincy decisions benefit from knowing more than just "expected
points": a safe, nailed-on 6-8 point player and an explosive-or-blank
differential can share almost the same mean prediction while having very
different distributions around it. This trains a 10th-percentile
("floor") and 90th-percentile ("ceiling") LightGBM quantile regressor per
position on the same feature set as the main model, so a squad's
starting XI can be compared not just on who scores the most on average,
but on who's the safe pick vs. who's the high-ceiling swing.

These are a supplementary signal, not the primary prediction, so they're
trained with a single fixed, untuned hyperparameter configuration rather
than the main model's full walk-forward search -- keeps this cheap to
add on top of an already-trained pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from lightgbm import LGBMRegressor

from fpl_predictor.config import POS_MAP, SEED

QUANTILES = {"floor": 0.1, "ceiling": 0.9}
_PARAMS = {"n_estimators": 400, "learning_rate": 0.06, "max_depth": 5, "num_leaves": 24, "min_child_samples": 20}
MIN_ROWS = 60  # per position, below this a quantile split is too noisy to trust


@dataclass
class QuantileModel:
    feature_cols: list[str]
    models: dict = field(default_factory=dict)  # (position_id, "floor"|"ceiling") -> LGBMRegressor

    def predict(self, X: pd.DataFrame, position_col: str = "element_type") -> pd.DataFrame:
        """Returns a DataFrame with 'floor' and 'ceiling' columns, index-aligned to X.
        Positions with no trained model (too little data) get NaN, which
        callers should treat as "no interval available" rather than 0.
        """
        out = pd.DataFrame(index=X.index, columns=list(QUANTILES.keys()), dtype=float)
        for pos_id in X[position_col].unique():
            mask = (X[position_col] == pos_id).to_numpy()
            if not mask.any():
                continue
            Xf = X.loc[mask, self.feature_cols].fillna(0)
            for name in QUANTILES:
                model = self.models.get((pos_id, name))
                if model is not None:
                    out.loc[mask, name] = model.predict(Xf)
        return out


def train_quantile_model(feature_df: pd.DataFrame, feature_cols: list[str], label_col: str = "label_points", min_rows: int = MIN_ROWS) -> QuantileModel:
    qm = QuantileModel(feature_cols=feature_cols)
    for pos_id in POS_MAP:
        pos_df = feature_df[feature_df["element_type"] == pos_id]
        if len(pos_df) < min_rows:
            continue
        X = pos_df[feature_cols].fillna(0)
        y = pos_df[label_col]
        for name, alpha in QUANTILES.items():
            model = LGBMRegressor(random_state=SEED, n_jobs=-1, verbosity=-1, objective="quantile", alpha=alpha, **_PARAMS)
            model.fit(X, y)
            qm.models[(pos_id, name)] = model
    return qm
