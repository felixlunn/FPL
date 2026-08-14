"""A dedicated model for P(a player plays 60+ minutes) in the next
gameweek -- separating "will they feature meaningfully" from "how well
will they score if selected".

Rotation risk is a different, and often larger, source of FPL prediction
error than pure scoring ability: a player's rolling points average already
reflects their historical minutes pattern baked in, but that's a blunt,
implicit signal. An explicit classifier lets the app show a "start
probability" as its own number (useful on its own for squad decisions) and
lets a squad-rotation-prone player be flagged as risky even when their
underlying output, when they do play, still looks strong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

from fpl_predictor.config import SEED

# Deliberately just the minutes/form-pattern signal (not fixture difficulty
# or set-piece duty, which describe scoring quality, not selection odds).
MINUTES_FEATURE_COLS = [
    "now_cost", "mins_roll3_mean", "mins_roll3_std", "mins_frac_last3",
    "started_60_last6", "total_points_roll3_mean",
]

_FALLBACK_START_PROB = 0.6  # used when there's no model to ask (too little data)


@dataclass
class MinutesModel:
    model: LGBMClassifier | None
    feature_cols: list
    brier_score: float  # lower is better; NaN if untrained

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(60+ minutes) for each row of X. Falls back to a neutral
        constant if the model couldn't be trained (e.g. too little data).
        """
        if self.model is None or X.empty:
            return np.full(len(X), _FALLBACK_START_PROB)
        Xf = X.reindex(columns=self.feature_cols, fill_value=0.0).fillna(0.0)
        return self.model.predict_proba(Xf)[:, 1]


def train_minutes_model(feat_df: pd.DataFrame, min_rows: int = 60) -> MinutesModel:
    feature_cols = [c for c in MINUTES_FEATURE_COLS if c in feat_df.columns]
    if feat_df.empty or not feature_cols or "minutes" not in feat_df.columns:
        return MinutesModel(None, feature_cols, float("nan"))

    X = feat_df[feature_cols].fillna(0)
    y = (feat_df["minutes"] >= 60).astype(int)
    if len(X) < min_rows or y.nunique() < 2:
        # Not enough data (or a degenerate all-same-label target) to learn
        # anything meaningfully better than the fallback constant.
        return MinutesModel(None, feature_cols, float("nan"))

    n_splits = min(5, max(2, len(X) // 300))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    briers = []
    for train_idx, test_idx in tscv.split(X):
        y_train = y.iloc[train_idx]
        if y_train.nunique() < 2:
            continue  # a fold with only one class can't train a classifier
        clf = LGBMClassifier(random_state=SEED, n_jobs=-1, verbosity=-1, n_estimators=200, max_depth=4, learning_rate=0.08)
        clf.fit(X.iloc[train_idx], y_train)
        p = clf.predict_proba(X.iloc[test_idx])[:, 1]
        briers.append(brier_score_loss(y.iloc[test_idx], p))

    final = LGBMClassifier(random_state=SEED, n_jobs=-1, verbosity=-1, n_estimators=200, max_depth=4, learning_rate=0.08)
    final.fit(X, y)
    brier = float(np.mean(briers)) if briers else float("nan")
    return MinutesModel(final, feature_cols, brier)
