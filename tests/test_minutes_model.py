import numpy as np
import pandas as pd
import pytest

from fpl_predictor.minutes_model import MinutesModel, train_minutes_model


def _toy_feat_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    mins_roll3_mean = rng.uniform(0, 90, n)
    # Construct a target that's actually learnable from the minutes pattern
    # (mostly-nailed players start, fringe players don't), plus noise.
    prob_60 = np.clip(mins_roll3_mean / 90.0 + rng.normal(0, 0.05, n), 0, 1)
    minutes = np.where(rng.random(n) < prob_60, rng.integers(60, 91, n), rng.integers(0, 45, n))
    return pd.DataFrame({
        "now_cost": rng.uniform(4, 13, n),
        "mins_roll3_mean": mins_roll3_mean,
        "mins_roll3_std": rng.uniform(0, 30, n),
        "mins_frac_last3": mins_roll3_mean / 90.0,
        "started_60_last6": rng.uniform(0, 1, n),
        "total_points_roll3_mean": rng.uniform(0, 8, n),
        "minutes": minutes,
    })


def test_train_minutes_model_produces_a_usable_classifier():
    mm = train_minutes_model(_toy_feat_df(n=300))
    assert mm.model is not None
    assert mm.brier_score == mm.brier_score  # not NaN
    assert 0 <= mm.brier_score <= 1


def test_minutes_model_predict_proba_is_bounded_and_discriminates():
    df = _toy_feat_df(n=300)
    mm = train_minutes_model(df)
    proba = mm.predict_proba(df[mm.feature_cols])
    assert len(proba) == len(df)
    assert ((proba >= 0) & (proba <= 1)).all()
    # Players with high recent minutes should get a higher predicted
    # start probability on average than players with low recent minutes.
    high = df["mins_roll3_mean"] > 70
    low = df["mins_roll3_mean"] < 20
    assert proba[high.to_numpy()].mean() > proba[low.to_numpy()].mean()


def test_minutes_model_falls_back_gracefully_on_too_little_data():
    mm = train_minutes_model(_toy_feat_df(n=10))
    assert mm.model is None
    assert mm.brier_score != mm.brier_score  # NaN
    proba = mm.predict_proba(pd.DataFrame({"now_cost": [5.0, 8.0]}))
    assert len(proba) == 2
    assert ((proba > 0) & (proba < 1)).all()


def test_minutes_model_handles_empty_input():
    mm = train_minutes_model(pd.DataFrame())
    assert mm.model is None
    assert mm.predict_proba(pd.DataFrame()).tolist() == []
