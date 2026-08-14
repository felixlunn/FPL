import numpy as np
import pandas as pd

from fpl_predictor.quantile_model import train_quantile_model


def _toy_feature_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    now_cost = rng.uniform(4, 13, n)
    element_type = rng.choice([1, 2, 3, 4], size=n)
    label = np.maximum(0, rng.normal(now_cost * 0.5, 2.0, n))
    return pd.DataFrame({"now_cost": now_cost, "element_type": element_type, "label_points": label})


def test_train_quantile_model_produces_sensible_floor_ceiling_ordering():
    df = _toy_feature_df()
    qm = train_quantile_model(df, feature_cols=["now_cost"])
    preds = qm.predict(df)
    valid = preds.dropna()
    assert not valid.empty
    # Ceiling should be >= floor for (almost) every row -- quantile
    # regressors are trained independently so a rare crossover is
    # possible, but the vast majority must be correctly ordered.
    assert (valid["ceiling"] >= valid["floor"]).mean() > 0.95


def test_train_quantile_model_skips_positions_with_too_little_data():
    df = _toy_feature_df(n=300)
    df.loc[df["element_type"] == 1, "element_type"] = 2  # empty out GKP entirely
    qm = train_quantile_model(df, feature_cols=["now_cost"], min_rows=60)
    assert (1, "floor") not in qm.models
    assert (2, "floor") in qm.models


def test_quantile_model_predict_returns_nan_for_untrained_position():
    df = _toy_feature_df(n=300)
    qm = train_quantile_model(df, feature_cols=["now_cost"], min_rows=10_000)  # nothing trains
    preds = qm.predict(df)
    assert preds["floor"].isna().all()
    assert preds["ceiling"].isna().all()


def test_quantile_model_handles_empty_input():
    qm = train_quantile_model(pd.DataFrame(columns=["now_cost", "element_type", "label_points"]), feature_cols=["now_cost"])
    assert qm.models == {}
