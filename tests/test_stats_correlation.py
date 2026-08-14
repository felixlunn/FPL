import numpy as np
import pandas as pd

from fpl_predictor.stats_correlation import compute_stat_correlations, compute_stat_correlations_by_position


def _toy_history(n=100, seed=0):
    rng = np.random.default_rng(seed)
    ict = rng.uniform(0, 20, n)
    # total_points strongly driven by ict_index plus noise, threat is pure
    # noise -- so the correlation ranking should recover that distinction.
    points = ict * 0.4 + rng.normal(0, 1, n)
    return pd.DataFrame({
        "player_id": np.arange(n) % 10,
        "ict_index": ict,
        "threat": rng.uniform(0, 20, n),
        "total_points": points,
    })


def test_compute_stat_correlations_recovers_known_relationship():
    corr = compute_stat_correlations(_toy_history())
    assert set(corr["stat"]) == {"ict_index", "threat"}
    by_stat = corr.set_index("stat")
    assert by_stat.loc["ict_index", "pearson_r"] > 0.7
    assert abs(by_stat.loc["threat", "pearson_r"]) < 0.3
    assert by_stat.loc["ict_index", "pearson_r"] > by_stat.loc["threat", "pearson_r"]


def test_compute_stat_correlations_handles_empty_and_missing_label():
    assert compute_stat_correlations(pd.DataFrame()).empty
    assert compute_stat_correlations(pd.DataFrame({"ict_index": [1, 2, 3]})).empty


def test_compute_stat_correlations_skips_constant_columns():
    df = _toy_history()
    df["threat"] = 5.0  # zero variance -> undefined correlation, must be skipped not crash
    corr = compute_stat_correlations(df)
    assert "threat" not in set(corr["stat"])
    assert "ict_index" in set(corr["stat"])


def test_compute_stat_correlations_by_position_splits_correctly():
    hist = _toy_history(n=200)
    players = pd.DataFrame({"id": range(10), "element_type": [1, 1, 2, 2, 2, 2, 3, 3, 4, 4]})
    by_pos = compute_stat_correlations_by_position(hist, players)
    assert not by_pos.empty
    assert set(by_pos["position"]).issubset({"GKP", "DEF", "MID", "FWD"})
