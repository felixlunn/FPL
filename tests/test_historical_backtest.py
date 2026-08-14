"""Tests the orchestration logic only, with fetch_season_data mocked out --
this module is inherently network-dependent (fetches real season data from
a public GitHub archive), unlike the rest of the offline test suite, so
real network access is never exercised here.
"""

from unittest.mock import patch

import fpl_predictor.historical_backtest as hb
from fpl_predictor.pipeline import build_demo_data


def _fake_fetch_ok(season):
    # Reuse the synthetic demo dataset as a stand-in "season" so the
    # backtest has something real to chew on without hitting the network.
    return build_demo_data(n_gws=8, n_future_gws=1, seed=hash(season) % 1000)


def _fake_fetch_fail(season):
    import pandas as pd
    from fpl_predictor.pipeline import DataBundle
    return DataBundle(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), meta={"ok": False, "error": "network unavailable"})


def test_run_historical_backtest_aggregates_successful_seasons():
    with patch.object(hb, "fetch_season_data", side_effect=_fake_fetch_ok):
        summary, reports = hb.run_historical_backtest(seasons=["2023-24", "2024-25"], min_train_gws=3, tune=False)
    assert len(summary) == 2
    assert summary["ok"].all()
    assert set(reports.keys()) == {"2023-24", "2024-25"}
    assert (summary["n_gws"] > 0).all()


def test_run_historical_backtest_handles_fetch_failure_gracefully():
    with patch.object(hb, "fetch_season_data", side_effect=_fake_fetch_fail):
        summary, reports = hb.run_historical_backtest(seasons=["2023-24"], min_train_gws=3, tune=False)
    assert len(summary) == 1
    assert summary.iloc[0]["ok"] == False  # noqa: E712
    assert summary.iloc[0]["n_gws"] == 0
    assert reports == {}
