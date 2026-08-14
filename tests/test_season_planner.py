import pandas as pd
import pytest

from fpl_predictor.optimizer import find_optimal_squad
from fpl_predictor.pipeline import build_demo_data, run_pipeline
from fpl_predictor.season_planner import blank_double_gw_summary, build_multi_gw_forecast, recommend_chip_windows


@pytest.fixture(scope="module")
def result():
    data = build_demo_data(n_gws=10, n_future_gws=6, seed=1)
    return run_pipeline(data)


def test_build_multi_gw_forecast_covers_the_full_horizon(result):
    multi_pred, gws = build_multi_gw_forecast(result, n_gws=5)
    assert len(gws) == 5
    assert gws[0] == result.future_gws[0]
    assert not multi_pred.empty
    for gw in gws:
        assert f"GW{gw}_Points" in multi_pred.columns
    # Floor/ceiling should also come through since the demo pipeline trains
    # a quantile model.
    assert f"GW{gws[0]}_Floor" in multi_pred.columns
    assert (multi_pred["pred_points_total"] >= 0).all()


def test_build_multi_gw_forecast_cold_start_mode():
    data = build_demo_data(n_gws=0, n_future_gws=4, seed=2)
    result = run_pipeline(data)
    assert result.mode == "cold_start"
    multi_pred, gws = build_multi_gw_forecast(result, n_gws=3)
    assert len(gws) == 3
    assert not multi_pred.empty
    for gw in gws:
        assert f"GW{gw}_Points" in multi_pred.columns
    assert (multi_pred["pred_points_total"] >= 0).all()


def test_blank_double_gw_summary_matches_fixture_counts(result):
    gws = result.future_gws[0:1] + [result.future_gws[0] + 1]
    summary = blank_double_gw_summary(result.data.fixtures_df, result.data.teams_df, gws)
    assert list(summary["gw"]) == gws
    assert (summary["n_blank"] >= 0).all()
    assert (summary["n_double"] >= 0).all()
    # The demo dataset pairs teams up every gameweek with no blanks/doubles.
    assert (summary["n_blank"] == 0).all()
    assert (summary["n_double"] == 0).all()


def test_recommend_chip_windows_returns_sorted_recommendations_with_reasons(result):
    multi_pred, gws = build_multi_gw_forecast(result, n_gws=6)
    squad = find_optimal_squad(result.pred_df, points_col="pred_points_total_adj")
    recs = recommend_chip_windows(multi_pred, gws, result.data.fixtures_df, result.data.teams_df, current_squad_df=squad)
    # No blanks/doubles in the demo data, so Free Hit/Bench Boost shouldn't
    # fire, but Triple Captain (always has a max) and possibly Wildcard can.
    chips = {r.chip for r in recs}
    assert "Free Hit" not in chips
    assert "Bench Boost" not in chips
    for r in recs:
        assert r.reason
        assert r.gw in gws
    assert [r.gw for r in recs] == sorted(r.gw for r in recs)


def test_recommend_chip_windows_handles_empty_forecast():
    assert recommend_chip_windows(pd.DataFrame(), [], pd.DataFrame(), pd.DataFrame()) == []
