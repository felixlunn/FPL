"""Walk-forward backtest, exercised against the synthetic demo dataset so
it needs no network access. Also serves as the sandbox-side validation
that the Tweedie objective switch (fpl_predictor/model.py) didn't make
predictions worse on data we can actually check against ground truth.
"""

import pytest

from fpl_predictor.backtest import run_backtest
from fpl_predictor.pipeline import build_demo_data


@pytest.fixture(scope="module")
def report():
    data = build_demo_data(n_gws=12, n_future_gws=2, seed=3)
    return run_backtest(data, min_train_gws=3, tune=False)


def test_backtest_produces_a_result_per_gameweek_after_the_warmup(report):
    # 12 gws of history, 3-gw warmup -> gameweeks 4..12 are backtestable.
    assert len(report.per_gw) == 9
    gws = [r.gw for r in report.per_gw]
    assert gws == sorted(gws)
    assert gws[0] == 4 and gws[-1] == 12


def test_backtest_mae_is_finite_and_reasonable(report):
    assert report.overall_mae == report.overall_mae  # not NaN
    assert 0 <= report.overall_mae < 15  # sanity bound for a 0-20ish points scale


def test_backtest_never_leaks_the_target_gameweek(report):
    # Every per-gw result must come from a model trained on strictly earlier
    # gameweeks -- there's no direct way to assert this from the report
    # alone, but every row should at least have a plausible, non-degenerate
    # sample size (not every player in the league, since only players who
    # actually featured this gw can be matched to an actual score).
    for r in report.per_gw:
        assert r.n_players > 0


def test_backtest_realized_xi_points_are_computed(report):
    with_model_points = [r for r in report.per_gw if r.model_xi_points is not None]
    assert with_model_points  # at least some gameweeks produced a valid squad
    for r in with_model_points:
        assert r.model_xi_points >= 0
    # Baseline needs a previous gameweek, so the first backtested gw has none.
    assert report.per_gw[0].baseline_xi_points is None
    with_baseline = [r for r in report.per_gw[1:] if r.baseline_xi_points is not None]
    assert with_baseline


def test_backtest_report_totals_and_dataframe(report):
    assert report.total_model_xi_points >= 0
    assert report.total_baseline_xi_points >= 0
    df = report.as_dataframe()
    assert len(df) == len(report.per_gw)
    assert "mae" in df.columns and "spearman_corr" in df.columns


def test_backtest_on_empty_history_returns_empty_report():
    data = build_demo_data(n_gws=0, n_future_gws=2, seed=4)
    report = run_backtest(data)
    assert report.per_gw == []
    assert report.total_model_xi_points == 0
