"""Smoke test for the ablation harness -- fast/small since each variant
retrains its own full backtest; the interesting numbers (see
fpl_predictor.ablation.__main__) are for human inspection, not asserted
on exactly here, since demo-data noise means exact values aren't stable.
"""

import pytest

from fpl_predictor.ablation import VARIANTS, run_ablation_study
from fpl_predictor.pipeline import build_demo_data


def test_ablation_study_runs_every_variant_and_returns_a_comparison_table():
    data = build_demo_data(n_gws=6, n_future_gws=1, seed=9)
    result = run_ablation_study(data, min_train_gws=2, tune=False)
    assert len(result) == len(VARIANTS)
    assert set(result["variant"]) == set(VARIANTS.keys())
    for col in ["overall_mae", "mean_spearman", "model_xi_points", "vs_full_points"]:
        assert col in result.columns
    full_row = result[result["variant"] == "full (current)"].iloc[0]
    assert full_row["vs_full_points"] == 0.0


def test_ablation_study_handles_no_backtestable_gameweeks():
    data = build_demo_data(n_gws=1, n_future_gws=1, seed=9)  # too little history to backtest
    result = run_ablation_study(data, min_train_gws=3, tune=False)
    assert (result["n_gws"] == 0).all()
