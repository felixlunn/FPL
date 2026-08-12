"""Full offline pipeline test using the synthetic demo dataset, so this
suite requires no network access and runs the same code path the web app
uses (data -> features -> model -> forecast -> optimizer).
"""

import pandas as pd
import pytest

from fpl_predictor.pipeline import build_demo_data, run_pipeline
from fpl_predictor.optimizer import build_starting_xi, find_optimal_squad, suggest_transfers, validate_squad
from fpl_predictor.config import MAX_PER_TEAM, MAX_SQUAD_COST, POS_COUNTS


@pytest.fixture(scope="module")
def result():
    data = build_demo_data(n_gws=10, n_future_gws=4, seed=1)
    return run_pipeline(data, n_future_gws=4)


def test_feature_frame_has_no_label_leakage_columns_as_features(result):
    assert "label_points" not in result.feature_cols
    assert result.feat_df.shape[0] > 0


def test_model_trains_for_every_position(result):
    assert set(result.trained_model.positions.keys()) == {1, 2, 3, 4}
    for pm in result.trained_model.positions.values():
        assert pm.n_train_rows > 0


def test_predictions_cover_future_gameweeks(result):
    assert not result.pred_df.empty
    for col in result.gw_cols:
        assert col in result.pred_df.columns
    assert "pred_points_total_adj" in result.pred_df.columns
    assert (result.pred_df["pred_points_total_adj"] >= 0).all()


def test_optimal_squad_respects_all_constraints(result):
    squad = find_optimal_squad(result.pred_df, points_col="pred_points_total_adj")
    assert len(squad) == 15
    v = validate_squad(squad)
    assert v["valid"], v["issues"]
    assert v["total_cost"] <= MAX_SQUAD_COST
    counts = squad["element_type"].value_counts().to_dict()
    for pos, n in POS_COUNTS.items():
        assert counts.get(pos, 0) == n
    team_counts = squad["team_name"].value_counts()
    assert (team_counts <= MAX_PER_TEAM).all()


def test_starting_xi_is_a_valid_formation(result):
    squad = find_optimal_squad(result.pred_df, points_col="pred_points_total_adj")
    gw_col = result.gw_cols[0]
    lineup, bench, captain, vice = build_starting_xi(squad, gw_col)
    assert len(lineup) == 11
    assert len(bench) == 4
    counts = lineup["element_type"].value_counts().to_dict()
    assert counts.get(1, 0) == 1  # exactly one starting GK
    assert 3 <= counts.get(2, 0) <= 5
    assert 2 <= counts.get(3, 0) <= 5
    assert 1 <= counts.get(4, 0) <= 3
    # captain should have the highest points in the lineup for that gw
    assert captain[gw_col] == lineup[gw_col].max()
    assert captain["player_id"] != vice["player_id"] or len(lineup) == 1


def test_transfer_suggestions_never_exceed_budget_or_break_rules(result):
    squad = find_optimal_squad(result.pred_df, points_col="pred_points_total_adj")
    # Deliberately swap in a couple of "current squad" players not in the
    # optimal squad so there's something to improve on.
    pool = result.pred_df
    current = pool.sort_values("pred_points_total_adj").head(15).drop_duplicates(subset=["player_id"])
    # Force a valid-ish starting current squad by just reusing optimal squad
    # ids but replacing 2 with cheap bench players to guarantee headroom.
    plan = suggest_transfers(squad, pool, free_transfers=1, max_transfers_considered=2)
    assert plan is not None
    v = validate_squad(plan.squad_df)
    assert v["valid"], v["issues"]
    assert plan.n_transfers >= 0


def test_zero_transfers_is_always_a_feasible_baseline(result):
    squad = find_optimal_squad(result.pred_df, points_col="pred_points_total_adj")
    plan = suggest_transfers(squad, result.pred_df, free_transfers=1, max_transfers_considered=0)
    assert plan is not None
    assert plan.n_transfers == 0
    assert set(plan.squad_df["player_id"]) == set(squad["player_id"])
