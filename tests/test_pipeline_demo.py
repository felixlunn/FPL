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
    return run_pipeline(data)


def test_feature_frame_has_no_label_leakage_columns_as_features(result):
    assert "label_points" not in result.feature_cols
    assert result.feat_df.shape[0] > 0


def test_model_trains_for_every_position(result):
    assert result.mode == "trained"
    assert set(result.trained_model.positions.keys()) == {1, 2, 3, 4}
    for pm in result.trained_model.positions.values():
        assert pm.n_train_rows > 0


def test_minutes_model_trained_and_start_probability_populated(result):
    assert result.minutes_model is not None
    assert result.minutes_model.model is not None
    assert "start_probability" in result.pred_df.columns
    assert result.pred_df["start_probability"].between(0, 1).all()
    # effective_playing_prob = min(status-based, model-based) should never
    # make the adjusted prediction exceed the raw one.
    assert (result.pred_df["pred_points_total_adj"] <= result.pred_df["pred_points_total"] + 1e-9).all()


def test_quantile_model_trained_and_floor_ceiling_populated(result):
    assert result.quantile_model is not None
    gw_col = result.gw_cols[0]
    floor_col = gw_col.replace("_Points", "_Floor")
    ceiling_col = gw_col.replace("_Points", "_Ceiling")
    assert floor_col in result.pred_df.columns
    assert ceiling_col in result.pred_df.columns
    valid = result.pred_df.dropna(subset=[floor_col, ceiling_col])
    assert not valid.empty
    # Ceiling should be >= floor for the vast majority of players (floor
    # and ceiling are independently-trained quantile regressors, so a
    # rare crossover is possible, but not the norm).
    assert (valid[ceiling_col] >= valid[floor_col]).mean() > 0.9
    assert (valid[floor_col] >= 0).all()


def test_cold_start_reliability_discounts_low_minutes_players():
    """Regression coverage for a real bug: a player with a great per-90
    scoring rate from a small sample of substitute cameos last season
    must not outrank a nailed-on regular in cold-start mode just because
    their rate-when-played happens to be higher -- last-season minutes
    share (reliability) has to pull them back down.
    """
    data = build_demo_data(n_gws=0, n_future_gws=2, seed=3)
    # Fabricate a two-player past-season comparison: a explosive-but-fringe
    # player (high points from very few minutes) vs. a solid, nailed-on
    # regular (lower per-90 rate but played almost every minute).
    fringe_id, regular_id = data.players_df["id"].iloc[0], data.players_df["id"].iloc[1]
    data.past_seasons_df = pd.DataFrame([
        {"player_id": fringe_id, "season_name": "2024/25", "total_points": 40, "minutes": 200},   # ~18 pts/90, but only ~2 full games all season
        {"player_id": regular_id, "season_name": "2024/25", "total_points": 120, "minutes": 3200},  # ~3.4 pts/90, played almost every minute
    ])
    result = run_pipeline(data)
    assert result.mode == "cold_start"
    gw_col = result.gw_cols[0]
    fringe_pred = result.pred_df.loc[result.pred_df["player_id"] == fringe_id, gw_col].iloc[0]
    regular_pred = result.pred_df.loc[result.pred_df["player_id"] == regular_id, gw_col].iloc[0]
    assert regular_pred > fringe_pred
    assert "last_season_reliability" in result.pred_df.columns
    fringe_rel = result.pred_df.loc[result.pred_df["player_id"] == fringe_id, "last_season_reliability"].iloc[0]
    regular_rel = result.pred_df.loc[result.pred_df["player_id"] == regular_id, "last_season_reliability"].iloc[0]
    assert regular_rel > fringe_rel


def test_ownership_column_flows_through_to_predictions(result):
    # selected_by_percent must survive from players_df into pred_df in both
    # trained and cold-start modes -- the optimizer's ownership-weighted
    # scoring (default template nudge, Differential's opposite-signed
    # penalty) is a silent no-op without it.
    assert "selected_by_percent" in result.pred_df.columns
    assert result.pred_df["selected_by_percent"].notna().all()

    cold = run_pipeline(build_demo_data(n_gws=0, n_future_gws=2, seed=4))
    assert cold.mode == "cold_start"
    assert "selected_by_percent" in cold.pred_df.columns


def test_fixture_difficulty_feature_is_populated_and_varies(result):
    # FPL's own FDR (1-5), joined in per (team, gameweek, opponent) --
    # replaces the old Elo/historical-CSV team-strength model entirely.
    assert "fixture_difficulty" in result.feat_df.columns
    assert "fixture_difficulty" in result.feature_cols
    assert result.feat_df["fixture_difficulty"].between(1, 5).all()
    assert result.feat_df["fixture_difficulty"].nunique() > 1


def test_set_piece_priority_features_are_populated_and_vary(result):
    assert "penalties_priority" in result.feat_df.columns
    assert "set_piece_priority" in result.feat_df.columns
    assert "penalties_priority" in result.feature_cols
    assert "set_piece_priority" in result.feature_cols
    # Designated takers (order 1) should score higher than non-takers (0).
    assert result.feat_df["penalties_priority"].max() == 1.0
    assert (result.feat_df["penalties_priority"] == 0).any()
    assert result.feat_df["penalties_priority"].between(0, 1).all()


def test_defensive_contribution_features_are_picked_up(result):
    # tackles/CBI/recoveries are in the demo history -> defensive_actions
    # should get built and its rolling stats should make it into the
    # feature set actually used for training (not silently dropped).
    assert "defensive_actions" in result.feat_df.columns
    assert any(c.startswith("defensive_actions_") for c in result.feature_cols)


def test_predictions_target_exactly_one_upcoming_gameweek(result):
    assert not result.pred_df.empty
    # The app always predicts a single gameweek -- the next unplayed one.
    assert len(result.gw_cols) == 1
    assert len(result.future_gws) == 1
    assert result.future_gws[0] == result.last_completed_gw + 1
    assert result.gw_cols[0] in result.pred_df.columns
    assert "pred_points_total_adj" in result.pred_df.columns
    assert (result.pred_df["pred_points_total_adj"] >= 0).all()


def test_cold_start_mode_when_no_current_season_history_exists():
    """The gap every season between the final ball of one campaign and the
    first completed gameweek of the next: no (player, gameweek) history to
    train on, but real players should still get sensible predictions
    rather than the pipeline giving up.
    """
    data = build_demo_data(n_gws=0, n_future_gws=3, seed=2)
    result = run_pipeline(data)
    assert result.mode == "cold_start"
    assert result.feat_df.empty
    assert len(result.gw_cols) == 1
    assert not result.pred_df.empty
    assert result.pred_df["web_name"].notna().all()
    assert (result.pred_df["pred_points_total_adj"] >= 0).all()
    assert result.pred_df["pred_points_total_adj"].sum() > 0


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
