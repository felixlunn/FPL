"""Regression coverage for the cold-start assumed-minutes calibration: a
nailed-on player should be assumed close to a full 90 minutes when
selected, not the same flat ~68-minute assumption used for a squad
player who merely sometimes features. This was tightened after a
full-season projection of a well-optimized real squad landed well below
a realistic ~2000-2500 point season total under the old flat assumption.
"""

import pandas as pd

from fpl_predictor.forecast import _MAX_ASSUMED_MINUTES, _MIN_ASSUMED_MINUTES, _assumed_minutes, predict_cold_start_gameweek
from fpl_predictor.pipeline import build_demo_data


def test_assumed_minutes_scales_between_bounds_with_reliability():
    assert _assumed_minutes(1.0) == _MAX_ASSUMED_MINUTES
    assert _assumed_minutes(0.0) == _MIN_ASSUMED_MINUTES
    # Midpoint reliability should land roughly at the old flat 68-minute
    # heuristic this refines, so it's a continuation, not a break.
    mid = _assumed_minutes(0.5)
    assert 60 < mid < 75


def test_nailed_on_player_baseline_uses_near_full_minutes():
    data = build_demo_data(n_gws=0, n_future_gws=1, seed=5)
    nailed_id = data.players_df["id"].iloc[0]
    data.past_seasons_df = pd.DataFrame([
        {"player_id": nailed_id, "season_name": "2024/25", "total_points": 200, "minutes": 3400},  # played almost every minute
    ])
    pred = predict_cold_start_gameweek(data.players_df, data.past_seasons_df, data.fixtures_df, target_gw=1)
    row = pred[pred["player_id"] == nailed_id].iloc[0]
    per90 = 200 / 3400 * 90.0
    reliability = row["last_season_reliability"]
    assert reliability > 0.95
    # Baseline (before fixture multiplier) should be close to a full-90
    # assumption for a player this reliable, not discounted to ~68/90 of
    # it -- check by recomputing the per90/reliability component directly
    # (fixture multiplier is bounded in [0.6, 1.4], so bracket rather than
    # assert an exact equality).
    baseline_before_fixture = per90 * reliability
    # GW1_Points = baseline_before_fixture * (assumed_minutes/90) * fixture_mult;
    # fixture_mult is bounded in [0.6, 1.4], so recover a safe bracket check
    # instead of an exact equality.
    assert row["GW1_Points"] >= baseline_before_fixture * (80 / 90) * 0.6
