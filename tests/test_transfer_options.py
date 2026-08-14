"""Unit tests for the multi-strategy transfer suggestions
(``suggest_transfer_options``), using small hand-built pools rather than
the full pipeline so the Differential strategy's ownership-driven
behaviour is deterministic and easy to reason about (the synthetic demo
dataset gives every player the same flat ownership, which wouldn't
exercise this at all).
"""

import pandas as pd

from fpl_predictor.optimizer import suggest_transfer_options, validate_squad


def _player(pid, pos, team, cost=4.5, pts=4.0, owned=5.0, name=None):
    return {
        "player_id": pid, "web_name": name or f"P{pid}", "team_name": team,
        "element_type": pos, "now_cost": cost, "pred_points_total_adj": pts,
        "selected_by_percent": owned,
    }


def _base_squad_and_pool():
    """A valid 15-man current squad (2 GKP, 5 DEF, 5 MID, 3 FWD) spread
    across enough teams to respect the max-3-per-team rule, plus a
    transfer-market pool offering two extra MID candidates: "A" (higher
    raw points, heavily owned) and "B" (almost identical points, barely
    owned) -- so Optimal should prefer A and Differential should be
    steerable toward B once the ownership penalty is applied.
    """
    teams = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
    squad_rows = []
    pid = 1
    # 2 GKP, 5 DEF, 3 FWD: unremarkable fillers, spread across teams.
    for pos, n in ((1, 2), (2, 5), (4, 3)):
        for i in range(n):
            squad_rows.append(_player(pid, pos, teams[pid % len(teams)], pts=3.0 + 0.1 * i))
            pid += 1
    # 5 current MIDs, the weakest of which (lowest points) is the one a
    # transfer should realistically replace.
    weak_mid_id = pid
    squad_rows.append(_player(pid, 3, teams[pid % len(teams)], pts=2.0, name="WeakMid"))
    pid += 1
    for i in range(4):
        squad_rows.append(_player(pid, 3, teams[pid % len(teams)], pts=4.0 + 0.1 * i))
        pid += 1
    current_squad_df = pd.DataFrame(squad_rows)

    a_id, b_id = pid, pid + 1
    pool_rows = [
        _player(a_id, 3, "T1", pts=6.0, owned=80.0, name="HighOwnedA"),
        _player(b_id, 3, "T2", pts=5.95, owned=1.0, name="LowOwnedB"),
    ]
    pool_df = pd.DataFrame(pool_rows)
    return current_squad_df, pool_df, weak_mid_id, a_id, b_id


def test_suggest_transfer_options_returns_labeled_strategies_with_correct_hit_costs():
    current_squad_df, pool_df, *_ = _base_squad_and_pool()
    plans = suggest_transfer_options(current_squad_df, pool_df, free_transfers=1, max_transfers_considered=2)

    assert "optimal" in plans and "safe" in plans and "differential" in plans
    for plan in plans.values():
        assert plan.strategy and plan.rationale
        v = validate_squad(plan.squad_df)
        assert v["valid"], v["issues"]
        assert plan.hit_cost == max(0, plan.n_transfers - 1) * 4.0
        assert plan.net_points == plan.raw_points - plan.hit_cost

    # Safe strategy is capped at the free transfer count -- it can never
    # recommend a hit.
    assert plans["safe"].n_transfers <= 1
    assert plans["safe"].hit_cost == 0.0


def test_differential_strategy_prefers_the_lower_owned_near_equal_player():
    current_squad_df, pool_df, weak_mid_id, a_id, b_id = _base_squad_and_pool()
    plans = suggest_transfer_options(current_squad_df, pool_df, free_transfers=1, max_transfers_considered=1)

    optimal_in = set(plans["optimal"].squad_df["player_id"])
    differential_in = set(plans["differential"].squad_df["player_id"])

    # Optimal (pure points) should bring in the higher-raw-points, heavily
    # owned player over the near-identical, barely-owned one.
    assert a_id in optimal_in
    assert b_id not in optimal_in

    # Differential's ownership penalty (0.04/%) makes B's adjusted score
    # higher than A's (6.0 - 0.04*80 = 2.8 vs. 5.95 - 0.04*1 = 5.91), so it
    # should bring in B instead.
    assert b_id in differential_in
    assert a_id not in differential_in


def test_suggest_transfer_options_omits_strategies_that_are_infeasible_without_error():
    # free_transfers=0 with max_transfers_considered=0 -- "safe" degenerates
    # to "keep the current squad", which must still be a valid, returned plan.
    current_squad_df, pool_df, *_ = _base_squad_and_pool()
    plans = suggest_transfer_options(current_squad_df, pool_df, free_transfers=0, max_transfers_considered=0)
    assert "safe" in plans
    assert plans["safe"].n_transfers == 0
    assert set(plans["safe"].squad_df["player_id"]) == set(current_squad_df["player_id"])
