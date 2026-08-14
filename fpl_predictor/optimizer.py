"""Squad optimization: turns per-player point predictions into concrete
FPL decisions -- the optimal 15-man squad under budget/position/team
constraints, the best valid starting XI + captain for a given gameweek,
and the best set of transfers from an existing squad.

All selection problems are solved exactly as integer linear programs with
PuLP/CBC rather than greedy heuristics, so results are provably optimal
given the model's predictions (the model's accuracy is the only remaining
source of error, not the optimizer).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, PULP_CBC_CMD, lpSum

# PULP_CBC_CMD bundles its own CBC binary so the app works out of the box
# with no external solver install; pulp>=3.3 flags it deprecated in favour
# of COIN_CMD, which requires that separate install, so we keep using the
# bundled solver and just silence the (harmless) deprecation notice.
warnings.filterwarnings("ignore", message=".*PULP_CBC_CMD is deprecated.*")
warnings.filterwarnings("ignore", message=".*LpVariable.dicts is deprecated.*")

from fpl_predictor.config import (
    MAX_PER_TEAM,
    MAX_SQUAD_COST,
    POS_COUNTS,
    POS_MAP,
    STARTING_XI_MAX,
    STARTING_XI_MIN,
    STARTING_XI_SIZE,
)

TRANSFER_HIT_COST = 4.0

# Squad selection is driven overwhelmingly by what actually plays and
# captains, not by raw sum-of-15 predicted points -- a 15th name on the
# bench that never starts shouldn't compete on equal footing with a
# starter for which 15 get picked. BENCH_WEIGHT keeps bench depth *some*
# value (so a squad isn't left with unplayable bench fodder, and autosubs
# still matter) without letting it distort selection the way an unweighted
# sum-of-15 objective used to -- that was the real bug behind "substitute
# suggestions don't really work": a squad could win on paper by stacking a
# position's bench instead of upgrading a starter, since only 11 of the 15
# can ever actually play.
BENCH_WEIGHT = 0.1

# A gentle *positive* nudge toward heavily-owned players when the model is
# otherwise close to indifferent -- heavy ownership is itself weak evidence
# (many independent analyses converging on the same player) and protects
# rank against the template. The Differential transfer strategy overrides
# this with a larger *negative* weight instead, deliberately hunting the
# opposite signal. Applied to squad/XI *membership* selection only -- it
# never changes the predicted-points values themselves, just which
# similarly-rated player gets chosen.
DEFAULT_OWNERSHIP_WEIGHT = 0.015
DIFFERENTIAL_OWNERSHIP_WEIGHT = -0.04


def _solve_squad_ilp(
    pool_df: pd.DataFrame,
    points_col: str,
    budget: float = MAX_SQUAD_COST,
    must_include_ids: set | None = None,
    must_exclude_ids: set | None = None,
    current_ids: set | None = None,
    transfers_equal: int | None = None,
    ownership_col: str | None = "selected_by_percent",
    ownership_weight: float = DEFAULT_OWNERSHIP_WEIGHT,
    bench_weight: float = BENCH_WEIGHT,
) -> list | None:
    """Core squad ILP -- jointly selects the 15-man squad *and* its best
    starting XI + captain in one solve, so which 15 get picked is actually
    driven by what would improve the starting XI (and who'd captain it),
    not by raw sum-of-15 predicted points. A squad can only ever field 11,
    so maximizing all 15 equally can waste budget on bench depth instead of
    upgrading a starter -- this fixes that by only counting the other 4 at
    ``bench_weight`` of a starter's value, with the captain's points
    counted twice, matching how a real gameweek actually scores.

    ``ownership_weight`` blends ``ownership_col`` (selected-by %) into the
    score used to *choose between* otherwise-similar players when picking
    who's in vs. out -- it never changes the points values used for actual
    predicted-points totals.

    Returns the list of selected 15 player_ids, or None if infeasible.
    """
    df = pool_df.drop_duplicates(subset=["player_id"]).set_index("player_id", drop=False)
    if must_exclude_ids:
        df = df[~df.index.isin(must_exclude_ids)]
    keys = df.index.tolist()
    if not keys:
        return None

    score = df[points_col].astype(float)
    if ownership_weight and ownership_col and ownership_col in df.columns:
        ownership_pct = pd.to_numeric(df[ownership_col], errors="coerce").fillna(0.0)
        score = score + ownership_pct * ownership_weight

    prob = LpProblem("fpl_squad", LpMaximize)
    x = LpVariable.dicts("select", keys, 0, 1, cat="Binary")   # in the 15-man squad
    s = LpVariable.dicts("start", keys, 0, 1, cat="Binary")    # in the starting XI
    c = LpVariable.dicts("captain", keys, 0, 1, cat="Binary")  # captained (counts a 2nd time)

    prob += (
        lpSum(score[k] * s[k] for k in keys)
        + lpSum(score[k] * c[k] for k in keys)
        + bench_weight * lpSum(score[k] * (x[k] - s[k]) for k in keys)
    )
    prob += lpSum(x[k] for k in keys) == 15
    prob += lpSum(df.loc[k, "now_cost"] * x[k] for k in keys) <= budget

    for pos_id, count in POS_COUNTS.items():
        pos_keys = [k for k in keys if df.loc[k, "element_type"] == pos_id]
        prob += lpSum(x[k] for k in pos_keys) == count
        prob += lpSum(s[k] for k in pos_keys) >= STARTING_XI_MIN[pos_id]
        prob += lpSum(s[k] for k in pos_keys) <= STARTING_XI_MAX[pos_id]

    for team_name in df["team_name"].unique():
        team_keys = [k for k in keys if df.loc[k, "team_name"] == team_name]
        prob += lpSum(x[k] for k in team_keys) <= MAX_PER_TEAM

    for k in keys:
        prob += s[k] <= x[k]
        prob += c[k] <= s[k]
    prob += lpSum(s[k] for k in keys) == STARTING_XI_SIZE
    prob += lpSum(c[k] for k in keys) == 1

    if must_include_ids:
        for pid in must_include_ids:
            if pid in x:
                prob += x[pid] == 1

    if transfers_equal is not None and current_ids is not None:
        new_player_keys = [k for k in keys if k not in current_ids]
        prob += lpSum(x[k] for k in new_player_keys) == transfers_equal

    prob.solve(PULP_CBC_CMD(msg=0))
    if prob.status != 1:
        return None
    return [k for k in keys if x[k].value() == 1.0]


def find_optimal_squad(
    pred_df: pd.DataFrame,
    budget: float = MAX_SQUAD_COST,
    points_col: str = "pred_points_total_adj",
    ownership_col: str | None = "selected_by_percent",
    ownership_weight: float = DEFAULT_OWNERSHIP_WEIGHT,
) -> pd.DataFrame:
    """Best possible 15-man squad under budget/position/team constraints,
    ignoring any squad the user currently owns -- selected jointly with its
    best starting XI + captain (see ``_solve_squad_ilp``) so the squad
    itself is chosen for what it can actually field, not raw sum-of-15
    points."""
    ids = _solve_squad_ilp(pred_df, points_col, budget=budget, ownership_col=ownership_col, ownership_weight=ownership_weight)
    if not ids:
        return pd.DataFrame()
    return pred_df[pred_df["player_id"].isin(ids)].drop_duplicates(subset=["player_id"]).reset_index(drop=True)


def build_starting_xi(squad_df: pd.DataFrame, gw_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None, pd.Series | None]:
    """Exactly-optimal starting XI (+ bench, captain, vice-captain) for a
    single gameweek, solved as a small ILP so formation constraints
    (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD, 11 total) are always respected --
    unlike a greedy "top N per position" pick, which can produce an invalid
    or suboptimal formation.
    """
    empty_series = pd.Series({"web_name": "N/A", gw_col: 0.0})
    if squad_df.empty or gw_col not in squad_df.columns:
        return pd.DataFrame(), pd.DataFrame(), empty_series, empty_series

    df = squad_df.drop_duplicates(subset=["player_id"]).set_index("player_id", drop=False)
    keys = df.index.tolist()
    prob = LpProblem("starting_xi", LpMaximize)
    x = LpVariable.dicts("start", keys, 0, 1, cat="Binary")

    prob += lpSum(df.loc[k, gw_col] * x[k] for k in keys)
    prob += lpSum(x[k] for k in keys) == STARTING_XI_SIZE
    for pos_id in POS_MAP:
        pos_keys = [k for k in keys if df.loc[k, "element_type"] == pos_id]
        prob += lpSum(x[k] for k in pos_keys) >= STARTING_XI_MIN[pos_id]
        prob += lpSum(x[k] for k in pos_keys) <= STARTING_XI_MAX[pos_id]

    prob.solve(PULP_CBC_CMD(msg=0))
    if prob.status != 1:
        # Fall back to highest-points-first if somehow infeasible (shouldn't
        # happen for a valid 15-man squad with standard position counts).
        lineup_ids = df.nlargest(STARTING_XI_SIZE, gw_col).index.tolist()
    else:
        lineup_ids = [k for k in keys if x[k].value() == 1.0]

    lineup = df.loc[lineup_ids].sort_values(gw_col, ascending=False)
    bench = df.drop(lineup_ids).sort_values(gw_col, ascending=False)

    captain = lineup.iloc[0] if not lineup.empty else empty_series
    vice = lineup.iloc[1] if len(lineup) > 1 else captain
    return lineup, bench, captain, vice


def _squad_objective_value(
    squad_df: pd.DataFrame,
    points_col: str,
    ownership_col: str | None,
    ownership_weight: float,
    bench_weight: float,
) -> float:
    """Re-derives the same XI + captain + bench-weighted score
    ``_solve_squad_ilp`` optimizes for, on a concrete squad -- used to
    compare candidate plans across different transfer counts on a
    consistent basis (raw sum-of-15 points is no longer what's actually
    being optimized, so it's not a fair comparison metric between plans).
    """
    lineup, bench, captain, vice = build_starting_xi(squad_df, points_col)

    def _scored_sum(df_: pd.DataFrame) -> float:
        if df_.empty:
            return 0.0
        pts = df_[points_col].astype(float)
        if ownership_weight and ownership_col and ownership_col in df_.columns:
            pts = pts + pd.to_numeric(df_[ownership_col], errors="coerce").fillna(0.0) * ownership_weight
        return float(pts.sum())

    captain_bonus = 0.0
    if isinstance(captain, pd.Series) and points_col in captain.index:
        captain_bonus = float(captain[points_col])
        if ownership_weight and ownership_col and ownership_col in captain.index:
            owned = pd.to_numeric(pd.Series([captain[ownership_col]]), errors="coerce").fillna(0.0).iloc[0]
            captain_bonus += float(owned) * ownership_weight

    return _scored_sum(lineup) + captain_bonus + bench_weight * _scored_sum(bench)


@dataclass
class TransferPlan:
    squad_df: pd.DataFrame
    transfers_out: list
    transfers_in: list
    n_transfers: int
    hit_cost: float
    raw_points: float
    net_points: float
    strategy: str = "Optimal"
    rationale: str = ""


def _best_plan_over_transfer_counts(
    pool: pd.DataFrame,
    current_squad_df: pd.DataFrame,
    current_ids: set,
    points_col: str,
    budget: float,
    free_transfers: int,
    transfer_counts,
    ownership_col: str | None = "selected_by_percent",
    ownership_weight: float = DEFAULT_OWNERSHIP_WEIGHT,
    bench_weight: float = BENCH_WEIGHT,
) -> TransferPlan | None:
    """Search the given transfer counts and return the plan with the best
    net *objective value* -- the same XI + captain + bench-weighted score
    the ILP itself optimizes for (see ``_squad_objective_value``), not raw
    sum-of-15 points, so "best plan" actually means "most improves what you
    can field", consistent with what was searched for.
    """
    best_score, best_plan = None, None
    for t in transfer_counts:
        ids = _solve_squad_ilp(
            pool, points_col, budget=budget, current_ids=current_ids, transfers_equal=t,
            ownership_col=ownership_col, ownership_weight=ownership_weight, bench_weight=bench_weight,
        )
        if not ids:
            continue
        squad = pool[pool["player_id"].isin(ids)].drop_duplicates(subset=["player_id"])
        raw_points = float(squad[points_col].sum())
        hit_cost = TRANSFER_HIT_COST * max(0, t - free_transfers)
        objective_value = _squad_objective_value(squad, points_col, ownership_col, ownership_weight, bench_weight)
        net_score = objective_value - hit_cost
        if best_score is None or net_score > best_score:
            out_ids = current_ids - set(ids)
            in_ids = set(ids) - current_ids
            best_score = net_score
            best_plan = TransferPlan(
                squad_df=squad.reset_index(drop=True),
                transfers_out=current_squad_df[current_squad_df["player_id"].isin(out_ids)]["web_name"].tolist(),
                transfers_in=squad[squad["player_id"].isin(in_ids)]["web_name"].tolist(),
                n_transfers=t,
                hit_cost=hit_cost,
                raw_points=raw_points,
                net_points=raw_points - hit_cost,
            )
    return best_plan


def suggest_transfers(
    current_squad_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    free_transfers: int = 1,
    max_transfers_considered: int = 3,
    budget: float = MAX_SQUAD_COST,
    points_col: str = "pred_points_total_adj",
) -> TransferPlan | None:
    """Back-compat single-plan helper: the best plan (by starting-XI +
    captain value, see above) across 0..max_transfers_considered transfers
    -- only takes a hit when the model expects it to pay off. See
    ``suggest_transfer_options`` for multiple differently-motivated plans
    side by side.
    """
    current_ids = set(current_squad_df["player_id"])
    pool = pd.concat([pool_df, current_squad_df]).drop_duplicates(subset=["player_id"])
    plan = _best_plan_over_transfer_counts(pool, current_squad_df, current_ids, points_col, budget, free_transfers, range(0, max_transfers_considered + 1))
    if plan is not None:
        plan.strategy = "Optimal"
    return plan


def suggest_transfer_options(
    current_squad_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    free_transfers: int = 1,
    max_transfers_considered: int = 3,
    budget: float = MAX_SQUAD_COST,
    points_col: str = "pred_points_total_adj",
    ownership_col: str = "selected_by_percent",
) -> dict[str, TransferPlan]:
    """Multiple transfer plans, each optimizing for something different, so
    a manager isn't just handed one number -- they're shown the tradeoffs:

    - **Optimal**: best net starting-XI value (captain counted double),
      hits included if they pay off, with a small nudge toward
      well-owned players when otherwise close.
    - **Safe (no hit)**: best plan using only the free transfer(s)
      available -- never risks a -4.
    - **Differential**: biases toward lower-ownership players -- useful
      for climbing rank against a mini-league full of the same template
      picks, which raw expected-points maximization alone doesn't capture.

    Any strategy whose search comes back infeasible (e.g. no valid 0-hit
    squad exists) is simply omitted from the result rather than erroring.
    """
    current_ids = set(current_squad_df["player_id"])
    pool = pd.concat([pool_df, current_squad_df]).drop_duplicates(subset=["player_id"])
    all_counts = list(range(0, max_transfers_considered + 1))
    plans: dict[str, TransferPlan] = {}

    optimal = _best_plan_over_transfer_counts(pool, current_squad_df, current_ids, points_col, budget, free_transfers, all_counts, ownership_col=ownership_col, ownership_weight=DEFAULT_OWNERSHIP_WEIGHT)
    if optimal is not None:
        optimal.strategy = "Optimal"
        optimal.rationale = (
            f"Best expected net value for your starting XI (captain counted double) across up to "
            f"{max_transfers_considered} transfer(s) -- takes a hit only when the model expects it to pay off."
        )
        plans["optimal"] = optimal

    safe_counts = [t for t in all_counts if t <= free_transfers]
    safe = _best_plan_over_transfer_counts(pool, current_squad_df, current_ids, points_col, budget, free_transfers, safe_counts, ownership_col=ownership_col, ownership_weight=DEFAULT_OWNERSHIP_WEIGHT)
    if safe is not None:
        safe.strategy = "Safe (no hit)"
        safe.rationale = f"Best plan using only your {free_transfers} free transfer(s) -- never takes a -4 hit."
        plans["safe"] = safe

    differential = _best_plan_over_transfer_counts(pool, current_squad_df, current_ids, points_col, budget, free_transfers, all_counts, ownership_col=ownership_col, ownership_weight=DIFFERENTIAL_OWNERSHIP_WEIGHT)
    if differential is not None:
        differential.strategy = "Differential"
        differential.rationale = (
            "Leans toward lower-ownership players with strong predicted points -- for climbing rank "
            "against the template rather than just maximizing raw expected points."
        )
        plans["differential"] = differential

    return plans


def validate_squad(squad_df: pd.DataFrame, budget: float = MAX_SQUAD_COST) -> dict:
    """Check a (possibly user-supplied) 15-man squad against all FPL rules.
    Returns a dict with a boolean ``valid`` plus human-readable issue lists
    so a web UI can show exactly what's wrong.
    """
    issues = []
    pos_counts = squad_df["element_type"].value_counts().to_dict()
    for pos_id, required in POS_COUNTS.items():
        have = pos_counts.get(pos_id, 0)
        if have != required:
            issues.append(f"{POS_MAP[pos_id]}: {have}/{required} required")

    team_counts = squad_df["team_name"].value_counts().to_dict()
    for team, count in team_counts.items():
        if count > MAX_PER_TEAM:
            issues.append(f"{team}: {count} players (max {MAX_PER_TEAM})")

    total_cost = float(squad_df["now_cost"].sum())
    if total_cost > budget:
        issues.append(f"Budget exceeded: £{total_cost:.1f}m > £{budget:.1f}m")

    if squad_df["player_id"].duplicated().any():
        issues.append("Duplicate players selected")

    if len(squad_df) != 15:
        issues.append(f"Squad has {len(squad_df)} players, need 15")

    return {"valid": len(issues) == 0, "issues": issues, "total_cost": total_cost}
