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


def _solve_squad_ilp(
    pool_df: pd.DataFrame,
    points_col: str,
    budget: float = MAX_SQUAD_COST,
    must_include_ids: set | None = None,
    must_exclude_ids: set | None = None,
    current_ids: set | None = None,
    transfers_equal: int | None = None,
) -> list | None:
    """Core 15-man squad ILP. Returns the list of selected player_ids, or
    None if the constraints are infeasible (e.g. budget too tight)."""
    df = pool_df.drop_duplicates(subset=["player_id"]).set_index("player_id", drop=False)
    if must_exclude_ids:
        df = df[~df.index.isin(must_exclude_ids)]
    keys = df.index.tolist()
    if not keys:
        return None

    prob = LpProblem("fpl_squad", LpMaximize)
    x = LpVariable.dicts("select", keys, 0, 1, cat="Binary")

    prob += lpSum(df.loc[k, points_col] * x[k] for k in keys)
    prob += lpSum(x[k] for k in keys) == 15
    prob += lpSum(df.loc[k, "now_cost"] * x[k] for k in keys) <= budget

    for pos_id, count in POS_COUNTS.items():
        pos_keys = [k for k in keys if df.loc[k, "element_type"] == pos_id]
        prob += lpSum(x[k] for k in pos_keys) == count

    for team_name in df["team_name"].unique():
        team_keys = [k for k in keys if df.loc[k, "team_name"] == team_name]
        prob += lpSum(x[k] for k in team_keys) <= MAX_PER_TEAM

    if must_include_ids:
        for pid in must_include_ids:
            if pid in x:
                prob += x[pid] == 1

    if transfers_equal is not None and current_ids is not None:
        new_player_keys = [k for k in keys if k not in current_ids]
        prob += lpSum(x[k] for k in new_player_keys) == transfers_equal

    status = prob.solve(PULP_CBC_CMD(msg=0))
    if prob.status != 1:
        return None
    return [k for k in keys if x[k].value() == 1.0]


def find_optimal_squad(pred_df: pd.DataFrame, budget: float = MAX_SQUAD_COST, points_col: str = "pred_points_total_adj") -> pd.DataFrame:
    """Best possible 15-man squad under budget/position/team constraints,
    ignoring any squad the user currently owns."""
    ids = _solve_squad_ilp(pred_df, points_col, budget=budget)
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


# How many predicted points a transfer plan "gives up" per percentage point
# of ownership on a player it brings in, when searching for the Differential
# option -- makes the ILP trade a little raw expected-points for players far
# fewer rivals also own, which is what actually moves your rank against a
# mini-league full of the same template picks.
DIFFERENTIAL_OWNERSHIP_WEIGHT = 0.04


def _best_plan_over_transfer_counts(
    pool: pd.DataFrame,
    current_squad_df: pd.DataFrame,
    current_ids: set,
    points_col: str,
    budget: float,
    free_transfers: int,
    transfer_counts,
    score_col: str | None = None,
) -> TransferPlan | None:
    """Search the given transfer counts and return the plan with the best
    net *score* (``score_col`` if given, e.g. an ownership-adjusted column
    for the Differential strategy -- else ``points_col`` itself). Raw/net
    points on the returned plan are always the true predicted-points totals
    regardless of what ``score_col`` was used to pick between candidates.
    """
    score_col = score_col or points_col
    best_score, best_plan = None, None
    for t in transfer_counts:
        ids = _solve_squad_ilp(pool, score_col, budget=budget, current_ids=current_ids, transfers_equal=t)
        if not ids:
            continue
        squad = pool[pool["player_id"].isin(ids)].drop_duplicates(subset=["player_id"])
        raw_points = float(squad[points_col].sum())
        hit_cost = TRANSFER_HIT_COST * max(0, t - free_transfers)
        net_score = float(squad[score_col].sum()) - hit_cost
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
    """Back-compat single-plan helper: the best *net* predicted-points plan
    across 0..max_transfers_considered transfers (only takes a hit when the
    model expects it to pay off). See ``suggest_transfer_options`` for
    multiple differently-motivated plans side by side.
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

    - **Optimal**: best net predicted points, hits included if they pay off.
    - **Safe (no hit)**: best plan using only the free transfer(s) available
      -- never risks a -4.
    - **Differential**: biases toward lower-ownership players (via
      ``ownership_col``, when present in the pool) -- useful for climbing
      rank against a mini-league full of the same template picks, which
      raw expected-points maximization alone doesn't capture.

    Any strategy whose search comes back infeasible (e.g. no valid 0-hit
    squad exists) is simply omitted from the result rather than erroring.
    """
    current_ids = set(current_squad_df["player_id"])
    pool = pd.concat([pool_df, current_squad_df]).drop_duplicates(subset=["player_id"])
    all_counts = list(range(0, max_transfers_considered + 1))
    plans: dict[str, TransferPlan] = {}

    optimal = _best_plan_over_transfer_counts(pool, current_squad_df, current_ids, points_col, budget, free_transfers, all_counts)
    if optimal is not None:
        optimal.strategy = "Optimal"
        optimal.rationale = (
            f"Best expected net points across up to {max_transfers_considered} transfer(s) -- "
            "takes a hit only when the model expects it to pay off."
        )
        plans["optimal"] = optimal

    safe_counts = [t for t in all_counts if t <= free_transfers]
    safe = _best_plan_over_transfer_counts(pool, current_squad_df, current_ids, points_col, budget, free_transfers, safe_counts)
    if safe is not None:
        safe.strategy = "Safe (no hit)"
        safe.rationale = f"Best plan using only your {free_transfers} free transfer(s) -- never takes a -4 hit."
        plans["safe"] = safe

    if ownership_col in pool.columns:
        pool_diff = pool.copy()
        ownership_pct = pd.to_numeric(pool_diff[ownership_col], errors="coerce").fillna(0.0)
        pool_diff["_differential_score"] = pool_diff[points_col] - ownership_pct * DIFFERENTIAL_OWNERSHIP_WEIGHT
        differential = _best_plan_over_transfer_counts(
            pool_diff, current_squad_df, current_ids, points_col, budget, free_transfers, all_counts, score_col="_differential_score",
        )
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
