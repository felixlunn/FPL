"""Multi-gameweek season planning: rolls the already-trained models forward
across several future gameweeks (rather than just the single next one the
main pipeline targets), so the app can answer two questions the one-week
view can't -- where's the outlook trending, and when's the right moment to
play a chip (Wildcard / Free Hit / Bench Boost / Triple Captain).

Deliberately a thin layer on top of an already-run ``PipelineResult``: it
reuses the same trained per-position/minutes/quantile models rather than
retraining, so it's cheap to add to a page that's already loaded the main
Optimal Squad view. Chip timing is inherently a judgement call, not
something with one provably-correct answer, so the recommendations below
are transparent heuristics (each with a stated reason), not another ILP --
they're meant to prompt the right week to look closer, not be followed
blindly.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fpl_predictor.config import MAX_FPL_GW
from fpl_predictor.data_sources.fpl_api import fixtures_long_by_team
from fpl_predictor.forecast import apply_playing_probability, predict_cold_start_gameweek, predict_future_gameweeks
from fpl_predictor.optimizer import build_starting_xi, find_optimal_squad

DEFAULT_HORIZON = 6


def build_multi_gw_forecast(result, n_gws: int = DEFAULT_HORIZON) -> tuple[pd.DataFrame, list[int]]:
    """Predict every gameweek from the next one out to ``n_gws`` ahead
    (capped at gameweek 38), reusing whatever models the pipeline already
    trained. Works in both "trained" mode (real per-position models) and
    "cold_start" mode (pre-season heuristic, looped per gameweek since
    ``predict_cold_start_gameweek`` is single-gameweek by design). Returns
    an empty frame if there's no target gameweek to start from at all
    (``season_over``/``no_data`` modes).
    """
    start_gw = result.future_gws[0] if result.future_gws else None
    if start_gw is None:
        return pd.DataFrame(), []
    gws = [g for g in range(start_gw, start_gw + n_gws) if g <= MAX_FPL_GW]
    if not gws:
        return pd.DataFrame(), []

    if result.mode == "trained" and not result.feat_df.empty:
        pred = predict_future_gameweeks(
            result.trained_model, result.feat_df, result.data.players_df, result.data.fixtures_df,
            gws, minutes_model=result.minutes_model, quantile_model=result.quantile_model,
        )
    elif result.mode == "cold_start":
        merged = None
        keep_cols = ["player_id", "web_name", "team_name", "element_type", "now_cost", "last_season_reliability", "selected_by_percent"]
        for gw in gws:
            single = predict_cold_start_gameweek(result.data.players_df, result.data.past_seasons_df, result.data.fixtures_df, gw)
            if single.empty:
                continue
            gw_col = f"GW{gw}_Points"
            piece = single[[c for c in keep_cols if c in single.columns] + [gw_col]]
            merged = piece if merged is None else merged.merge(piece.drop(columns=[c for c in keep_cols if c != "player_id"]), on="player_id", how="outer")
        if merged is None:
            return pd.DataFrame(), []
        pred = merged
    else:
        return pd.DataFrame(), []

    gw_cols = [f"GW{g}_Points" for g in gws]
    if pred.empty:
        return pred, gws
    for c in gw_cols:
        if c not in pred.columns:
            pred[c] = 0.0
        pred[c] = pred[c].fillna(0.0)
    pred["pred_points_total"] = pred[gw_cols].sum(axis=1)
    pred = apply_playing_probability(pred, result.data.players_df, gw_cols)
    return pred, gws


def blank_double_gw_summary(fixtures_df: pd.DataFrame, teams_df: pd.DataFrame, gws: list[int]) -> pd.DataFrame:
    """Per gameweek in ``gws``: how many teams have no fixture (blank) and
    how many play twice (double) -- the two fixture-calendar events that
    actually drive Free Hit / Bench Boost timing, as opposed to routine
    week-to-week difficulty.
    """
    long_df = fixtures_long_by_team(fixtures_df)
    has_teams = teams_df is not None and not teams_df.empty
    # "Every team that should have a fixture" -- prefer the real team list;
    # fall back to whatever team ids actually show up in the fixture list
    # itself (e.g. the synthetic demo data, which has no separate teams_df
    # lookup wired through here) so this degrades gracefully rather than
    # reporting every team as blank.
    universe = set(teams_df["id"]) if has_teams else (set(long_df["team"].unique()) if not long_df.empty else set())
    team_name = dict(zip(teams_df["id"], teams_df["name"])) if has_teams else {}
    rows = []
    for gw in gws:
        this_gw = long_df[long_df["event"] == gw] if not long_df.empty else long_df
        counts = this_gw.groupby("team").size() if not this_gw.empty else pd.Series(dtype=int)
        teams_this_gw = set(counts.index)
        blank_teams = sorted(universe - teams_this_gw)
        double_teams = sorted(counts[counts >= 2].index.tolist())
        rows.append({
            "gw": gw,
            "n_blank": len(blank_teams),
            "n_double": len(double_teams),
            "blank_teams": ", ".join(team_name.get(t, str(t)) for t in blank_teams),
            "double_teams": ", ".join(team_name.get(t, str(t)) for t in double_teams),
        })
    return pd.DataFrame(rows)


@dataclass
class ChipRecommendation:
    chip: str
    gw: int
    reason: str
    score: float


# Only surface a chip suggestion once its trigger clears a "genuinely
# notable" bar -- otherwise every gameweek would get a recommendation and
# the signal would be worthless. These are judgement thresholds, not
# tuned constants.
_MIN_BLANK_TEAMS_FOR_FREE_HIT = 3
_MIN_DOUBLE_TEAMS_FOR_BENCH_BOOST = 2
_MIN_WILDCARD_GAP = 8.0


def recommend_chip_windows(
    multi_pred_df: pd.DataFrame,
    gws: list[int],
    fixtures_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    current_squad_df: pd.DataFrame | None = None,
) -> list[ChipRecommendation]:
    """Heuristic best-window suggestion for each of the four chips within
    the forecast horizon. Any chip with no window clearing its threshold is
    simply omitted -- "nothing stands out yet" is a legitimate answer,
    especially early in a short horizon.
    """
    if multi_pred_df.empty or not gws:
        return []
    recs: list[ChipRecommendation] = []
    bd = blank_double_gw_summary(fixtures_df, teams_df, gws)

    if not bd.empty and bd["n_blank"].max() >= _MIN_BLANK_TEAMS_FOR_FREE_HIT:
        row = bd.loc[bd["n_blank"].idxmax()]
        recs.append(ChipRecommendation(
            chip="Free Hit", gw=int(row["gw"]),
            reason=f"{int(row['n_blank'])} teams have no fixture ({row['blank_teams']}) -- the biggest blank gameweek in this window.",
            score=float(row["n_blank"]),
        ))

    if not bd.empty and bd["n_double"].max() >= _MIN_DOUBLE_TEAMS_FOR_BENCH_BOOST:
        row = bd.loc[bd["n_double"].idxmax()]
        recs.append(ChipRecommendation(
            chip="Bench Boost", gw=int(row["gw"]),
            reason=f"{int(row['n_double'])} teams play twice ({row['double_teams']}) -- the best chance a full bench also has fixtures.",
            score=float(row["n_double"]),
        ))

    ceiling_best = None
    for gw in gws:
        pcol, ccol = f"GW{gw}_Points", f"GW{gw}_Ceiling"
        if pcol not in multi_pred_df.columns:
            continue
        score_col = ccol if ccol in multi_pred_df.columns else pcol
        sub = multi_pred_df.dropna(subset=[score_col])
        if sub.empty:
            continue
        top = sub.loc[sub[score_col].idxmax()]
        if ceiling_best is None or top[score_col] > ceiling_best[1]:
            ceiling_best = (gw, float(top[score_col]), str(top["web_name"]), float(top.get(pcol, top[score_col])))
    if ceiling_best is not None:
        gw, score, name, pts = ceiling_best
        recs.append(ChipRecommendation(
            chip="Triple Captain", gw=gw,
            reason=f"{name}'s predicted ceiling ({score:.1f} pts) is the highest of any player across this window (expected {pts:.1f}).",
            score=score,
        ))

    if current_squad_df is not None and not current_squad_df.empty:
        current_ids = set(current_squad_df["player_id"])
        best_gap, best_gw = None, None
        for gw in gws:
            pcol = f"GW{gw}_Points"
            if pcol not in multi_pred_df.columns:
                continue
            optimal = find_optimal_squad(multi_pred_df, points_col=pcol)
            if optimal.empty:
                continue
            optimal_pts = float(optimal[pcol].sum())
            current_pts = float(multi_pred_df[multi_pred_df["player_id"].isin(current_ids)][pcol].sum())
            gap = optimal_pts - current_pts
            if best_gap is None or gap > best_gap:
                best_gap, best_gw = gap, gw
        if best_gw is not None and best_gap >= _MIN_WILDCARD_GAP:
            recs.append(ChipRecommendation(
                chip="Wildcard", gw=best_gw,
                reason=f"An unconstrained optimal 15 would outscore your current squad by {best_gap:.1f} pts that gameweek -- the biggest gap in this window.",
                score=best_gap,
            ))

    return sorted(recs, key=lambda r: r.gw)


def project_full_season(result, squad_df: pd.DataFrame) -> dict:
    """Projects a squad's points across the *entire remaining season* (the
    next unplayed gameweek through GW38), applying captain doubling each
    week the same way a real gameweek actually scores -- so the total is
    directly comparable to a real manager's reported season total, not
    just a raw sum of XI point-estimates.

    This reuses the same per-gameweek forecast as the short look-ahead,
    just extended across the whole remaining calendar -- fixture data for
    the full season is already published up front, even though far-future
    form/rotation estimates are necessarily less certain than next week's;
    treat this as an order-of-magnitude sanity check on a squad, not a
    precise forecast that far out. Returns ``{}`` if there's no target
    gameweek to project from, or the squad is empty.
    """
    start_gw = result.future_gws[0] if result.future_gws else None
    if start_gw is None or squad_df is None or squad_df.empty:
        return {}
    remaining = MAX_FPL_GW - start_gw + 1
    if remaining <= 0:
        return {}
    multi_pred, gws = build_multi_gw_forecast(result, n_gws=remaining)
    if multi_pred.empty:
        return {}

    squad_pred = multi_pred[multi_pred["player_id"].isin(squad_df["player_id"])]
    per_gw_rows, total = [], 0.0
    for gw in gws:
        gw_col = f"GW{gw}_Points"
        if gw_col not in squad_pred.columns:
            continue
        lineup, bench, captain, vice = build_starting_xi(squad_pred, gw_col)
        captain_bonus = float(captain[gw_col]) if isinstance(captain, pd.Series) and gw_col in captain.index else 0.0
        gw_points = float(lineup[gw_col].sum()) + captain_bonus
        total += gw_points
        per_gw_rows.append({"gw": gw, "points": gw_points})

    return {
        "total_points": total,
        "n_gws": len(per_gw_rows),
        "start_gw": start_gw,
        "per_gw": pd.DataFrame(per_gw_rows),
        "points_per_gw": total / len(per_gw_rows) if per_gw_rows else 0.0,
    }


def _avg_team_fdr(fixtures_df: pd.DataFrame, teams_df: pd.DataFrame, team_name: str, gws: list[int]) -> float | None:
    """Average FDR (easier of the two on a double gameweek) a team faces
    across ``gws`` -- used to explain *why* a rotation dip is suggested,
    not just that one was detected.
    """
    if not gws or teams_df is None or teams_df.empty:
        return None
    long_df = fixtures_long_by_team(fixtures_df)
    if long_df.empty:
        return None
    name_to_id = dict(zip(teams_df["name"], teams_df["id"]))
    team_id = name_to_id.get(team_name)
    if team_id is None:
        return None
    sub = long_df[(long_df["team"] == team_id) & (long_df["event"].isin(gws))]
    if sub.empty:
        return None
    return float(sub.groupby("event")["difficulty"].min().mean())


def recommend_rotation_plan(
    multi_pred_df: pd.DataFrame,
    gws: list[int],
    squad_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> pd.DataFrame:
    """Bench-now, start-later advice for players already in the squad --
    distinct from a transfer: nobody leaves the squad, they just sit out
    the gameweeks where the model's own week-by-week optimal XI wouldn't
    start them (typically a tough run of fixtures), flagged to return once
    it would start them again.

    Only reports a *bounded* dip -- started before AND after within the
    window -- a player who simply never makes the XI in this window isn't
    a rotation case, they're a "doesn't deserve a squad spot" case (which
    the Transfers tab, not this, is meant to catch).
    """
    if multi_pred_df.empty or not gws or squad_df is None or squad_df.empty:
        return pd.DataFrame()

    squad_pred = multi_pred_df[multi_pred_df["player_id"].isin(squad_df["player_id"])]
    starters_by_gw: dict[int, set] = {}
    for gw in gws:
        gw_col = f"GW{gw}_Points"
        if gw_col not in squad_pred.columns:
            continue
        lineup, *_ = build_starting_xi(squad_pred, gw_col)
        starters_by_gw[gw] = set(lineup["player_id"]) if not lineup.empty else set()

    ordered_gws = [g for g in gws if g in starters_by_gw]
    team_by_player = dict(zip(squad_df["player_id"], squad_df.get("team_name", pd.Series(dtype=str))))
    name_by_player = dict(zip(squad_df["player_id"], squad_df["web_name"]))

    rows = []
    for pid in squad_df["player_id"]:
        flags = [pid in starters_by_gw[gw] for gw in ordered_gws]
        i = 0
        while i < len(flags):
            if flags[i]:
                i += 1
                continue
            j = i
            while j < len(flags) and not flags[j]:
                j += 1
            if i > 0 and j < len(flags):  # a genuine dip: started before AND after
                bench_gws = ordered_gws[i:j]
                avg_fdr = _avg_team_fdr(fixtures_df, teams_df, team_by_player.get(pid), bench_gws)
                span = f"GW{bench_gws[0]}" if len(bench_gws) == 1 else f"GW{bench_gws[0]}–GW{bench_gws[-1]}"
                reason = f"Drops out of your model-optimal XI for {span}" + (f" (avg fixture difficulty {avg_fdr:.1f})" if avg_fdr is not None else "") + f", back in for GW{ordered_gws[j]}."
                rows.append({
                    "player_id": pid, "player": name_by_player.get(pid, str(pid)),
                    "bench_from_gw": bench_gws[0], "bench_to_gw": bench_gws[-1],
                    "return_gw": ordered_gws[j], "reason": reason,
                })
            i = j
    return pd.DataFrame(rows)
