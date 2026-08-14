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
        keep_cols = ["player_id", "web_name", "team_name", "element_type", "now_cost", "last_season_reliability"]
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
        from fpl_predictor.optimizer import find_optimal_squad
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
