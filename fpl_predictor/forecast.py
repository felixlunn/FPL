"""Project trained per-position models forward onto specific future
gameweeks, correctly handling blank gameweeks (team has no fixture -> 0
points) and double gameweeks (team plays twice -> points summed across
both fixtures) rather than naively repeating the player's last known row.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from fpl_predictor.config import NEUTRAL_FIXTURE_DIFFICULTY
from fpl_predictor.data_sources.fpl_api import fixtures_long_by_team
from fpl_predictor.model import TrainedModel


def _fixtures_by_team_and_gw(fixtures_df: pd.DataFrame) -> dict:
    """team_id -> gw -> list of (opponent_team_id, was_home, difficulty)."""
    out = defaultdict(lambda: defaultdict(list))
    long_df = fixtures_long_by_team(fixtures_df)
    if long_df.empty:
        return out
    for row in long_df.itertuples(index=False):
        if row.event != row.event:  # NaN check
            continue
        difficulty = row.difficulty if row.difficulty == row.difficulty else NEUTRAL_FIXTURE_DIFFICULTY
        out[row.team][int(row.event)].append((row.opponent, bool(row.was_home), float(difficulty)))
    return out


def predict_future_gameweeks(
    trained_model: TrainedModel,
    feat_df: pd.DataFrame,
    players_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    gameweeks: list[int],
) -> pd.DataFrame:
    """Return one row per player with a ``GW{n}_Points`` column per requested
    gameweek plus ``pred_points_total`` summed across all of them.
    """
    if feat_df.empty or not gameweeks:
        return pd.DataFrame()

    base_rows = feat_df.sort_values("round").groupby("player_id").tail(1).set_index("player_id", drop=False)
    fixtures_lookup = _fixtures_by_team_and_gw(fixtures_df)
    feature_cols = trained_model.feature_cols

    result = base_rows[["web_name", "team_name", "element_type", "now_cost", "player_id"]].copy()

    for gw in gameweeks:
        col_name = f"GW{gw}_Points"
        points = pd.Series(0.0, index=base_rows.index)

        # Group players by team so each team's fixture list is looked up once.
        for team_id, team_rows in base_rows.groupby("team"):
            fixtures_this_gw = fixtures_lookup.get(team_id, {}).get(gw, [])
            if not fixtures_this_gw:
                continue  # blank gameweek for this team

            sim = pd.concat([team_rows] * len(fixtures_this_gw), keys=range(len(fixtures_this_gw)))
            was_home_vals, difficulty_vals = [], []
            for opp_id, was_home, difficulty in fixtures_this_gw:
                was_home_vals.extend([1.0 if was_home else 0.0] * len(team_rows))
                difficulty_vals.extend([difficulty] * len(team_rows))

            sim = sim.reset_index(level=0, drop=True)
            sim["was_home"] = was_home_vals
            sim["fixture_difficulty"] = difficulty_vals

            for c in feature_cols:
                if c not in sim.columns:
                    sim[c] = 0.0
            preds = trained_model.predict(sim[list(dict.fromkeys(feature_cols + ["element_type"]))])
            fixture_points = pd.Series(preds, index=sim["player_id"].to_numpy()).groupby(level=0).sum()
            points.loc[fixture_points.index] = points.loc[fixture_points.index].add(fixture_points, fill_value=0)

        result[col_name] = points.to_numpy()

    gw_cols = [f"GW{gw}_Points" for gw in gameweeks]
    result["pred_points_total"] = result[gw_cols].sum(axis=1)
    return result.sort_values("pred_points_total", ascending=False).reset_index(drop=True)


_COLD_START_DEFAULT_PER90 = {1: 2.0, 2: 2.4, 3: 2.6, 4: 2.6}  # GKP, DEF, MID, FWD


def predict_cold_start_gameweek(
    players_df: pd.DataFrame,
    past_seasons_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    target_gw: int,
) -> pd.DataFrame:
    """Predict a single gameweek with zero current-season history to learn
    from -- the situation every season between the final ball of one
    campaign and the first completed gameweek of the next. There's no
    (player, gameweek) -> points data yet to train a model on, so instead
    of refusing to show real players, fall back to each player's own
    last-season per-90 output (from the FPL API's ``history_past``),
    scaled by an assumed ~68 minutes when selected and by fixture
    difficulty (FPL's own FDR) -- then the usual playing-probability
    adjustment is layered on top by the caller, exactly as for the
    trained-model path.
    """
    if players_df.empty:
        return pd.DataFrame()
    col = f"GW{target_gw}_Points"

    if past_seasons_df is not None and not past_seasons_df.empty:
        latest_past = past_seasons_df.sort_values("season_name").groupby("player_id").tail(1).set_index("player_id")
    else:
        latest_past = pd.DataFrame(columns=["total_points", "minutes"])

    def _per90(row) -> float:
        pid = row["id"]
        if pid in latest_past.index:
            mins = float(latest_past.loc[pid].get("minutes", 0) or 0)
            pts = float(latest_past.loc[pid].get("total_points", 0) or 0)
            if mins >= 180:  # enough minutes last season for the rate to mean anything
                return pts / mins * 90.0
        return _COLD_START_DEFAULT_PER90.get(row["element_type"], 2.0)

    base = players_df.copy()
    base["_per90"] = base.apply(_per90, axis=1)
    base["_baseline_pts"] = base["_per90"] * (68.0 / 90.0)  # assumed minutes when selected

    fixtures_lookup = _fixtures_by_team_and_gw(fixtures_df)

    points = pd.Series(0.0, index=base.index)
    for team_id, team_rows in base.groupby("team"):
        fixtures_this_gw = fixtures_lookup.get(team_id, {}).get(target_gw, [])
        if not fixtures_this_gw:
            continue
        fixture_mult_total = 0.0
        for opp_id, was_home, difficulty in fixtures_this_gw:
            # FDR 1 (easiest) -> a boost; FDR 5 (hardest) -> a penalty; 3 is neutral.
            mult = 1.0 + (NEUTRAL_FIXTURE_DIFFICULTY - difficulty) * 0.12
            mult = max(0.6, min(1.4, mult))
            mult *= 1.05 if was_home else 0.97
            fixture_mult_total += mult
        points.loc[team_rows.index] = team_rows["_baseline_pts"] * fixture_mult_total

    out = base[["id", "web_name", "team_name", "element_type", "now_cost"]].rename(columns={"id": "player_id"}).copy()
    out["now_cost"] = out["now_cost"] / 10.0
    out[col] = points.to_numpy()
    out["pred_points_total"] = out[col]
    return out.sort_values("pred_points_total", ascending=False).reset_index(drop=True)


def apply_playing_probability(pred_df: pd.DataFrame, players_df: pd.DataFrame, gw_cols: list[str]) -> pd.DataFrame:
    """Scale each per-gameweek prediction (and the total) by a player's
    estimated probability of actually playing, so an injured star doesn't
    outrank a fit squad player just on raw ability.
    """
    df = pred_df.merge(players_df[["id", "playing_prob"]].rename(columns={"id": "player_id"}), on="player_id", how="left")
    df["playing_prob"] = df["playing_prob"].fillna(1.0)
    for c in gw_cols:
        if c in df.columns:
            df[c] = df[c] * df["playing_prob"]
    df["pred_points_total_adj"] = df[gw_cols].sum(axis=1) if gw_cols else df.get("pred_points_total", 0.0) * df["playing_prob"]
    return df
