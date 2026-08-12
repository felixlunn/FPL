"""Project trained per-position models forward onto specific future
gameweeks, correctly handling blank gameweeks (team has no fixture -> 0
points) and double gameweeks (team plays twice -> points summed across
both fixtures) rather than naively repeating the player's last known row.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from fpl_predictor.config import canonical_team_name
from fpl_predictor.data_sources.team_strength import EloRatings
from fpl_predictor.model import TrainedModel


def _fixtures_by_team_and_gw(fixtures_df: pd.DataFrame) -> dict:
    """team_id -> gw -> list of (opponent_team_id, was_home)."""
    out = defaultdict(lambda: defaultdict(list))
    if fixtures_df is None or fixtures_df.empty:
        return out
    for row in fixtures_df.itertuples(index=False):
        gw = int(row.event)
        out[row.team_h][gw].append((row.team_a, True))
        out[row.team_a][gw].append((row.team_h, False))
    return out


def predict_future_gameweeks(
    trained_model: TrainedModel,
    feat_df: pd.DataFrame,
    players_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    elo_ratings: EloRatings,
    gameweeks: list[int],
) -> pd.DataFrame:
    """Return one row per player with a ``GW{n}_Points`` column per requested
    gameweek plus ``pred_points_total`` summed across all of them.
    """
    if feat_df.empty or not gameweeks:
        return pd.DataFrame()

    base_rows = feat_df.sort_values("round").groupby("player_id").tail(1).set_index("player_id", drop=False)
    team_id_to_name = players_df.drop_duplicates("team").set_index("team")["team_name"].to_dict()
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
            team_name = canonical_team_name(team_id_to_name.get(team_id, ""))
            team_strength = elo_ratings.strength(team_name)

            sim = pd.concat([team_rows] * len(fixtures_this_gw), keys=range(len(fixtures_this_gw)))
            was_home_vals, opp_strength_vals = [], []
            for opp_id, was_home in fixtures_this_gw:
                opp_name = canonical_team_name(team_id_to_name.get(opp_id, ""))
                was_home_vals.extend([1.0 if was_home else 0.0] * len(team_rows))
                opp_strength_vals.extend([elo_ratings.strength(opp_name)] * len(team_rows))

            sim = sim.reset_index(level=0, drop=True)
            sim["was_home"] = was_home_vals
            sim["opponent_strength"] = opp_strength_vals
            sim["team_strength"] = team_strength
            sim["strength_diff"] = team_strength - sim["opponent_strength"]

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
