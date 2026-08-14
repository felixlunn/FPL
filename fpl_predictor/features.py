"""Feature engineering: turns raw per-gameweek player history into a model
ready table of (player, gameweek) rows with predictive features and a
``label_points`` target (the actual points scored that gameweek).

Design notes
------------
* All rolling/lag features are shifted by one gameweek before aggregating,
  so a row's features only ever use information available *before* that
  gameweek kicked off -- this is what makes the eventual time-series CV
  honest (no leakage of the target into its own features).
* Fixture difficulty comes straight from FPL's own Fixture Difficulty
  Rating (``fpl_predictor.data_sources.fpl_api.fixtures_long_by_team``),
  joined on (team, gameweek, opponent) -- no separate team-strength model
  or historical data needed, and it's always as current as the live API.
* The feature set degrades gracefully: optional columns (underlying-stats
  like expected goals only exist in recent FPL seasons) are included only
  when present, so the pipeline still works on older/partial data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpl_predictor.config import NEUTRAL_FIXTURE_DIFFICULTY
from fpl_predictor.data_sources.fpl_api import fixtures_long_by_team

# Base rolling stats computed for every player regardless of position.
_ROLL_SOURCE_COLS = [
    "total_points", "minutes", "bps", "ict_index", "influence", "creativity",
    "threat", "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "saves", "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "bonus",
    # Defensive-contribution scoring (introduced 2025/26): defenders get 2pts
    # for 10+ CBIT (clearances+blocks+interceptions+tackles) in a match,
    # midfielders/forwards for 12+ combined defensive actions. The API
    # exposes the raw counting stats per gameweek; "defensive_actions" below
    # sums whichever of them are present into a single CBIT-style proxy so
    # the model can learn each player's run-rate against that threshold
    # directly, without hardcoding the exact scoring rule (which may be
    # tweaked season to season) -- if a field name below doesn't match what
    # the live API actually returns, it's silently skipped, not an error;
    # check a fetched DataBundle's history_df.columns to confirm/adjust.
    "tackles", "clearances_blocks_interceptions", "recoveries", "defensive_contribution",
    "defensive_actions",
]

# Components summed into the "defensive_actions" CBIT proxy, see above.
_DEFENSIVE_ACTION_COMPONENTS = ["tackles", "clearances_blocks_interceptions", "recoveries"]

# Set-piece duty order fields from bootstrap-static (1 = primary taker, 2 =
# second choice, ... null = not on the list). These are a *current-season
# snapshot*, not per-gameweek history -- the live API doesn't expose who
# took set pieces in gameweek 3 specifically, only who's currently
# nominated. That means if duty changed hands mid-season, historical
# training rows see today's taker rather than whoever it was at the time --
# a known, accepted simplification given no other data source for it.
_SET_PIECE_ORDER_COLS = ["penalties_order", "direct_freekicks_order", "corners_and_indirect_freekicks_order"]

ROLLING_WINDOWS = (3, 6)

# Columns always present in the resulting feature frame (used by model.py).
# Rolling stats derived from _ROLL_SOURCE_COLS (total_points_last1,
# total_points_roll3_mean, expected_goals_roll6_mean, ...) are appended on
# top of these by feature_columns_present().
CORE_FEATURE_COLS = [
    "now_cost", "mins_frac_last3", "started_60_last6",
    "form_ratio", "was_home", "fixture_difficulty",
    "penalties_priority", "set_piece_priority",
]


def _order_to_priority(series: pd.Series) -> pd.Series:
    """1st-choice taker -> 1.0, 2nd choice -> 0.5, ..., not on the list -> 0.
    A simple, monotonically-decreasing encoding of "how likely are they to
    actually take it" from FPL's 1-indexed order field.
    """
    return series.apply(lambda o: 1.0 / o if pd.notna(o) and o and o > 0 else 0.0)


def create_feature_frame(
    history_df: pd.DataFrame,
    players_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the (player, gameweek) training/inference table.

    ``history_df`` is one row per player per past gameweek (as returned by
    the FPL ``element-summary`` endpoint's ``history`` list, concatenated
    across all players). ``players_df`` carries static info (position,
    price, team). ``fixtures_df`` provides each fixture's FPL-assigned
    difficulty rating, joined in per (team, gameweek, opponent).
    """
    df = history_df.copy()
    if df.empty:
        return df

    player_info_cols = [c for c in ["id", "element_type", "now_cost", "team", "team_name", "web_name"] + _SET_PIECE_ORDER_COLS if c in players_df.columns]
    player_info = players_df[player_info_cols].rename(columns={"id": "player_id"})
    df = df.merge(player_info, on="player_id", how="left")

    sort_cols = ["player_id"] + (["round"] if "round" in df.columns else [])
    df = df.sort_values(sort_cols)

    df["penalties_priority"] = _order_to_priority(df["penalties_order"]) if "penalties_order" in df.columns else 0.0
    fk_priority = _order_to_priority(df["direct_freekicks_order"]) if "direct_freekicks_order" in df.columns else 0.0
    corner_priority = _order_to_priority(df["corners_and_indirect_freekicks_order"]) if "corners_and_indirect_freekicks_order" in df.columns else 0.0
    df["set_piece_priority"] = fk_priority + corner_priority

    present_def_cols = [c for c in _DEFENSIVE_ACTION_COMPONENTS if c in df.columns]
    if present_def_cols:
        df["defensive_actions"] = df[present_def_cols].sum(axis=1)

    # --- rolling / lag features -------------------------------------------------
    g = df.groupby("player_id")
    for col in _ROLL_SOURCE_COLS:
        if col not in df.columns:
            continue
        shifted = g[col].shift(1)
        df[f"{col}_last1"] = shifted.fillna(0)
        for w in ROLLING_WINDOWS:
            df[f"{col}_roll{w}_mean"] = shifted.groupby(df["player_id"]).rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"{col}_roll3_std"] = shifted.groupby(df["player_id"]).rolling(window=3, min_periods=1).std().reset_index(level=0, drop=True)

    # convenient aliases matching the original prototype's naming
    df["pts_last1"] = df.get("total_points_last1", 0)
    df["pts_roll3_mean"] = df.get("total_points_roll3_mean", 0)
    df["pts_roll6_mean"] = df.get("total_points_roll6_mean", 0)
    df["pts_roll3_std"] = df.get("total_points_roll3_std", 0)
    df["mins_roll3_mean"] = df.get("minutes_roll3_mean", 0)
    df["mins_roll3_std"] = df.get("minutes_roll3_std", 0)
    df["mins_frac_last3"] = df["mins_roll3_mean"] / 90.0

    minutes_shifted = g["minutes"].shift(1) if "minutes" in df.columns else pd.Series(0, index=df.index)
    df["started_60_last6"] = (
        minutes_shifted.groupby(df["player_id"])
        .rolling(window=6, min_periods=1)
        .apply(lambda s: (s >= 60).sum(), raw=True)
        .reset_index(level=0, drop=True)
    ) / 6.0

    df["now_cost"] = df["now_cost"] / 10.0
    df["form_ratio"] = (df["pts_roll3_mean"] / (df["pts_roll6_mean"] + 1e-6)).replace([np.inf, -np.inf], 0).fillna(0)

    # --- fixture difficulty (FPL's own FDR) ---------------------------------------
    if "was_home" in df.columns:
        df["was_home"] = df["was_home"].astype(float)
    else:
        df["was_home"] = 0.5

    fixtures_long = fixtures_long_by_team(fixtures_df)[["team", "event", "opponent", "difficulty"]]
    if "opponent_team" in df.columns and not fixtures_long.empty:
        df = df.merge(
            fixtures_long, how="left",
            left_on=["team", "round", "opponent_team"], right_on=["team", "event", "opponent"],
        )
        df.drop(columns=[c for c in ["event", "opponent"] if c in df.columns], inplace=True)
    else:
        df["difficulty"] = np.nan
    df["fixture_difficulty"] = df["difficulty"].fillna(NEUTRAL_FIXTURE_DIFFICULTY)
    df.drop(columns=["difficulty"], inplace=True)

    df["goal_contrib"] = df.get("goals_scored", 0) if "goals_scored" in df.columns else 0
    if "assists" in df.columns:
        df["goal_contrib"] = df["goal_contrib"] + df["assists"]

    df["label_points"] = df["total_points"].astype(float) if "total_points" in df.columns else 0.0
    df = df.fillna(0)
    return df


def feature_columns_present(df: pd.DataFrame) -> list[str]:
    """The full feature set actually available in a built feature frame,
    i.e. :data:`CORE_FEATURE_COLS` plus any optional rolling stats that
    could be computed from the source data (xG, bps, ict_index, ...).
    """
    optional = []
    for col in _ROLL_SOURCE_COLS:
        for suffix in ["_last1", "_roll3_mean", "_roll6_mean", "_roll3_std"]:
            name = f"{col}{suffix}"
            if name in df.columns and name not in optional:
                optional.append(name)
    cols = CORE_FEATURE_COLS + [c for c in optional if c not in CORE_FEATURE_COLS]
    return [c for c in cols if c in df.columns]
