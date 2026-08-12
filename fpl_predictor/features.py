"""Feature engineering: turns raw per-gameweek player history into a model
ready table of (player, gameweek) rows with predictive features and a
``label_points`` target (the actual points scored that gameweek).

Design notes
------------
* All rolling/lag features are shifted by one gameweek before aggregating,
  so a row's features only ever use information available *before* that
  gameweek kicked off -- this is what makes the eventual time-series CV
  honest (no leakage of the target into its own features).
* Fixture difficulty is derived from the Elo ratings in
  :mod:`fpl_predictor.data_sources.team_strength`, joined "as of" each
  fixture's date via ``merge_asof`` so historical rows see the strength
  each opponent actually had at the time, not their current rating.
* The feature set degrades gracefully: optional columns (underlying-stats
  like expected goals only exist in recent FPL seasons) are included only
  when present, so the pipeline still works on older/partial data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpl_predictor.config import canonical_team_name
from fpl_predictor.data_sources.team_strength import EloRatings

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

ROLLING_WINDOWS = (3, 6)

# Columns always present in the resulting feature frame (used by model.py).
# Rolling stats derived from _ROLL_SOURCE_COLS (total_points_last1,
# total_points_roll3_mean, expected_goals_roll6_mean, ...) are appended on
# top of these by feature_columns_present().
CORE_FEATURE_COLS = [
    "now_cost", "mins_frac_last3", "started_60_last6",
    "form_ratio", "was_home", "team_strength", "opponent_strength", "strength_diff",
]


def _elo_timeline(elo_ratings: EloRatings) -> pd.DataFrame:
    """Long (team, date, elo) frame of pre-match ratings for merge_asof lookups."""
    hist = elo_ratings.history
    if hist.empty:
        return pd.DataFrame(columns=["team", "date", "elo"])
    home = hist[["date", "home_team", "pre_home_elo"]].rename(columns={"home_team": "team", "pre_home_elo": "elo"})
    away = hist[["date", "away_team", "pre_away_elo"]].rename(columns={"away_team": "team", "pre_away_elo": "elo"})
    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.dropna(subset=["date"]).sort_values(["team", "date"]).reset_index(drop=True)
    return long_df


def _asof_elo(dates: pd.Series, teams: pd.Series, timeline: pd.DataFrame, fallback: EloRatings) -> np.ndarray:
    """Vectorised "Elo as of date" lookup, falling back to current rating."""
    n = len(dates)
    out = np.full(n, np.nan)
    parsed_dates = pd.Series(pd.to_datetime(dates, errors="coerce", utc=True)).dt.tz_localize(None)
    frame = pd.DataFrame({"date": parsed_dates.to_numpy(), "team": pd.Series(teams).to_numpy(), "_ord": np.arange(n)})
    if timeline.empty:
        vals = frame["team"].map(lambda t: fallback.strength(t))
        return vals.to_numpy(dtype=float)
    for team, grp in frame.groupby("team", sort=False):
        grp_sorted = grp.sort_values("date")
        tl = timeline[timeline["team"] == team]
        if tl.empty or grp_sorted["date"].isna().all():
            merged_vals = np.full(len(grp_sorted), fallback.strength(team))
        else:
            m = pd.merge_asof(grp_sorted, tl[["date", "elo"]], on="date", direction="backward")
            merged_vals = m["elo"].fillna(fallback.strength(team)).to_numpy()
        out[grp_sorted["_ord"].to_numpy()] = merged_vals
    return out


def create_feature_frame(
    history_df: pd.DataFrame,
    players_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    elo_ratings: EloRatings,
) -> pd.DataFrame:
    """Build the (player, gameweek) training/inference table.

    ``history_df`` is one row per player per past gameweek (as returned by
    the FPL ``element-summary`` endpoint's ``history`` list, concatenated
    across all players). ``players_df`` carries static info (position,
    price, team). ``fixtures_df`` provides kickoff dates and home/away
    opponents used for the Elo "as of" lookup.
    """
    df = history_df.copy()
    if df.empty:
        return df

    player_info_cols = [c for c in ["id", "element_type", "now_cost", "team", "team_name", "web_name"] if c in players_df.columns]
    player_info = players_df[player_info_cols].rename(columns={"id": "player_id"})
    df = df.merge(player_info, on="player_id", how="left")

    sort_cols = ["player_id"] + (["round"] if "round" in df.columns else [])
    df = df.sort_values(sort_cols)

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

    # --- fixture / opponent strength ---------------------------------------------
    if "was_home" in df.columns:
        df["was_home"] = df["was_home"].astype(float)
    else:
        df["was_home"] = 0.5

    kickoff = df["kickoff_time"] if "kickoff_time" in df.columns else pd.NaT
    df["_date"] = pd.to_datetime(kickoff, errors="coerce")
    team_names = df["team_name"].map(canonical_team_name) if "team_name" in df.columns else pd.Series("", index=df.index)

    if "opponent_team" in df.columns and "team_name" in players_df.columns and "team" in players_df.columns:
        opp_id_to_name = players_df.drop_duplicates("team")[["team", "team_name"]].set_index("team")["team_name"].to_dict()
        opponent_names = df["opponent_team"].map(opp_id_to_name).map(canonical_team_name)
    else:
        opponent_names = pd.Series("", index=df.index)

    timeline = _elo_timeline(elo_ratings)
    df["team_strength"] = _asof_elo(df["_date"], team_names, timeline, elo_ratings)
    df["opponent_strength"] = _asof_elo(df["_date"], opponent_names, timeline, elo_ratings)
    df["strength_diff"] = df["team_strength"] - df["opponent_strength"]
    df.drop(columns=["_date"], inplace=True)

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
