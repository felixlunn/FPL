"""Loads real historical FPL seasons from the public vaastav/Fantasy-
Premier-League GitHub archive (community-maintained, gameweek-by-gameweek
CSVs going back to 2016-17: https://github.com/vaastav/Fantasy-Premier-League).

This is not an official FPL data source, and it's used here purely to let
the backtest/ablation tooling run against *real* past seasons instead of
only the synthetic demo dataset -- the live FPL API itself only exposes
per-gameweek history for the *current* season, so there's no other way to
validate against real prior seasons without a separate archive like this.

Column names in the archive line up closely with the live API's own
shape (unsurprising -- it's scraped from that same API each gameweek),
so the resulting DataBundle slots into the exact same
create_feature_frame/train_model/run_backtest pipeline unchanged.
"""

from __future__ import annotations

import io
import os

import pandas as pd
import requests

from fpl_predictor.config import CACHE_DIR
from fpl_predictor.pipeline import DataBundle

ARCHIVE_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
DEFAULT_TIMEOUT = 30

# Most recent completed seasons, newest first (the season the app is
# actually predicting for is always excluded -- there's nothing to
# backtest yet for a season in progress).
RECENT_COMPLETED_SEASONS = ["2025-26", "2024-25", "2023-24", "2022-23", "2021-22"]

_HISTORY_COLS = [
    "player_id", "round", "total_points", "minutes", "goals_scored", "assists",
    "clean_sheets", "goals_conceded", "bps", "ict_index", "influence", "creativity",
    "threat", "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "bonus", "was_home", "opponent_team", "kickoff_time",
    "saves", "yellow_cards", "red_cards", "own_goals", "penalties_missed", "penalties_saved",
]
_PLAYER_COLS = [
    "id", "element_type", "now_cost", "web_name", "team", "status", "news",
    "penalties_order", "direct_freekicks_order", "corners_and_indirect_freekicks_order",
]
_FIXTURE_COLS = ["id", "event", "team_h", "team_a", "team_h_score", "team_a_score", "team_h_difficulty", "team_a_difficulty", "kickoff_time", "finished"]


def _cached_csv(url: str, cache_name: str, ttl_seconds: int = 30 * 24 * 3600) -> pd.DataFrame | None:
    """A completed season never changes, so this caches aggressively (30
    days) -- there's no reason to re-download it every run.
    """
    path = os.path.join(CACHE_DIR, cache_name + ".csv")
    if os.path.exists(path) and (__import__("time").time() - os.path.getmtime(path)) < ttl_seconds:
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception:
            pass
    try:
        r = requests.get(url, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        with open(path, "w", encoding="utf-8") as f:
            f.write(r.text)
        return pd.read_csv(io.StringIO(r.text), low_memory=False)
    except Exception:
        if os.path.exists(path):
            try:
                return pd.read_csv(path, low_memory=False)
            except Exception:
                return None
        return None


def fetch_season_data(season: str) -> DataBundle:
    """``season`` like ``"2024-25"``. Returns an empty, ``meta['ok']=False``
    DataBundle if any of the four required files couldn't be fetched.
    """
    base = f"{ARCHIVE_BASE}/{season}"
    teams_raw = _cached_csv(f"{base}/teams.csv", f"hist_{season}_teams")
    fixtures_raw = _cached_csv(f"{base}/fixtures.csv", f"hist_{season}_fixtures")
    players_raw = _cached_csv(f"{base}/players_raw.csv", f"hist_{season}_players")
    gw_raw = _cached_csv(f"{base}/gws/merged_gw.csv", f"hist_{season}_gws")

    if any(df is None for df in (teams_raw, fixtures_raw, players_raw, gw_raw)):
        return DataBundle(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), meta={"ok": False, "error": f"could not fetch {season} data from the historical archive"})

    teams_df = teams_raw[["id", "name"]].copy()
    team_name_by_id = dict(zip(teams_df["id"], teams_df["name"]))

    fixtures_cols = [c for c in _FIXTURE_COLS if c in fixtures_raw.columns]
    fixtures_df = fixtures_raw[fixtures_cols].dropna(subset=["event"]).copy()
    fixtures_df["event"] = fixtures_df["event"].astype(int)

    player_cols = [c for c in _PLAYER_COLS if c in players_raw.columns]
    players_df = players_raw[player_cols].copy()
    players_df["team_name"] = players_df["team"].map(team_name_by_id)
    players_df["playing_prob"] = 1.0  # backtesting doesn't need a live-injury adjustment
    if "id" in players_df.columns:
        players_df["web_name"] = players_df.get("web_name", players_df["id"].astype(str))

    gw = gw_raw.rename(columns={"element": "player_id"}).copy()
    round_source = "GW" if "GW" in gw.columns else "round"
    gw["round"] = pd.to_numeric(gw[round_source], errors="coerce")
    gw = gw.dropna(subset=["round", "player_id"])
    gw["round"] = gw["round"].astype(int)
    hist_cols = [c for c in _HISTORY_COLS if c in gw.columns]
    history_df = gw[hist_cols].copy()

    meta = {
        "ok": True, "bootstrap_source": f"historical:{season}", "fixtures_source": f"historical:{season}",
        "n_players": len(players_df), "n_history_rows": len(history_df), "n_players_failed": 0,
        "season": season, "history_columns": sorted(history_df.columns.tolist()),
    }
    return DataBundle(players_df, teams_df, fixtures_df, history_df, past_seasons_df=pd.DataFrame(), next_gw=None, current_gw=None, meta=meta)
