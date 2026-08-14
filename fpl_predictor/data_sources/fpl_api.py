"""Fetching and disk-caching of live data from the official FPL API.

Every function degrades gracefully: on network failure it falls back to
whatever is on disk (however stale), and only returns ``None``/empty when
nothing is available at all. This lets the rest of the pipeline (and the
web app) distinguish "no internet, using cache" from "no data exists yet".
"""

from __future__ import annotations

import json
import os
import re
import time

import pandas as pd
import requests

from fpl_predictor.config import CACHE_DIR, FPL_BASE

DEFAULT_TIMEOUT = 15


class FplApiError(RuntimeError):
    """Raised when data cannot be fetched and no usable cache exists."""


def _safe_cache_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)[:200]


def _cache_path(cache_name: str) -> str:
    return os.path.join(CACHE_DIR, cache_name + ".json")


def fetch_json(url: str, cache_name: str | None = None, ttl_seconds: int = 3600, timeout: int = DEFAULT_TIMEOUT):
    """GET a URL as JSON, with a disk cache and a stale-cache fallback.

    Returns ``(data, meta)`` where ``meta`` describes provenance so callers
    (and the UI) can surface e.g. "showing cached data from 3h ago because
    the live API was unreachable" rather than failing silently.
    """
    cache_name = cache_name or ("fpl_" + _safe_cache_name(url))
    cache_file = _cache_path(cache_name)

    if os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < ttl_seconds:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f), {"source": "cache", "age_seconds": age}
            except (json.JSONDecodeError, OSError):
                pass

    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return data, {"source": "live", "age_seconds": 0}
    except (requests.RequestException, json.JSONDecodeError) as exc:
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    age = time.time() - os.path.getmtime(cache_file)
                    return json.load(f), {"source": "stale_cache", "age_seconds": age, "error": str(exc)}
            except (json.JSONDecodeError, OSError):
                pass
        return None, {"source": "unavailable", "error": str(exc)}


def fetch_bootstrap(ttl_seconds: int = 3600):
    return fetch_json(FPL_BASE + "bootstrap-static/", cache_name="bootstrap", ttl_seconds=ttl_seconds)


def fetch_fixtures(ttl_seconds: int = 3600):
    return fetch_json(FPL_BASE + "fixtures/", cache_name="fixtures", ttl_seconds=ttl_seconds)


def fetch_player_history(player_id: int, ttl_seconds: int = 24 * 3600):
    return fetch_json(FPL_BASE + f"element-summary/{player_id}/", cache_name=f"player_{player_id}", ttl_seconds=ttl_seconds)


def players_and_teams_from_bootstrap(bootstrap: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    players = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])
    team_map = dict(zip(teams["id"], teams["name"]))
    players["team_name"] = players["team"].map(team_map)
    cols = [c for c in [
        "id", "first_name", "second_name", "web_name", "now_cost", "element_type",
        "team", "team_name", "total_points", "minutes", "points_per_game", "news", "status",
        "chance_of_playing_next_round", "selected_by_percent", "form",
        # Set-piece duty order: 1 = primary taker, 2 = second choice, etc.
        # (null/missing = not on the list). A season-level snapshot, not
        # per-gameweek -- see the note in features.py on what that implies.
        "penalties_order", "direct_freekicks_order", "corners_and_indirect_freekicks_order",
    ] if c in players.columns]
    return players[cols].copy(), teams


def next_gameweek_from_bootstrap(bootstrap: dict) -> tuple[int | None, int | None]:
    """(current_gw, next_gw) from the API's own event flags -- this is the
    authoritative source for "what gameweek are we predicting for", and
    works even when zero gameweeks have been played yet (pre-season/new
    season), unlike inferring it from completed-gameweek history.
    """
    events = bootstrap.get("events", []) if bootstrap else []
    current = next((e["id"] for e in events if e.get("is_current")), None)
    nxt = next((e["id"] for e in events if e.get("is_next")), None)
    if nxt is None:
        unfinished = [e["id"] for e in events if not e.get("finished")]
        nxt = min(unfinished) if unfinished else None
    return current, nxt


def fixtures_df_from_json(fixtures_json: list) -> pd.DataFrame:
    if not fixtures_json:
        return pd.DataFrame()
    df = pd.DataFrame(fixtures_json)
    if "event" in df.columns:
        df = df.dropna(subset=["event"])
        df["event"] = df["event"].astype(int)
    keep = [c for c in [
        "id", "event", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty", "kickoff_time", "finished",
        "team_h_score", "team_a_score",
    ] if c in df.columns]
    return df[keep].copy()


def fixtures_long_by_team(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the fixtures list from one-row-per-match to one-row-per-team-
    per-fixture: ``team, event, opponent, was_home, difficulty``.

    ``difficulty`` is FPL's own Fixture Difficulty Rating (1 = easiest,
    5 = hardest) from that team's perspective for that specific fixture --
    already maintained by FPL itself, so this needs no separate team-
    strength model or historical data to stay current. Used for both
    historical feature engineering (join on team+round+opponent) and
    future-gameweek forecasting (lookup by team+gameweek).
    """
    cols = ["team", "event", "opponent", "was_home", "difficulty"]
    if fixtures_df is None or fixtures_df.empty:
        return pd.DataFrame(columns=cols)
    df = fixtures_df.copy()
    for c in ("team_h_difficulty", "team_a_difficulty"):
        if c not in df.columns:
            df[c] = float("nan")  # e.g. a fixtures source that doesn't carry FDR
    home = df.rename(columns={"team_h": "team", "team_a": "opponent", "team_h_difficulty": "difficulty"})
    home = home.assign(was_home=True)[cols]
    away = df.rename(columns={"team_a": "team", "team_h": "opponent", "team_a_difficulty": "difficulty"})
    away = away.assign(was_home=False)[cols]
    return pd.concat([home, away], ignore_index=True)


def team_fixture_strings(fixtures_df: pd.DataFrame, teams_df: pd.DataFrame, from_gw: int, n: int = 5) -> dict:
    """team_id -> compact "OPP(H) OPP(A) ..." string of each team's next
    ``n`` fixtures from ``from_gw`` onward, for a per-player "what's coming
    up" column in player tables. A double gameweek shows both fixtures.
    """
    long_df = fixtures_long_by_team(fixtures_df)
    if long_df.empty or teams_df is None or teams_df.empty:
        return {}
    team_short = {row.id: str(row.name)[:3].upper() for row in teams_df.itertuples()}
    upcoming = long_df[long_df["event"] >= from_gw].sort_values("event")
    out = {}
    for team_id, grp in upcoming.groupby("team"):
        parts = [f"{team_short.get(r.opponent, '???')}({'H' if r.was_home else 'A'})" for r in grp.head(n).itertuples()]
        out[team_id] = " ".join(parts)
    return out


def fixture_ticker_table(fixtures_df: pd.DataFrame, teams_df: pd.DataFrame, from_gw: int, n_gws: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A team x gameweek grid for a visual fixture-difficulty ticker.

    Returns ``(labels, difficulty)`` -- same shape, indexed by team name
    with one column per gameweek in range. ``labels`` holds display strings
    ("OPP (H)", or "OPP1(H) + OPP2(A)" for a double gameweek, "-" for a
    blank); ``difficulty`` holds the matching numeric FDR (the *easier* of
    the two on a double gameweek) so a caller can colour the cells --
    kept separate rather than baked into the label so any renderer works,
    not just ones that can parse the FDR back out of display text.
    """
    long_df = fixtures_long_by_team(fixtures_df)
    if long_df.empty or teams_df is None or teams_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    team_name = dict(zip(teams_df["id"], teams_df["name"]))
    team_short = {tid: str(name)[:3].upper() for tid, name in team_name.items()}
    gws = list(range(from_gw, from_gw + n_gws))
    cols = [f"GW{g}" for g in gws]
    names = [team_name[t] for t in sorted(team_name)]
    labels = pd.DataFrame("-", index=names, columns=cols, dtype=object)
    difficulty = pd.DataFrame(float("nan"), index=names, columns=cols, dtype=float)

    sub = long_df[long_df["event"].isin(gws)]
    for row in sub.itertuples():
        tname = team_name.get(row.team)
        if tname is None:
            continue
        col = f"GW{row.event}"
        label = f"{team_short.get(row.opponent, '???')}({'H' if row.was_home else 'A'})"
        cur = labels.at[tname, col]
        labels.at[tname, col] = label if cur == "-" else f"{cur} + {label}"
        cur_d = difficulty.at[tname, col]
        difficulty.at[tname, col] = row.difficulty if pd.isna(cur_d) else min(cur_d, row.difficulty)
    return labels, difficulty
