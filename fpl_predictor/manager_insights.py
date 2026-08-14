"""Learnings from the best-performing real managers, via the FPL API's
public "Overall" classic league (id 314) -- fully public data, no login
required.

Important constraint this module is built around: manager-level pick and
chip history only exists for the *currently in-progress* season. Once a
season ends the league standings reset for the new one and the per-
gameweek picks/chips endpoints stop serving the old season's data (unlike
player-level stats, which the vaastav archive preserves indefinitely --
see ``historical_data.py``). So "what did last year's winners do" isn't
retrievable once a new season has started; this instead tracks the
*current* season's top managers as it unfolds, which is a better fit for
a "constantly updating" view anyway. Before the season has any finished
gameweeks this returns an explicit "not enough data yet" result rather
than pretending there's something to show.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field

import pandas as pd

from fpl_predictor.config import FPL_BASE
from fpl_predictor.data_sources.fpl_api import fetch_json

OVERALL_LEAGUE_ID = 314
DEFAULT_TOP_N = 50


def fetch_top_managers(league_id: int = OVERALL_LEAGUE_ID, top_n: int = DEFAULT_TOP_N, ttl_seconds: int = 3600) -> pd.DataFrame:
    """Top ``top_n`` entries of a public classic league's current
    standings (50 per page, paginated). Empty if the league has no ranked
    entries yet (e.g. before gameweek 1 of a new season).
    """
    rows, page = [], 1
    while len(rows) < top_n:
        data, _ = fetch_json(
            f"{FPL_BASE}leagues-classic/{league_id}/standings/?page_standings={page}",
            cache_name=f"league_{league_id}_standings_p{page}", ttl_seconds=ttl_seconds,
        )
        results = (data or {}).get("standings", {}).get("results", [])
        if not results:
            break
        rows.extend(results)
        if not (data or {}).get("standings", {}).get("has_next"):
            break
        page += 1
    return pd.DataFrame(rows[:top_n])


def fetch_manager_history(entry_id: int, ttl_seconds: int = 3600) -> dict:
    data, _ = fetch_json(f"{FPL_BASE}entry/{entry_id}/history/", cache_name=f"entry_{entry_id}_history", ttl_seconds=ttl_seconds)
    return data or {}


def fetch_manager_picks(entry_id: int, gw: int, ttl_seconds: int = 3600) -> dict:
    data, _ = fetch_json(f"{FPL_BASE}entry/{entry_id}/event/{gw}/picks/", cache_name=f"entry_{entry_id}_picks_gw{gw}", ttl_seconds=ttl_seconds)
    return data or {}


def fetch_entry_info(entry_id: int, ttl_seconds: int = 3600) -> dict:
    data, _ = fetch_json(f"{FPL_BASE}entry/{entry_id}/", cache_name=f"entry_{entry_id}_info", ttl_seconds=ttl_seconds)
    return data or {}


def fetch_entry_squad(entry_id: int, gw: int | None) -> dict:
    """A real manager's actual 15-man squad (element ids + captain/vice)
    for a specific gameweek, via the same public per-gameweek picks
    endpoint used for top-manager sampling -- the "give the app your
    team" entry point, so a manager doesn't have to rebuild their squad
    by hand from a list of names.

    Important constraint: FPL keeps a team's picks private until that
    gameweek's deadline has passed (so rivals can't copy a last-minute
    move), so this only works for a gameweek that's already locked --
    never the *upcoming* one. Pass ``gw=None`` (e.g. before any gameweek
    this season has a passed deadline) to get the explanatory "not
    available yet" result directly, without a wasted request.
    """
    if gw is None:
        return {"ok": False, "message": "Team import isn't available yet — no gameweek this season has passed its deadline, so picks aren't public yet. Build your squad manually below instead."}

    entry_info = fetch_entry_info(entry_id)
    if not entry_info:
        return {"ok": False, "message": f"No FPL team found with ID {entry_id}. Double-check the number from your team's URL on the FPL site (fantasy.premierleague.com/entry/<ID>/...)."}

    picks_data = fetch_manager_picks(entry_id, gw)
    picks = picks_data.get("picks", [])
    if not picks:
        return {"ok": False, "message": f"GW{gw} picks aren't public for this team yet. This becomes available once that gameweek's deadline has passed -- build your squad manually below in the meantime."}

    return {
        "ok": True,
        "team_name": entry_info.get("name", "Your team"),
        "manager_name": f"{entry_info.get('player_first_name', '')} {entry_info.get('player_last_name', '')}".strip(),
        "gw": gw,
        "element_ids": [p["element"] for p in picks],
        "captain_element": next((p["element"] for p in picks if p.get("is_captain")), None),
        "vice_captain_element": next((p["element"] for p in picks if p.get("is_vice_captain")), None),
    }


@dataclass
class ManagerInsights:
    n_managers_sampled: int = 0
    chip_usage: pd.DataFrame = field(default_factory=pd.DataFrame)  # columns: chip, gw, n_managers
    ok: bool = False
    message: str = ""


def summarize_top_manager_chips(league_id: int = OVERALL_LEAGUE_ID, top_n: int = DEFAULT_TOP_N, max_workers: int = 10) -> ManagerInsights:
    """When have the best-ranked real managers played their chips so far
    this season? Fetches each sampled manager's history individually (one
    request per manager), so intentionally capped at ``top_n`` rather than
    the full top-10k leaderboard -- enough to see a real pattern without
    hammering the public API.
    """
    top = fetch_top_managers(league_id, top_n)
    if top.empty or "entry" not in top.columns:
        return ManagerInsights(ok=False, message="No ranked entries yet for this league (season hasn't started, or hasn't had a finished gameweek).")

    entry_ids = top["entry"].tolist()
    chip_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_manager_history, eid): eid for eid in entry_ids}
        for fut in concurrent.futures.as_completed(futures):
            entry_id = futures[fut]
            try:
                hist = fut.result()
            except Exception:
                continue
            for chip in hist.get("chips", []):
                chip_rows.append({"entry": entry_id, "chip": chip.get("name"), "gw": chip.get("event")})

    if not chip_rows:
        return ManagerInsights(n_managers_sampled=len(entry_ids), ok=False, message="Sampled top managers haven't played any chips yet this season.")

    chip_df = pd.DataFrame(chip_rows)
    usage = chip_df.groupby(["chip", "gw"]).size().reset_index(name="n_managers").sort_values(["chip", "gw"]).reset_index(drop=True)
    return ManagerInsights(n_managers_sampled=len(entry_ids), chip_usage=usage, ok=True)


def summarize_top_manager_captains(gw: int, league_id: int = OVERALL_LEAGUE_ID, top_n: int = DEFAULT_TOP_N, max_workers: int = 10) -> pd.DataFrame:
    """Captain-choice frequency among the top ``top_n`` managers for a
    specific, already-played gameweek -- empty if that gameweek hasn't
    been played (or scored) yet. Returns raw ``captain_element`` (FPL
    player id) counts; join against ``players_df`` for names.
    """
    top = fetch_top_managers(league_id, top_n)
    if top.empty or "entry" not in top.columns:
        return pd.DataFrame()
    entry_ids = top["entry"].tolist()
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_manager_picks, eid, gw): eid for eid in entry_ids}
        for fut in concurrent.futures.as_completed(futures):
            try:
                picks_data = fut.result()
            except Exception:
                continue
            for p in picks_data.get("picks", []):
                if p.get("is_captain"):
                    rows.append({"entry": futures[fut], "captain_element": p.get("element")})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.groupby("captain_element").size().reset_index(name="n_managers").sort_values("n_managers", ascending=False).reset_index(drop=True)
