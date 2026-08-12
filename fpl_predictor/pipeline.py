"""End-to-end orchestration: live FPL data -> features -> trained model ->
future-gameweek predictions. This is the single entry point the web app
(and any other frontend) should call; it also generates a self-contained
synthetic dataset for offline development/tests when the live API can't
be reached (e.g. from a sandboxed environment without internet access).
"""

from __future__ import annotations

import concurrent.futures
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fpl_predictor.config import MAX_FPL_GW, SEED
from fpl_predictor.data_sources import fpl_api
from fpl_predictor.data_sources.team_strength import EloRatings, compute_elo_ratings
from fpl_predictor.features import create_feature_frame, feature_columns_present
from fpl_predictor.forecast import apply_playing_probability, predict_future_gameweeks
from fpl_predictor.model import TrainedModel, train_model


def assign_playing_probability(status: str, news: str) -> float:
    if status == "a":
        return 1.0
    if status == "d":
        return 0.5
    if status in ("i", "s"):
        return 0.0
    match = re.search(r"(\d+)% chance of playing", news or "")
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


@dataclass
class DataBundle:
    players_df: pd.DataFrame
    teams_df: pd.DataFrame
    fixtures_df: pd.DataFrame
    history_df: pd.DataFrame
    meta: dict = field(default_factory=dict)


def fetch_live_data(max_workers: int = 12, ttl_seconds: int = 3600, progress_cb=None) -> DataBundle:
    """Pull everything needed from the live FPL API (bootstrap, fixtures,
    per-player gameweek history), with disk caching and graceful fallback
    baked in at the ``fpl_api`` layer.
    """
    boot, boot_meta = fpl_api.fetch_bootstrap(ttl_seconds=ttl_seconds)
    if boot is None:
        return DataBundle(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), meta={"ok": False, "error": boot_meta.get("error", "bootstrap unavailable")})

    players_df, teams_df = fpl_api.players_and_teams_from_bootstrap(boot)
    players_df["playing_prob"] = players_df.apply(lambda r: assign_playing_probability(r.get("status", "a"), r.get("news", "")), axis=1)

    fixtures_json, fx_meta = fpl_api.fetch_fixtures(ttl_seconds=ttl_seconds)
    fixtures_df = fpl_api.fixtures_df_from_json(fixtures_json or [])

    rows = []
    failed = 0

    def _fetch_one(pid):
        data, _ = fpl_api.fetch_player_history(pid, ttl_seconds=24 * 3600)
        out = []
        if data:
            for gw in data.get("history", []):
                r = dict(gw)
                r["player_id"] = pid
                out.append(r)
        return out

    player_ids = players_df["id"].tolist()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_one, pid): pid for pid in player_ids}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            try:
                rows.extend(fut.result())
            except Exception:
                failed += 1
            if progress_cb:
                progress_cb(done, len(player_ids))

    history_df = pd.DataFrame(rows)
    if not history_df.empty and "round" in history_df.columns:
        history_df["round"] = history_df["round"].astype(int)

    meta = {
        "ok": True,
        "bootstrap_source": boot_meta.get("source"),
        "fixtures_source": fx_meta.get("source"),
        "n_players": len(players_df),
        "n_history_rows": len(history_df),
        "n_players_failed": failed,
    }
    return DataBundle(players_df, teams_df, fixtures_df, history_df, meta=meta)


@dataclass
class PipelineResult:
    data: DataBundle
    elo_ratings: EloRatings
    feat_df: pd.DataFrame
    feature_cols: list
    trained_model: TrainedModel
    pred_df: pd.DataFrame
    gw_cols: list
    last_completed_gw: int
    future_gws: list


def run_pipeline(data: DataBundle, n_future_gws: int | None = None) -> PipelineResult:
    """Build features, train the per-position models, and project points
    onto upcoming gameweeks -- everything the UI needs, in one call.
    """
    elo_ratings = compute_elo_ratings()
    feat_df = create_feature_frame(data.history_df, data.players_df, data.fixtures_df, elo_ratings)
    if feat_df.empty:
        return PipelineResult(data, elo_ratings, feat_df, [], TrainedModel([]), pd.DataFrame(), [], 0, [])

    feature_cols = feature_columns_present(feat_df)
    trained_model = train_model(feat_df, feature_cols)

    last_completed_gw = int(feat_df["round"].max())
    max_fixture_gw = int(data.fixtures_df["event"].max()) if not data.fixtures_df.empty else MAX_FPL_GW
    if n_future_gws is None:
        n_future_gws = max(0, max_fixture_gw - last_completed_gw)
    future_gws = list(range(last_completed_gw + 1, last_completed_gw + 1 + n_future_gws))

    pred_df = predict_future_gameweeks(trained_model, feat_df, data.players_df, data.fixtures_df, elo_ratings, future_gws)
    gw_cols = [f"GW{g}_Points" for g in future_gws]
    if not pred_df.empty:
        pred_df = apply_playing_probability(pred_df, data.players_df, gw_cols)

    return PipelineResult(data, elo_ratings, feat_df, feature_cols, trained_model, pred_df, gw_cols, last_completed_gw, future_gws)


# ---------------------------------------------------------------------------
# Synthetic demo data: lets the app and test-suite run fully offline with a
# small, deterministic, self-consistent league (used automatically when the
# live FPL API can't be reached).
# ---------------------------------------------------------------------------

_DEMO_TEAMS = ["Arsenal", "Aston Villa", "Chelsea", "Everton", "Liverpool", "Man City", "Man Utd", "Newcastle"]
_DEMO_POS_TEMPLATE = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4]  # per team: 2 GK, 4 DEF, 4 MID, 2 FWD


def build_demo_data(n_gws: int = 10, n_future_gws: int = 5, seed: int = SEED) -> DataBundle:
    rng = np.random.default_rng(seed)

    players, pid = [], 1
    for team_id, team in enumerate(_DEMO_TEAMS, start=1):
        for slot, pos in enumerate(_DEMO_POS_TEMPLATE):
            base_quality = rng.uniform(0.3, 1.0)
            cost = round(float(3.8 + base_quality * 4.5 + rng.uniform(-0.2, 0.2)), 1)
            players.append({
                "id": pid, "first_name": "Player", "second_name": f"{team[:3]}{pid}",
                "web_name": f"{team[:3]}{pid}", "now_cost": int(cost * 10), "element_type": pos,
                "team": team_id, "team_name": team, "total_points": 0, "minutes": 0,
                "points_per_game": "0", "news": "", "status": "a",
                "chance_of_playing_next_round": 100, "selected_by_percent": "5.0", "form": "0",
                "_quality": base_quality,
            })
            pid += 1
    players_df = pd.DataFrame(players)
    players_df["playing_prob"] = 1.0
    teams_df = pd.DataFrame({"id": range(1, len(_DEMO_TEAMS) + 1), "name": _DEMO_TEAMS})

    # Round-robin-ish fixture list: each gameweek, pair teams up.
    fixtures_rows, event = [], 1
    all_gws = n_gws + n_future_gws
    team_ids = list(range(1, len(_DEMO_TEAMS) + 1))
    fid = 1
    for gw in range(1, all_gws + 1):
        shuffled = team_ids[:]
        rng.shuffle(shuffled)
        for i in range(0, len(shuffled) - 1, 2):
            fixtures_rows.append({"id": fid, "event": gw, "team_h": shuffled[i], "team_a": shuffled[i + 1]})
            fid += 1
    fixtures_df = pd.DataFrame(fixtures_rows)

    history_rows = []
    for p in players:
        q = p["_quality"]
        for gw in range(1, n_gws + 1):
            minutes = int(rng.choice([0, 0, 60, 90, 90], p=[0.1, 0.1, 0.2, 0.3, 0.3]))
            pts_mean = 2 + q * 6 * (minutes / 90.0)
            pts = max(0, int(round(rng.normal(pts_mean, 2.0))))
            history_rows.append({
                "player_id": p["id"], "round": gw, "total_points": pts, "minutes": minutes,
                "goals_scored": int(rng.poisson(0.15 * q)) if p["element_type"] in (3, 4) else 0,
                "assists": int(rng.poisson(0.1 * q)),
                "clean_sheets": int(rng.random() < 0.3 * q) if p["element_type"] in (1, 2) else 0,
                "goals_conceded": int(rng.poisson(1.2)), "bps": int(rng.normal(20, 10)),
                "ict_index": round(float(rng.normal(5 * q, 2)), 1),
                "influence": round(float(rng.normal(20 * q, 5)), 1),
                "creativity": round(float(rng.normal(15 * q, 5)), 1),
                "threat": round(float(rng.normal(15 * q, 5)), 1),
                "was_home": bool(gw % 2 == 0), "opponent_team": ((p["team"] + gw) % len(_DEMO_TEAMS)) + 1,
                "kickoff_time": f"2025-{(gw % 12) + 1:02d}-01T15:00:00Z",
            })
    history_df = pd.DataFrame(history_rows)

    meta = {"ok": True, "bootstrap_source": "demo", "fixtures_source": "demo", "n_players": len(players_df), "n_history_rows": len(history_df), "n_players_failed": 0, "demo": True}
    return DataBundle(players_df.drop(columns=["_quality"]), teams_df, fixtures_df, history_df, meta=meta)
