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
from fpl_predictor.features import create_feature_frame, feature_columns_present
from fpl_predictor.forecast import apply_playing_probability, predict_cold_start_gameweek, predict_future_gameweeks
from fpl_predictor.minutes_model import MinutesModel, train_minutes_model
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
    past_seasons_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    next_gw: int | None = None
    current_gw: int | None = None
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
    current_gw, next_gw = fpl_api.next_gameweek_from_bootstrap(boot)

    fixtures_json, fx_meta = fpl_api.fetch_fixtures(ttl_seconds=ttl_seconds)
    fixtures_df = fpl_api.fixtures_df_from_json(fixtures_json or [])

    rows, past_rows = [], []
    failed = 0

    def _fetch_one(pid):
        data, _ = fpl_api.fetch_player_history(pid, ttl_seconds=24 * 3600)
        hist_out, past_out = [], []
        if data:
            for gw in data.get("history", []):
                r = dict(gw)
                r["player_id"] = pid
                hist_out.append(r)
            # history_past carries season-level (not per-gameweek) totals for
            # prior seasons -- the only per-player signal the live API gives
            # us before this season has any completed gameweeks of its own.
            for season in data.get("history_past", []):
                r = dict(season)
                r["player_id"] = pid
                past_out.append(r)
        return hist_out, past_out

    player_ids = players_df["id"].tolist()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_one, pid): pid for pid in player_ids}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            try:
                hist_out, past_out = fut.result()
                rows.extend(hist_out)
                past_rows.extend(past_out)
            except Exception:
                failed += 1
            if progress_cb:
                progress_cb(done, len(player_ids))

    history_df = pd.DataFrame(rows)
    if not history_df.empty and "round" in history_df.columns:
        history_df["round"] = history_df["round"].astype(int)
    past_seasons_df = pd.DataFrame(past_rows)

    meta = {
        "ok": True,
        "bootstrap_source": boot_meta.get("source"),
        "fixtures_source": fx_meta.get("source"),
        "n_players": len(players_df),
        "n_history_rows": len(history_df),
        "n_players_failed": failed,
        "next_gw": next_gw,
        "current_gw": current_gw,
        # Diagnostic: which raw per-gameweek stat fields the live API
        # actually returned this run. The FPL API has occasionally renamed
        # or added fields (e.g. the 2025/26 defensive-contribution stats);
        # this makes it easy to confirm feature engineering is picking up
        # what's really there instead of silently skipping a renamed field.
        "history_columns": sorted(history_df.columns.tolist()) if not history_df.empty else [],
    }
    return DataBundle(players_df, teams_df, fixtures_df, history_df, past_seasons_df, next_gw, current_gw, meta=meta)


@dataclass
class PipelineResult:
    data: DataBundle
    feat_df: pd.DataFrame
    feature_cols: list
    trained_model: TrainedModel
    pred_df: pd.DataFrame
    gw_cols: list
    last_completed_gw: int
    future_gws: list
    mode: str = "trained"  # "trained" | "cold_start" | "season_over" | "no_data"
    minutes_model: MinutesModel | None = None


def _resolve_target_gw(data: DataBundle, feat_df: pd.DataFrame) -> int | None:
    """The single gameweek to predict for. Prefer the FPL API's own
    is_next/is_current flags (authoritative, and correct even pre-season
    when there's no completed-gameweek history to infer from); fall back
    to "one past the last completed gameweek" for data sources that don't
    carry that flag (e.g. the synthetic demo dataset).
    """
    if data.next_gw is not None:
        return data.next_gw
    last_completed = int(feat_df["round"].max()) if not feat_df.empty else 0
    max_fixture_gw = int(data.fixtures_df["event"].max()) if not data.fixtures_df.empty else MAX_FPL_GW
    candidate = last_completed + 1
    return candidate if candidate <= max_fixture_gw else None


def run_pipeline(data: DataBundle) -> PipelineResult:
    """Build features, train the per-position models, and predict the
    single upcoming gameweek -- everything the UI needs, in one call.

    Always targets exactly one gameweek: the next one that hasn't been
    played. When that gameweek has no current-season history to learn
    from yet (every season, between the last ball of one campaign and the
    first completed gameweek of the next), predictions fall back to a
    last-season-form + fixture-difficulty baseline instead of failing.
    """
    feat_df = create_feature_frame(data.history_df, data.players_df, data.fixtures_df)

    target_gw = _resolve_target_gw(data, feat_df)
    last_completed_gw = int(feat_df["round"].max()) if not feat_df.empty else 0

    if target_gw is None:
        return PipelineResult(data, feat_df, [], TrainedModel([]), pd.DataFrame(), [], last_completed_gw, [], mode="season_over")

    gw_cols = [f"GW{target_gw}_Points"]

    if feat_df.empty:
        # No (player, gameweek) history to train on yet -- typically the
        # gap between seasons. Real players, heuristic baseline.
        if data.players_df.empty:
            return PipelineResult(data, feat_df, [], TrainedModel([]), pd.DataFrame(), gw_cols, last_completed_gw, [target_gw], mode="no_data")
        pred_df = predict_cold_start_gameweek(data.players_df, data.past_seasons_df, data.fixtures_df, target_gw)
        trained_model, feature_cols, mode, minutes_model = TrainedModel([]), [], "cold_start", None
    else:
        feature_cols = feature_columns_present(feat_df)
        trained_model = train_model(feat_df, feature_cols)
        minutes_model = train_minutes_model(feat_df)
        pred_df = predict_future_gameweeks(trained_model, feat_df, data.players_df, data.fixtures_df, [target_gw], minutes_model=minutes_model)
        mode = "trained"

    if not pred_df.empty:
        pred_df = apply_playing_probability(pred_df, data.players_df, gw_cols)

    return PipelineResult(data, feat_df, feature_cols, trained_model, pred_df, gw_cols, last_completed_gw, [target_gw], mode=mode, minutes_model=minutes_model)


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
                # Persistent (not redrawn per gameweek) "how nailed-on are
                # they" reliability, independent of scoring quality -- so
                # the minutes/rotation model has a genuine, learnable,
                # player-specific pattern to pick up on rather than iid
                # noise, and the ablation study can actually tell whether
                # having that model helps.
                "_start_reliability": rng.beta(3.0, 1.3),
            })
            pid += 1

    # Assign set-piece duty: the two highest-"quality" outfield players per
    # team are 1st/2nd choice on penalties/free-kicks/corners, everyone
    # else (and all GKs) has none -- so the priority features actually
    # carry a signal in tests, same idea as the defensive-action stats.
    for team_id in range(1, len(_DEMO_TEAMS) + 1):
        outfield_sorted = sorted((p for p in players if p["team"] == team_id and p["element_type"] != 1), key=lambda p: -p["_quality"])
        for rank, p in enumerate(outfield_sorted, start=1):
            order = rank if rank <= 2 else None
            p["penalties_order"] = order
            p["direct_freekicks_order"] = order
            p["corners_and_indirect_freekicks_order"] = order
        for p in players:
            if p["team"] == team_id and p["element_type"] == 1:
                p["penalties_order"] = p["direct_freekicks_order"] = p["corners_and_indirect_freekicks_order"] = None

    players_df = pd.DataFrame(players)
    players_df["playing_prob"] = 1.0
    teams_df = pd.DataFrame({"id": range(1, len(_DEMO_TEAMS) + 1), "name": _DEMO_TEAMS})

    # Each team's average player quality doubles as a crude "attack
    # strength" for simulating plausible match scores below (team_style.py
    # needs real-looking scorelines to classify playing styles from).
    team_strength = {tid: float(np.mean([p["_quality"] for p in players if p["team"] == tid])) for tid in range(1, len(_DEMO_TEAMS) + 1)}

    # Round-robin-ish fixture list: each gameweek, pair teams up, with a
    # random FPL-style 1-5 difficulty rating per side of each fixture, and
    # a simulated final score for already-played gameweeks (gw <= n_gws).
    # "schedule" also drives each player's per-gameweek history below, so
    # the two stay internally consistent (a player's recorded opponent
    # really is who their team played that gameweek) -- important now that
    # feature engineering joins fixture difficulty on (team, gw, opponent).
    fixtures_rows = []
    all_gws = n_gws + n_future_gws
    team_ids = list(range(1, len(_DEMO_TEAMS) + 1))
    schedule: dict[int, dict[int, tuple[int, bool, int]]] = {tid: {} for tid in team_ids}
    fid = 1
    for gw in range(1, all_gws + 1):
        shuffled = team_ids[:]
        rng.shuffle(shuffled)
        for i in range(0, len(shuffled) - 1, 2):
            home, away = shuffled[i], shuffled[i + 1]
            home_difficulty, away_difficulty = int(rng.integers(1, 6)), int(rng.integers(1, 6))
            finished = gw <= n_gws
            if finished:
                home_score = int(rng.poisson(0.8 + 1.4 * team_strength[home]))
                away_score = int(rng.poisson(0.5 + 1.1 * team_strength[away]))  # mild home advantage
            else:
                home_score = away_score = None
            fixtures_rows.append({
                "id": fid, "event": gw, "team_h": home, "team_a": away,
                "team_h_difficulty": home_difficulty, "team_a_difficulty": away_difficulty,
                "team_h_score": home_score, "team_a_score": away_score, "finished": finished,
            })
            schedule[home][gw] = (away, True, home_difficulty)
            schedule[away][gw] = (home, False, away_difficulty)
            fid += 1
    fixtures_df = pd.DataFrame(fixtures_rows)

    history_rows = []
    for p in players:
        q = p["_quality"]
        reliability = p["_start_reliability"]
        for gw in range(1, n_gws + 1):
            opponent, was_home, difficulty = schedule[p["team"]][gw]
            # Harder fixtures (higher FDR) suppress points a little, so the
            # synthetic data actually carries a fixture-difficulty signal
            # for the model (and the backtest) to pick up on.
            difficulty_mult = 1.0 + (3 - difficulty) * 0.08

            # Minutes driven by each player's own persistent reliability
            # (see _start_reliability above), not an iid coin flip -- a
            # "nailed-on" player mostly starts and plays 90, a fringe one
            # mostly doesn't feature.
            if rng.random() < reliability:
                minutes = int(rng.choice([90, 75, 60], p=[0.7, 0.2, 0.1]))
            else:
                minutes = int(rng.choice([0, 15, 30], p=[0.6, 0.25, 0.15]))

            tackles = int(rng.poisson(1.5 if p["element_type"] in (2, 3) else 0.3)) if minutes > 0 else 0
            cbi = int(rng.poisson(2.5 if p["element_type"] == 2 else 1.0)) if minutes > 0 else 0
            recoveries = int(rng.poisson(3.0 if p["element_type"] in (2, 3) else 1.0)) if minutes > 0 else 0
            defensive_actions = tackles + cbi + recoveries
            # Mirrors the real 2025/26 defensive-contribution rule (2pts for
            # defenders reaching 10 CBIT, mids/forwards reaching 12) so the
            # feature has a genuine, learnable relationship with points.
            defensive_bonus = 2 if (p["element_type"] == 2 and defensive_actions >= 10) or (p["element_type"] in (3, 4) and defensive_actions >= 12) else 0

            # Primary penalty/free-kick takers occasionally convert one,
            # worth a real points swing -- gives the set-piece priority
            # features an actual signal to learn, not just noise.
            set_piece_bonus = 0
            if minutes > 0:
                if p.get("penalties_order") == 1 and rng.random() < 0.06:
                    set_piece_bonus += 4
                if p.get("direct_freekicks_order") == 1 and rng.random() < 0.03:
                    set_piece_bonus += 4

            pts_mean = (2 + q * 6 * (minutes / 90.0)) * difficulty_mult
            pts = max(0, int(round(rng.normal(pts_mean, 2.0)))) + defensive_bonus + set_piece_bonus
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
                "tackles": tackles, "clearances_blocks_interceptions": cbi, "recoveries": recoveries,
                "was_home": was_home, "opponent_team": opponent,
                "kickoff_time": f"2025-{(gw % 12) + 1:02d}-01T15:00:00Z",
            })
    history_df = pd.DataFrame(history_rows)

    meta = {"ok": True, "bootstrap_source": "demo", "fixtures_source": "demo", "n_players": len(players_df), "n_history_rows": len(history_df), "n_players_failed": 0, "demo": True}
    return DataBundle(
        players_df.drop(columns=["_quality", "_start_reliability"]), teams_df, fixtures_df, history_df,
        past_seasons_df=pd.DataFrame(), next_gw=n_gws + 1, current_gw=n_gws, meta=meta,
    )
