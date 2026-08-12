"""Walk-forward backtest: measures whether this tool would actually have
been useful in the past, not just what its cross-validation error looks
like in isolation.

For each historical gameweek from ``min_train_gws`` onward, the model is
retrained using *only* gameweeks strictly before it (identical code path to
a real run, just replayed against history) and used to predict that
gameweek -- there is no lookahead. Predictions are compared to what
actually happened, and also used to pick a fresh best-XI-under-budget squad
(no real-world transfer constraints -- this isolates *prediction and
selection* quality, not season-long transfer strategy) whose *actual*
points that gameweek are compared against the simplest alternative a
manager might otherwise use: chasing last gameweek's top scorers.

Only usable once there's real multi-gameweek history to replay against
(the live FPL API mid/late-season, or the synthetic demo dataset) --
that's the whole point: pre-season there's nothing to backtest yet either.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fpl_predictor.config import MAX_SQUAD_COST
from fpl_predictor.features import create_feature_frame, feature_columns_present
from fpl_predictor.forecast import predict_future_gameweeks
from fpl_predictor.model import train_model
from fpl_predictor.optimizer import build_starting_xi, find_optimal_squad
from fpl_predictor.pipeline import DataBundle


@dataclass
class BacktestGwResult:
    gw: int
    n_players: int
    mae: float
    spearman_corr: float
    model_xi_points: float | None
    baseline_xi_points: float | None


@dataclass
class BacktestReport:
    per_gw: list[BacktestGwResult] = field(default_factory=list)

    @property
    def overall_mae(self) -> float:
        rows = [r for r in self.per_gw if r.mae == r.mae]
        if not rows:
            return float("nan")
        total_n = sum(r.n_players for r in rows)
        return sum(r.mae * r.n_players for r in rows) / total_n if total_n else float("nan")

    @property
    def mean_spearman(self) -> float:
        vals = [r.spearman_corr for r in self.per_gw if r.spearman_corr == r.spearman_corr]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def total_model_xi_points(self) -> float:
        return sum(r.model_xi_points for r in self.per_gw if r.model_xi_points is not None)

    @property
    def total_baseline_xi_points(self) -> float:
        return sum(r.baseline_xi_points for r in self.per_gw if r.baseline_xi_points is not None)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(r) for r in self.per_gw])


def _realized_xi_points(pred_pool: pd.DataFrame, rank_col: str, actual_map: dict, budget: float) -> float | None:
    """Pick a squad/XI/captain using ``rank_col`` (predicted or naive-baseline
    points -- whatever the manager would have known *before* the gameweek),
    then score it using real outcomes (``actual_map``), captain doubled --
    i.e. what a manager following this ranking would actually have scored.
    """
    squad = find_optimal_squad(pred_pool, budget=budget, points_col=rank_col)
    if squad.empty:
        return None
    lineup, bench, captain, vice = build_starting_xi(squad, rank_col)
    if lineup.empty:
        return None
    realized = sum(actual_map.get(pid, 0.0) for pid in lineup["player_id"])
    realized += actual_map.get(captain["player_id"], 0.0)  # captain's score counted twice
    return float(realized)


def run_backtest(
    data: DataBundle,
    min_train_gws: int = 3,
    start_gw: int | None = None,
    end_gw: int | None = None,
    budget: float = MAX_SQUAD_COST,
    tune: bool = False,
) -> BacktestReport:
    """Replay the pipeline gameweek-by-gameweek over already-completed
    history. ``tune=False`` (default) skips per-gameweek hyperparameter
    search for speed, since backtesting retrains many times; set
    ``tune=True`` for a slower but more representative run.
    """
    history_df = data.history_df
    report = BacktestReport()
    if history_df.empty or "round" not in history_df.columns:
        return report

    rounds = sorted(history_df["round"].unique())
    lo = start_gw if start_gw is not None else rounds[0] + min_train_gws
    hi = end_gw if end_gw is not None else rounds[-1]
    target_gws = [g for g in rounds if lo <= g <= hi]

    prev_actual_map: dict | None = None

    for gw in target_gws:
        train_history = history_df[history_df["round"] < gw]
        if train_history.empty:
            continue
        feat_df = create_feature_frame(train_history, data.players_df, data.fixtures_df)
        if feat_df.empty or feat_df["round"].nunique() < min_train_gws:
            continue

        feature_cols = feature_columns_present(feat_df)
        trained_model = train_model(feat_df, feature_cols, tune=tune)

        pred_df = predict_future_gameweeks(trained_model, feat_df, data.players_df, data.fixtures_df, [gw])
        gw_col = f"GW{gw}_Points"
        if pred_df.empty or gw_col not in pred_df.columns:
            continue

        actual_rows = history_df[history_df["round"] == gw][["player_id", "total_points"]].drop_duplicates("player_id")
        if actual_rows.empty:
            continue
        actual_map = dict(zip(actual_rows["player_id"], actual_rows["total_points"]))

        merged = pred_df[["player_id", gw_col]].merge(actual_rows, on="player_id", how="inner")
        if merged.empty:
            continue
        mae = float((merged[gw_col] - merged["total_points"]).abs().mean())
        corr = merged[gw_col].corr(merged["total_points"], method="spearman")
        corr = float(corr) if corr == corr else float("nan")

        model_xi_points = _realized_xi_points(pred_df, gw_col, actual_map, budget)

        baseline_xi_points = None
        if prev_actual_map is not None:
            baseline_pool = pred_df[["player_id", "web_name", "team_name", "element_type", "now_cost"]].copy()
            baseline_pool[gw_col] = baseline_pool["player_id"].map(prev_actual_map).fillna(0.0)
            baseline_xi_points = _realized_xi_points(baseline_pool, gw_col, actual_map, budget)

        report.per_gw.append(BacktestGwResult(gw, len(merged), mae, corr, model_xi_points, baseline_xi_points))
        prev_actual_map = actual_map

    return report
