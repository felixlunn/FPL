"""Runs the walk-forward backtest (fpl_predictor.backtest) against real
historical FPL seasons (fpl_predictor.historical_data) rather than only
the synthetic demo dataset -- this is the real answer to "has the model
actually been validated against past seasons", not just a methodology
sanity check.

Run via:

    python -m fpl_predictor.historical_backtest [--save path.csv]

Needs outbound network access to raw.githubusercontent.com (the public
vaastav/Fantasy-Premier-League archive) -- unavailable in fully offline
environments, unlike the rest of this project's test suite.
"""

from __future__ import annotations

import sys

import pandas as pd

from fpl_predictor.backtest import run_backtest
from fpl_predictor.historical_data import RECENT_COMPLETED_SEASONS, fetch_season_data


def run_historical_backtest(
    seasons: list[str] | None = None,
    min_train_gws: int = 4,
    tune: bool = False,
    start_gw: int | None = None,
    end_gw: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Returns (summary_df, {season: BacktestReport}) -- the per-season
    BacktestReport objects carry the full per-gameweek detail if needed.
    """
    seasons = seasons or RECENT_COMPLETED_SEASONS
    rows, reports = [], {}
    for season in seasons:
        data = fetch_season_data(season)
        if not data.meta.get("ok"):
            rows.append({"season": season, "ok": False, "n_gws": 0, "overall_mae": float("nan"),
                         "mean_spearman": float("nan"), "total_model_xi_points": 0.0,
                         "total_baseline_xi_points": 0.0, "error": data.meta.get("error")})
            continue
        report = run_backtest(data, min_train_gws=min_train_gws, tune=tune, start_gw=start_gw, end_gw=end_gw)
        reports[season] = report
        rows.append({
            "season": season, "ok": True, "n_gws": len(report.per_gw),
            "overall_mae": report.overall_mae, "mean_spearman": report.mean_spearman,
            "total_model_xi_points": report.total_model_xi_points,
            "total_baseline_xi_points": report.total_baseline_xi_points,
            "error": None,
        })
    return pd.DataFrame(rows), reports


if __name__ == "__main__":
    summary, reports = run_historical_backtest()
    pd.set_option("display.width", 140)
    summary["advantage_vs_baseline"] = summary["total_model_xi_points"] - summary["total_baseline_xi_points"]
    print(summary.to_string(index=False))

    ok = summary[summary["ok"]]
    if not ok.empty:
        weighted_mae = (ok["overall_mae"] * ok["n_gws"]).sum() / ok["n_gws"].sum()
        print(f"\nAcross {len(ok)} season(s), {int(ok['n_gws'].sum())} gameweeks total:")
        print(f"  Weighted-average MAE: {weighted_mae:.3f}")
        print(f"  Mean rank correlation: {ok['mean_spearman'].mean():.3f}")
        print(f"  Total model XI points: {ok['total_model_xi_points'].sum():.0f}")
        print(f"  Total baseline XI points: {ok['total_baseline_xi_points'].sum():.0f}")
        print(f"  Advantage vs. naive baseline: {ok['advantage_vs_baseline'].sum():+.0f}")

    if "--save" in sys.argv:
        idx = sys.argv.index("--save")
        path = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "historical_backtest_results.csv"
        summary.to_csv(path, index=False)
        print(f"\nSaved to {path}")
