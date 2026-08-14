"""One-off comparisons validating recent modelling choices -- the Tweedie
objective, defensive-contribution features, FDR fixture difficulty, and
the minutes/rotation-risk model -- against the walk-forward backtest, so
those choices are demonstrated to help rather than merely assumed.

Not part of the live app (each variant retrains the whole model per
backtested gameweek, so running all variants takes several times as long
as a single backtest). Run this as a script or from a notebook/REPL when
validating a modelling change:

    python -m fpl_predictor.ablation
"""

from __future__ import annotations

import pandas as pd

from fpl_predictor.backtest import run_backtest
from fpl_predictor.pipeline import DataBundle

# name -> kwargs passed to run_backtest(), each isolating exactly one
# change from the "full" configuration so its effect can be read off
# directly rather than confounded with any other change.
VARIANTS: dict[str, dict] = {
    "full (current)": {},
    "no Tweedie (plain L2)": {"objective_params": {"objective": "regression"}},
    "no defensive-contribution features": {"drop_feature_prefixes": ["defensive_actions", "tackles", "clearances_blocks_interceptions", "recoveries", "defensive_contribution"]},
    "no fixture difficulty (FDR)": {"drop_feature_prefixes": ["fixture_difficulty"]},
    "no set-piece priority features": {"drop_feature_prefixes": ["penalties_priority", "set_piece_priority"]},
    "no minutes/rotation model": {"use_minutes_model": False},
}


def run_ablation_study(data: DataBundle, min_train_gws: int = 3, tune: bool = False) -> pd.DataFrame:
    """Run the backtest once per variant and return a comparison table.
    Positive ``vs. full`` values mean the full configuration scored more
    (i.e. the ablated component was pulling its weight).
    """
    rows = []
    full_points = None
    for name, kwargs in VARIANTS.items():
        report = run_backtest(data, min_train_gws=min_train_gws, tune=tune, **kwargs)
        if name == "full (current)":
            full_points = report.total_model_xi_points
        rows.append({
            "variant": name,
            "n_gws": len(report.per_gw),
            "overall_mae": report.overall_mae,
            "mean_spearman": report.mean_spearman,
            "model_xi_points": report.total_model_xi_points,
        })
    out = pd.DataFrame(rows)
    if full_points is not None:
        out["vs_full_points"] = out["model_xi_points"] - full_points
    return out


if __name__ == "__main__":
    from fpl_predictor.pipeline import build_demo_data

    print("Running ablation study on the synthetic demo dataset "
          "(swap in a real DataBundle from fetch_live_data() for a real comparison)...\n")
    demo = build_demo_data(n_gws=16, n_future_gws=2, seed=11)
    result = run_ablation_study(demo, min_train_gws=3, tune=False)
    pd.set_option("display.width", 120)
    print(result.to_string(index=False))
