"""Quantifies how well FPL's underlying match-performance stats actually
correlate with the points a player scores.

These stats (expected goals/assists, ICT index and its influence/
creativity/threat components) are themselves derived from Opta (Stats
Perform) match event data licensed to the Premier League/FPL -- a direct
raw Opta feed integration would need a separate commercial license this
project doesn't have and there's no free public alternative, but these
aggregate per-gameweek figures are already Opta-derived and already flow
through the FPL API into ``history_df``, so this answers the same
underlying question ("does strong match performance predict FPL points?")
with data already on hand, quantified rather than just fed into the model
as an opaque input.

Correlations here are *same-gameweek* (stat and points from the same
match) -- i.e. "how much does this performance signal co-occur with
points", which is a different, complementary question to the *predictive*
relationship the trained model learns from lagged/rolling versions of
these same stats (see the feature-importance charts in Model Insights).
"""

from __future__ import annotations

import pandas as pd

from fpl_predictor.config import POS_MAP

# Every one of these is Opta-derived: xG/xA/xGI from Opta's shot/pass
# event data, ICT index and its three components computed by FPL from
# Opta match data, bps likewise built from Opta-sourced match actions.
CANDIDATE_STATS = [
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "ict_index", "influence", "creativity", "threat", "bps",
]

MIN_ROWS = 10  # below this, a correlation coefficient is too noisy to report


def compute_stat_correlations(history_df: pd.DataFrame, label_col: str = "total_points") -> pd.DataFrame:
    """Pearson (linear) and Spearman (rank) correlation of each available
    underlying stat against points scored that same gameweek.
    """
    if history_df.empty or label_col not in history_df.columns:
        return pd.DataFrame()
    rows = []
    for col in CANDIDATE_STATS:
        if col not in history_df.columns:
            continue
        sub = history_df[[col, label_col]].dropna()
        if len(sub) < MIN_ROWS or sub[col].std() == 0:
            continue
        rows.append({
            "stat": col,
            "pearson_r": sub[col].corr(sub[label_col], method="pearson"),
            "spearman_r": sub[col].corr(sub[label_col], method="spearman"),
            "n": len(sub),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("pearson_r", ascending=False).reset_index(drop=True) if not out.empty else out


def compute_stat_correlations_by_position(history_df: pd.DataFrame, players_df: pd.DataFrame, label_col: str = "total_points") -> pd.DataFrame:
    """Same, split per position -- e.g. defensive-minded stats correlate
    with points very differently for a defender than for a forward.
    """
    if history_df.empty or "player_id" not in history_df.columns or players_df.empty:
        return pd.DataFrame()
    merged = history_df.merge(players_df[["id", "element_type"]].rename(columns={"id": "player_id"}), on="player_id", how="left")
    frames = []
    for pos_id, pos_label in POS_MAP.items():
        corr = compute_stat_correlations(merged[merged["element_type"] == pos_id], label_col)
        if not corr.empty:
            corr.insert(0, "position", pos_label)
            frames.append(corr)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
