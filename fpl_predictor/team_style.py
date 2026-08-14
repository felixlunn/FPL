"""Classify each team's playing style -- Attacking, Balanced, or Defensive
-- from real match results (goals scored/conceded per game), z-scored
against the rest of the league so the label means "relative to this
season's actual competition", not an arbitrary fixed goals-per-game cutoff.

This is a *style* axis, not a quality one: a team can be simultaneously
excellent at scoring and excellent at defending (which lands as
"Balanced" here -- no lopsided trade-off between the two) versus a team
that leans hard into one at the expense of the other.

The classification requires genuinely above-average performance on
whichever axis a team leans towards (``attack_z``/``defense_z`` each
individually clearing ``balanced_band``), not just relative advantage --
a naive ``attack_z - defense_z`` alone would mislabel a team that's
simply bad at both ends as "Attacking" whenever its attack happens to be
*less* bad than its defense, without it being any good in absolute terms.
``style_score = attack_z - defense_z`` is still reported as a continuous
lean/magnitude indicator for display, just not used alone for the label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BALANCED_BAND = 0.5  # |style_score| within this band -> "Balanced"


def _finished_matches_long(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per finished match: team, scored, conceded."""
    cols = ["team", "scored", "conceded"]
    if fixtures_df is None or fixtures_df.empty:
        return pd.DataFrame(columns=cols)
    fx = fixtures_df.copy()
    if "finished" in fx.columns:
        fx = fx[fx["finished"] == True]  # noqa: E712
    needed = {"team_h", "team_a", "team_h_score", "team_a_score"}
    if not needed.issubset(fx.columns):
        return pd.DataFrame(columns=cols)
    fx = fx.dropna(subset=["team_h_score", "team_a_score"])
    if fx.empty:
        return pd.DataFrame(columns=cols)
    home = fx.rename(columns={"team_h": "team", "team_h_score": "scored", "team_a_score": "conceded"})[cols]
    away = fx.rename(columns={"team_a": "team", "team_a_score": "scored", "team_h_score": "conceded"})[cols]
    return pd.concat([home, away], ignore_index=True)


def compute_team_styles(fixtures_df: pd.DataFrame, teams_df: pd.DataFrame, balanced_band: float = BALANCED_BAND) -> pd.DataFrame:
    """One row per team with its playing-style classification. Teams with
    no finished matches yet (e.g. pre-season) aren't included -- there's
    nothing real to classify from yet.
    """
    long_df = _finished_matches_long(fixtures_df)
    if long_df.empty:
        return pd.DataFrame()

    agg = long_df.groupby("team").agg(games_played=("scored", "size"), goals_scored_per_game=("scored", "mean"), goals_conceded_per_game=("conceded", "mean")).reset_index()
    if len(agg) < 2:
        return pd.DataFrame()  # z-scores are meaningless against a league of one

    attack_std = agg["goals_scored_per_game"].std(ddof=0)
    concede_std = agg["goals_conceded_per_game"].std(ddof=0)
    agg["attack_z"] = (agg["goals_scored_per_game"] - agg["goals_scored_per_game"].mean()) / attack_std if attack_std > 0 else 0.0
    agg["defense_z"] = (agg["goals_conceded_per_game"].mean() - agg["goals_conceded_per_game"]) / concede_std if concede_std > 0 else 0.0
    agg["style_score"] = agg["attack_z"] - agg["defense_z"]
    agg["style"] = np.select(
        [
            (agg["attack_z"] > balanced_band) & (agg["attack_z"] >= agg["defense_z"]),
            (agg["defense_z"] > balanced_band) & (agg["defense_z"] > agg["attack_z"]),
        ],
        ["Attacking", "Defensive"],
        default="Balanced",
    )

    team_names = teams_df.set_index("id")["name"].to_dict() if teams_df is not None and not teams_df.empty else {}
    agg["team_name"] = agg["team"].map(team_names).fillna(agg["team"].astype(str))
    agg = agg.rename(columns={"team": "team_id"})
    return agg.sort_values("style_score", ascending=False).reset_index(drop=True)
