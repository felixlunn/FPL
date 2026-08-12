"""Team-strength (Elo) ratings derived from historical Premier League results.

This module generalises the standalone match-result predictor that used to
live at the repo root (``FPL Predictor.py``). Instead of predicting match
outcomes directly, it produces a running Elo rating per club that is used
downstream as a fixture-difficulty feature for the fantasy-points model:
a player facing a weak opponent (low opponent Elo relative to their own
team) should be expected to score more than the same player facing a
title-chasing side.

The ratings are built purely from the historical football-data.co.uk result
files bundled in the repo (``all-euro-data-*.csv``, English Premier League
"E0" division), so they require no network access and are fully
reproducible/testable offline.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fpl_predictor.config import (
    ELO_HOME_ADVANTAGE,
    ELO_INITIAL,
    ELO_K,
    HISTORICAL_RESULTS_GLOB,
    canonical_team_name,
)


@dataclass
class EloRatings:
    """Result of running the Elo model over historical results.

    Attributes:
        history: one row per match with the pre-match ratings of both sides
            (useful for backtesting / feature joins on past gameweeks).
        current: dict mapping canonical team name -> latest Elo rating.
    """

    history: pd.DataFrame
    current: dict

    def strength(self, team_name: str) -> float:
        """Latest known Elo rating for a team, falling back to the mean."""
        key = canonical_team_name(team_name)
        if key in self.current:
            return self.current[key]
        if self.current:
            return float(np.mean(list(self.current.values())))
        return ELO_INITIAL


def _load_historical_results(csv_glob: str = HISTORICAL_RESULTS_GLOB) -> pd.DataFrame:
    files = sorted(glob.glob(csv_glob))
    if not files:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals", "away_goals"])
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "Div" in df.columns:
            df = df[df["Div"] == "E0"]
        needed = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        if not needed.issubset(df.columns):
            continue
        df = df.rename(columns={
            "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
            "FTHG": "home_goals", "FTAG": "away_goals",
        })
        df = df[["date", "home_team", "away_team", "home_goals", "away_goals"]].copy()
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals", "away_goals"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce")
    out = out.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    out["home_team"] = out["home_team"].map(canonical_team_name)
    out["away_team"] = out["away_team"].map(canonical_team_name)
    out = out.sort_values("date").reset_index(drop=True)
    return out


def compute_elo_ratings(
    results: pd.DataFrame | None = None,
    k: float = ELO_K,
    home_advantage: float = ELO_HOME_ADVANTAGE,
    initial: float = ELO_INITIAL,
) -> EloRatings:
    """Run a simple Elo model, with home-field advantage, over match history.

    Elo naturally decays outdated form (a club that was strong five seasons
    ago but has since declined converges back towards its recent level) and
    needs no external data beyond final scores, which keeps this fixture
    difficulty signal robust and dependency-free.
    """
    if results is None:
        results = _load_historical_results()

    teams = pd.unique(pd.concat([results["home_team"], results["away_team"]])) if not results.empty else []
    elo = {t: initial for t in teams}
    rows = []
    for _, row in results.iterrows():
        ht, at = row["home_team"], row["away_team"]
        hg, ag = row["home_goals"], row["away_goals"]
        elo.setdefault(ht, initial)
        elo.setdefault(at, initial)

        rows.append({
            "date": row["date"], "home_team": ht, "away_team": at,
            "pre_home_elo": elo[ht], "pre_away_elo": elo[at],
        })

        exp_h = 1.0 / (1.0 + 10 ** (((elo[at]) - (elo[ht] + home_advantage)) / 400.0))
        exp_a = 1.0 - exp_h
        if hg > ag:
            s_h, s_a = 1.0, 0.0
        elif hg < ag:
            s_h, s_a = 0.0, 1.0
        else:
            s_h, s_a = 0.5, 0.5

        # Scale the update by goal difference so big wins move ratings more
        # than narrow ones (a common, well-tested Elo refinement for football).
        margin_mult = np.log1p(abs(hg - ag)) + 1.0
        elo[ht] += k * margin_mult * (s_h - exp_h)
        elo[at] += k * margin_mult * (s_a - exp_a)

    history = pd.DataFrame(rows)
    return EloRatings(history=history, current=dict(elo))


def build_team_strength_table(ratings: EloRatings | None = None) -> pd.DataFrame:
    """Convenience: current Elo ratings as a tidy, ranked DataFrame."""
    ratings = ratings or compute_elo_ratings()
    if not ratings.current:
        return pd.DataFrame(columns=["team", "elo"])
    df = pd.DataFrame(sorted(ratings.current.items(), key=lambda kv: -kv[1]), columns=["team", "elo"])
    df["rank"] = np.arange(1, len(df) + 1)
    return df
