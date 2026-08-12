import pandas as pd

from fpl_predictor.data_sources.team_strength import compute_elo_ratings, build_team_strength_table
from fpl_predictor.config import canonical_team_name, ELO_INITIAL


def _toy_results():
    # Team A beats Team B every time it plays -> A's rating should end up
    # clearly above B's, and above a team that never plays (stays at 1500).
    rows = []
    for i in range(10):
        rows.append({"date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=7 * i),
                      "home_team": "team a", "away_team": "team b", "home_goals": 3, "away_goals": 0})
    return pd.DataFrame(rows)


def test_elo_rewards_winning_team():
    ratings = compute_elo_ratings(_toy_results())
    assert ratings.current["team a"] > ratings.current["team b"]
    assert ratings.current["team a"] > ELO_INITIAL
    assert ratings.current["team b"] < ELO_INITIAL


def test_elo_strength_lookup_falls_back_gracefully():
    ratings = compute_elo_ratings(_toy_results())
    # Unknown team should fall back to the mean rather than raising.
    val = ratings.strength("some totally unknown fc")
    assert isinstance(val, float)


def test_canonical_team_name_aliases():
    assert canonical_team_name("Manchester United") == canonical_team_name("Man Utd")
    assert canonical_team_name("Tottenham Hotspur") == canonical_team_name("Spurs")


def test_real_historical_csvs_load_and_produce_ratings():
    ratings = compute_elo_ratings()
    table = build_team_strength_table(ratings)
    assert not table.empty
    # Known top-of-the-table club should exist with a plausible rating.
    assert canonical_team_name("Man City") in ratings.current
    assert table["elo"].max() > table["elo"].min()
