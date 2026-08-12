import pandas as pd

from fpl_predictor.data_sources.fpl_api import fixtures_long_by_team


def _toy_fixtures():
    return pd.DataFrame([
        {"id": 1, "event": 1, "team_h": 10, "team_a": 20, "team_h_difficulty": 2, "team_a_difficulty": 4},
        {"id": 2, "event": 2, "team_h": 20, "team_a": 10, "team_h_difficulty": 3, "team_a_difficulty": 3},
    ])


def test_fixtures_long_by_team_produces_one_row_per_side():
    long_df = fixtures_long_by_team(_toy_fixtures())
    assert len(long_df) == 4
    row = long_df[(long_df["team"] == 10) & (long_df["event"] == 1)].iloc[0]
    assert row["opponent"] == 20
    assert bool(row["was_home"]) is True
    assert row["difficulty"] == 2

    row2 = long_df[(long_df["team"] == 20) & (long_df["event"] == 1)].iloc[0]
    assert row2["opponent"] == 10
    assert bool(row2["was_home"]) is False
    assert row2["difficulty"] == 4


def test_fixtures_long_by_team_handles_missing_difficulty_columns():
    df = pd.DataFrame([{"id": 1, "event": 1, "team_h": 10, "team_a": 20}])  # no FDR columns at all
    long_df = fixtures_long_by_team(df)
    assert len(long_df) == 2
    assert long_df["difficulty"].isna().all()


def test_fixtures_long_by_team_handles_empty_input():
    assert fixtures_long_by_team(pd.DataFrame()).empty
    assert fixtures_long_by_team(None).empty
