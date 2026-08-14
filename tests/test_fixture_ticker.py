import pandas as pd

from fpl_predictor.data_sources.fpl_api import fixture_ticker_table, team_fixture_strings


def _fixtures_and_teams():
    teams_df = pd.DataFrame({"id": [1, 2, 3], "name": ["Arsenal", "Liverpool", "Chelsea"]})
    fixtures_df = pd.DataFrame([
        {"id": 1, "event": 5, "team_h": 1, "team_a": 2, "team_h_difficulty": 4, "team_a_difficulty": 2},
        {"id": 2, "event": 6, "team_h": 3, "team_a": 1, "team_h_difficulty": 3, "team_a_difficulty": 3},
        # A double gameweek for team 1 (Arsenal) in GW7.
        {"id": 3, "event": 7, "team_h": 1, "team_a": 2, "team_h_difficulty": 2, "team_a_difficulty": 4},
        {"id": 4, "event": 7, "team_h": 3, "team_a": 1, "team_h_difficulty": 3, "team_a_difficulty": 3},
    ])
    return fixtures_df, teams_df


def test_team_fixture_strings_shows_opponent_and_venue():
    fixtures_df, teams_df = _fixtures_and_teams()
    out = team_fixture_strings(fixtures_df, teams_df, from_gw=5, n=5)
    # Team 1 (Arsenal): GW5 home vs Liverpool, GW6 away at Chelsea, then a
    # GW7 double (home vs Liverpool, away at Chelsea) -- team_fixture_strings
    # lists fixtures individually (no double-gameweek grouping; that's what
    # fixture_ticker_table is for).
    assert out[1] == "LIV(H) CHE(A) LIV(H) CHE(A)"


def test_team_fixture_strings_handles_missing_data():
    assert team_fixture_strings(pd.DataFrame(), pd.DataFrame(), from_gw=1) == {}


def test_fixture_ticker_table_marks_blanks_and_doubles():
    fixtures_df, teams_df = _fixtures_and_teams()
    labels, difficulty = fixture_ticker_table(fixtures_df, teams_df, from_gw=5, n_gws=3)
    assert list(labels.columns) == ["GW5", "GW6", "GW7"]
    # Team 2 (Liverpool) has no GW6 fixture -> blank.
    assert labels.at["Liverpool", "GW6"] == "-"
    assert pd.isna(difficulty.at["Liverpool", "GW6"])
    # Team 1 (Arsenal) has a double gameweek in GW7 -> both shown, joined.
    assert "+" in labels.at["Arsenal", "GW7"]
    # Difficulty for the double gameweek cell takes the easier (lower) FDR.
    assert difficulty.at["Arsenal", "GW7"] == 2.0
