"""Tests the aggregation logic only, with the network layer (fetch_json)
mocked out -- this module talks to the live public FPL API (league
standings + per-manager history/picks), so it's never exercised against
the real network in the offline test suite.
"""

from unittest.mock import patch

import fpl_predictor.manager_insights as mi

_STANDINGS = {"standings": {"results": [{"entry": 100, "rank": 1}, {"entry": 200, "rank": 2}, {"entry": 300, "rank": 3}], "has_next": False}}
_EMPTY_STANDINGS = {"standings": {"results": [], "has_next": False}}

_HISTORY = {
    100: {"chips": [{"name": "wildcard", "event": 8}, {"name": "bboost", "event": 20}]},
    200: {"chips": [{"name": "wildcard", "event": 9}]},
    300: {"chips": []},
}

_PICKS_GW10 = {
    100: {"picks": [{"element": 351, "is_captain": True}, {"element": 5, "is_captain": False}]},
    200: {"picks": [{"element": 351, "is_captain": True}]},
    300: {"picks": [{"element": 99, "is_captain": True}]},
}

_ENTRY_INFO = {
    100: {"name": "Test FC", "player_first_name": "Ada", "player_last_name": "Lovelace"},
    400: {"name": "No Picks Yet FC", "player_first_name": "Grace", "player_last_name": "Hopper"},
}


def _fake_fetch_json(url, cache_name=None, ttl_seconds=3600, timeout=15):
    if cache_name and cache_name.startswith("league_"):
        return (_STANDINGS, {"source": "test"})
    if cache_name and cache_name.endswith("_info"):
        entry_id = int(cache_name.split("_")[1])
        return (_ENTRY_INFO.get(entry_id), {"source": "test"})
    if cache_name and "_history" in cache_name:
        entry_id = int(cache_name.split("_")[1])
        return (_HISTORY.get(entry_id, {}), {"source": "test"})
    if cache_name and "_picks_gw" in cache_name:
        entry_id = int(cache_name.split("_")[1])
        return (_PICKS_GW10.get(entry_id, {}), {"source": "test"})
    return (None, {"source": "unavailable"})


def _fake_fetch_json_empty_league(url, cache_name=None, ttl_seconds=3600, timeout=15):
    if cache_name and cache_name.startswith("league_"):
        return (_EMPTY_STANDINGS, {"source": "test"})
    return (None, {"source": "unavailable"})


def test_fetch_top_managers_paginates_and_caps_at_top_n():
    with patch.object(mi, "fetch_json", side_effect=_fake_fetch_json):
        df = mi.fetch_top_managers(top_n=2)
    assert len(df) == 2
    assert list(df["entry"]) == [100, 200]


def test_summarize_top_manager_chips_aggregates_by_chip_and_gw():
    with patch.object(mi, "fetch_json", side_effect=_fake_fetch_json):
        result = mi.summarize_top_manager_chips(top_n=3)
    assert result.ok
    assert result.n_managers_sampled == 3
    usage = result.chip_usage.set_index(["chip", "gw"])["n_managers"]
    assert usage.loc[("wildcard", 8)] == 1
    assert usage.loc[("wildcard", 9)] == 1
    assert usage.loc[("bboost", 20)] == 1


def test_summarize_top_manager_chips_handles_empty_league_gracefully():
    with patch.object(mi, "fetch_json", side_effect=_fake_fetch_json_empty_league):
        result = mi.summarize_top_manager_chips(top_n=50)
    assert not result.ok
    assert "No ranked entries" in result.message
    assert result.chip_usage.empty


def test_summarize_top_manager_captains_counts_captain_choices():
    with patch.object(mi, "fetch_json", side_effect=_fake_fetch_json):
        df = mi.summarize_top_manager_captains(gw=10, top_n=3)
    counts = df.set_index("captain_element")["n_managers"]
    assert counts.loc[351] == 2
    assert counts.loc[99] == 1


def test_summarize_top_manager_captains_empty_when_no_standings():
    with patch.object(mi, "fetch_json", side_effect=_fake_fetch_json_empty_league):
        df = mi.summarize_top_manager_captains(gw=10, top_n=50)
    assert df.empty


def test_fetch_entry_squad_returns_not_available_when_gw_is_none():
    result = mi.fetch_entry_squad(100, gw=None)
    assert not result["ok"]
    assert "isn't available yet" in result["message"]


def test_fetch_entry_squad_returns_squad_on_success():
    with patch.object(mi, "fetch_json", side_effect=_fake_fetch_json):
        result = mi.fetch_entry_squad(100, gw=10)
    assert result["ok"]
    assert result["team_name"] == "Test FC"
    assert result["manager_name"] == "Ada Lovelace"
    assert result["element_ids"] == [351, 5]
    assert result["captain_element"] == 351


def test_fetch_entry_squad_handles_unknown_entry():
    with patch.object(mi, "fetch_json", side_effect=_fake_fetch_json):
        result = mi.fetch_entry_squad(999, gw=10)
    assert not result["ok"]
    assert "No FPL team found" in result["message"]


def test_fetch_entry_squad_handles_picks_not_yet_public():
    # Entry 400 exists (has entry info) but has no picks recorded in the
    # fixture -- simulates a gameweek whose deadline hasn't passed yet.
    with patch.object(mi, "fetch_json", side_effect=_fake_fetch_json):
        result = mi.fetch_entry_squad(400, gw=1)
    assert not result["ok"]
    assert "aren't public" in result["message"]
