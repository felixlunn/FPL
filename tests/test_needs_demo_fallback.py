"""Regression coverage for a real bug: the app used to treat "fetch
succeeded but history_df is empty" (the ordinary pre-season state -- zero
gameweeks played yet, discovered when live network access first started
working against a season that hadn't kicked off) the same as "fetch
failed", and silently discarded real player data in favour of the
synthetic demo league. needs_demo_fallback must distinguish the two.
"""

import pandas as pd

from fpl_predictor.pipeline import DataBundle, needs_demo_fallback


def _bundle(ok: bool, players_empty: bool, history_empty: bool) -> DataBundle:
    players_df = pd.DataFrame() if players_empty else pd.DataFrame({"id": [1, 2], "web_name": ["A", "B"]})
    history_df = pd.DataFrame() if history_empty else pd.DataFrame({"player_id": [1], "round": [1], "total_points": [2]})
    return DataBundle(players_df, pd.DataFrame(), pd.DataFrame(), history_df, meta={"ok": ok})


def test_no_fallback_when_fetch_ok_even_with_empty_history():
    # The exact real-world case this regression-tests: live fetch works,
    # season just hasn't started, so there's no gameweek history yet --
    # this must NOT trigger a fallback to fake demo data.
    assert needs_demo_fallback(_bundle(ok=True, players_empty=False, history_empty=True)) is False


def test_no_fallback_when_fetch_ok_and_history_present():
    assert needs_demo_fallback(_bundle(ok=True, players_empty=False, history_empty=False)) is False


def test_fallback_when_fetch_failed():
    assert needs_demo_fallback(_bundle(ok=False, players_empty=True, history_empty=True)) is True


def test_fallback_when_no_players_returned_even_if_marked_ok():
    assert needs_demo_fallback(_bundle(ok=True, players_empty=True, history_empty=True)) is True
