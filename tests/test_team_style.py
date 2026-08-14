import pandas as pd

from fpl_predictor.team_style import compute_team_styles


def _toy_fixtures():
    # Team 1 "Attackers": high-scoring, ordinary defense -> Attacking.
    # Team 2 "Defenders": low-scoring, excellent defense -> Defensive.
    # Team 3 "Balanced-good": solid at both -> Balanced.
    # Team 4 "Bad-at-both": poor at both -> must stay Balanced, not get
    # mislabeled "Attacking" just for being *less* bad at scoring than
    # conceding (the exact failure mode a naive attack_z - defense_z
    # threshold falls into).
    scorelines = {1: (4, 1), 2: (1, 0), 3: (2, 1), 4: (0, 3)}
    rows, fid = [], 1
    for team, (scored, conceded) in scorelines.items():
        for gw in range(1, 6):
            # Team plays as home against itself-as-"filler" team 5 each time,
            # with team 5 just absorbing whatever the fixture needs.
            rows.append({"id": fid, "event": gw, "team_h": team, "team_a": 5, "team_h_score": scored, "team_a_score": conceded, "finished": True})
            fid += 1
    return pd.DataFrame(rows)


def _toy_teams():
    return pd.DataFrame({"id": [1, 2, 3, 4, 5], "name": ["Attackers", "Defenders", "Balanced-good", "Bad-at-both", "Filler"]})


def test_compute_team_styles_classifies_sensibly():
    styles = compute_team_styles(_toy_fixtures(), _toy_teams())
    by_name = styles.set_index("team_name")
    assert by_name.loc["Attackers", "style"] == "Attacking"
    assert by_name.loc["Defenders", "style"] == "Defensive"
    assert by_name.loc["Balanced-good", "style"] == "Balanced"
    # The key regression case: bad at both ends must not be mislabeled
    # "Attacking" merely for being relatively less bad at scoring.
    assert by_name.loc["Bad-at-both", "style"] == "Balanced"
    assert by_name.loc["Attackers", "style_score"] > by_name.loc["Balanced-good", "style_score"]
    assert by_name.loc["Defenders", "style_score"] < by_name.loc["Balanced-good", "style_score"]


def test_compute_team_styles_handles_no_finished_matches():
    fx = pd.DataFrame([{"id": 1, "event": 1, "team_h": 1, "team_a": 2, "team_h_score": None, "team_a_score": None, "finished": False}])
    assert compute_team_styles(fx, _toy_teams()).empty


def test_compute_team_styles_handles_empty_and_missing_columns():
    assert compute_team_styles(pd.DataFrame(), _toy_teams()).empty
    assert compute_team_styles(pd.DataFrame({"id": [1]}), _toy_teams()).empty
