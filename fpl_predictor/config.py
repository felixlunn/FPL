"""Shared constants and configuration for the FPL predictor."""

import os

# -----------------------------
# FPL API
# -----------------------------
FPL_BASE = "https://fantasy.premierleague.com/api/"

# -----------------------------
# Paths
# -----------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT_DIR, ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------------
# FPL game constants
# -----------------------------
SEED = 42

POS_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
POS_COUNTS = {1: 2, 2: 5, 3: 5, 4: 3}  # full 15-man squad
STARTING_XI_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
STARTING_XI_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
STARTING_XI_SIZE = 11

MAX_SQUAD_COST = 100.0
MAX_PER_TEAM = 3
MAX_FPL_GW = 38

# -----------------------------
# Fixture difficulty
# -----------------------------
# FPL's own Fixture Difficulty Rating (1 = easiest, 5 = hardest), returned
# per-fixture by the live API (team_h_difficulty / team_a_difficulty on
# fpl_predictor.data_sources.fpl_api.fixtures_df_from_json). Used as the
# neutral default when a fixture's rating is missing.
NEUTRAL_FIXTURE_DIFFICULTY = 3.0
