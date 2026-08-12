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

# Historical match-result CSVs (football-data.co.uk, EPL "E0" division) used
# to derive team-strength (Elo) ratings that feed the fixture-difficulty
# features of the fantasy points model.
HISTORICAL_RESULTS_GLOB = os.path.join(ROOT_DIR, "all-euro-data-*.csv")

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
# Elo model
# -----------------------------
ELO_INITIAL = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 60.0  # elo points added to the home side's expectation

# -----------------------------
# Team name normalisation
# -----------------------------
# football-data.co.uk and the FPL API sometimes spell club names differently.
# This maps a variety of spellings to a single canonical short name so that
# Elo ratings computed from historical results can be joined onto the FPL
# team list. Keys are lower-cased for matching.
TEAM_NAME_ALIASES = {
    "man united": "man utd", "manchester united": "man utd", "man utd": "man utd",
    "man city": "man city", "manchester city": "man city",
    "spurs": "spurs", "tottenham": "spurs", "tottenham hotspur": "spurs",
    "wolves": "wolves", "wolverhampton wanderers": "wolves", "wolverhampton": "wolves",
    "nott'm forest": "nott'm forest", "nottingham forest": "nott'm forest", "forest": "nott'm forest",
    "newcastle": "newcastle", "newcastle united": "newcastle",
    "west ham": "west ham", "west ham united": "west ham",
    "leicester": "leicester", "leicester city": "leicester",
    "leeds": "leeds", "leeds united": "leeds",
    "brighton": "brighton", "brighton and hove albion": "brighton", "brighton & hove albion": "brighton",
    "sheffield united": "sheffield utd", "sheffield utd": "sheffield utd",
    "west brom": "west brom", "west bromwich albion": "west brom",
    "cardiff": "cardiff", "cardiff city": "cardiff",
    "huddersfield": "huddersfield", "huddersfield town": "huddersfield",
    "stoke": "stoke", "stoke city": "stoke",
    "swansea": "swansea", "swansea city": "swansea",
    "hull": "hull", "hull city": "hull",
    "norwich": "norwich", "norwich city": "norwich",
    "qpr": "qpr", "queens park rangers": "qpr",
    "afc bournemouth": "bournemouth", "bournemouth": "bournemouth",
    "brentford": "brentford",
    "burnley": "burnley",
    "fulham": "fulham",
    "watford": "watford",
    "everton": "everton",
    "southampton": "southampton",
    "aston villa": "aston villa",
    "crystal palace": "crystal palace",
    "arsenal": "arsenal",
    "chelsea": "chelsea",
    "liverpool": "liverpool",
    "ipswich": "ipswich", "ipswich town": "ipswich",
    "luton": "luton", "luton town": "luton",
    "sunderland": "sunderland",
    "middlesbrough": "middlesbrough",
    "birmingham": "birmingham",
    "reading": "reading",
    "blackburn": "blackburn",
    "wigan": "wigan",
    "bolton": "bolton",
    "portsmouth": "portsmouth",
    "derby": "derby",
    "blackpool": "blackpool",
    "charlton": "charlton",
}


def canonical_team_name(name: str) -> str:
    """Normalise a club name from either data source to a canonical key."""
    if not isinstance(name, str):
        return name
    key = name.strip().lower()
    return TEAM_NAME_ALIASES.get(key, key)
