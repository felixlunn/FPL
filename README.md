# FPL Predictor

A Fantasy Premier League points predictor and squad optimizer, with an
interactive Streamlit web interface for picking the best possible team for
any upcoming gameweek.

## How it works

```
Live FPL API  ──┐
                ├─▶ feature engineering ─▶ per-position LightGBM models ─▶ ILP squad optimizer ─▶ web UI
Historical      │        (fpl_predictor/features.py)   (fpl_predictor/model.py)  (fpl_predictor/optimizer.py)
results (Elo) ──┘
```

- **Data**: live player/fixture data comes from the official FPL API
  (`fantasy.premierleague.com/api`), fetched in parallel and cached to disk
  (`fpl_predictor/data_sources/fpl_api.py`). If the API can't be reached,
  the app automatically falls back to a small synthetic "demo" league so
  it's still usable offline.
- **Fixture difficulty**: a team-strength Elo rating is computed from the
  historical Premier League results bundled in this repo
  (`all-euro-data-*.csv`, `fpl_predictor/data_sources/team_strength.py`).
  This replaces the placeholder `opponent_strength = 0` from the original
  prototype with a real, backtestable signal — a player facing a
  relegation-threatened side is expected to score more than the same
  player facing the league leaders.
- **Features**: rolling/lag stats (points, minutes, bps, ICT index,
  expected goals/assists where available) computed strictly from *past*
  gameweeks, plus fixture difficulty (opponent strength, home/away). See
  `fpl_predictor/features.py`.
- **Model**: a separate LightGBM regressor per position (GKP/DEF/MID/FWD),
  since clean sheets matter for defenders and goals matter for attackers.
  Hyperparameters are chosen by walk-forward (`TimeSeriesSplit`)
  cross-validation, never on future gameweeks, so validation MAE reflects
  real predictive skill (`fpl_predictor/model.py`).
- **Forecasting**: predictions are projected onto specific future
  gameweeks using the *actual* fixture list, correctly handling blank
  gameweeks (no fixture → 0 points) and double gameweeks (two fixtures →
  points summed), then scaled by each player's estimated probability of
  playing (`fpl_predictor/forecast.py`).
- **Optimization**: squad selection, starting XI, and transfer
  suggestions are all solved as integer linear programs (PuLP/CBC)
  against the real FPL rules (£100m budget, 15-man squad, max 3 per club,
  valid formations), so results are the actual optimum given the model's
  predictions — not a greedy approximation
  (`fpl_predictor/optimizer.py`).

## Running the web app

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (defaults to http://localhost:8501).

The sidebar lets you pick a gameweek, set your budget and free transfers,
and switch between live data and offline demo data. Tabs:

- **Optimal Squad** — the best possible 15 under budget/rules, plus the
  optimal starting XI, captain, and bench for the selected gameweek.
- **My Squad** — enter your own 15 players and see your best XI/captain,
  with validation against all squad rules.
- **Transfers** — best transfer(s) from your squad, only recommending a
  `-4` hit when the model expects it to pay off.
- **Player Explorer** — filterable/sortable table of every player's
  predicted points and points-per-million across upcoming gameweeks.
- **Model Insights** — per-position validation error and feature
  importance, plus the underlying team-strength ratings.

### A note on network access

This project needs outbound HTTPS access to `fantasy.premierleague.com`
to fetch live data. Some sandboxed/CI environments block that host — the
app detects this and switches to the bundled synthetic demo league
automatically (with a banner explaining why), so the UI and model
pipeline are still fully exercisable offline.

## Running the tests

```bash
pytest tests/ -q
```

The test suite runs the entire pipeline (features → model → forecast →
optimizer) against the synthetic demo dataset, so it needs no network
access and is fully deterministic.

## Project layout

```
fpl_predictor/
  config.py              constants, team-name aliases
  data_sources/
    fpl_api.py            live FPL API fetch + disk cache
    team_strength.py       Elo ratings from historical results
  features.py             rolling/lag feature engineering
  model.py                per-position LightGBM training + tuning
  forecast.py              project models onto future gameweeks
  optimizer.py             ILP squad/XI/transfer optimization
  pipeline.py              orchestration + synthetic demo dataset
app.py                     Streamlit web interface
tests/                     pytest suite (offline, uses demo data)
all-euro-data-*.csv        historical PL results (football-data.co.uk)
```

## Limitations / next steps

- The Elo model only uses final scores; incorporating underlying-stats
  (xG) team ratings would sharpen fixture difficulty further.
- Playing-probability comes straight from the FPL API's own status/news
  field; a dedicated minutes-prediction model would improve rotation risk
  handling for bench players.
- Transfer search caps at a small number of transfers per run (adjustable
  in the UI) since it re-solves the full ILP per candidate transfer count.
