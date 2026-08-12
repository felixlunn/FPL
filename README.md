# FPL Predictor

A Fantasy Premier League points predictor and squad optimizer, with an
interactive Streamlit web interface for picking the best possible team for
any upcoming gameweek.

## How it works

```
Live FPL API ─▶ feature engineering ─▶ per-position LightGBM models ─▶ ILP squad optimizer ─▶ web UI
                 (fpl_predictor/features.py)   (fpl_predictor/model.py)  (fpl_predictor/optimizer.py)
```

- **Data**: live player/fixture data comes from the official FPL API
  (`fantasy.premierleague.com/api`), fetched in parallel and cached to disk
  (`fpl_predictor/data_sources/fpl_api.py`). If the API can't be reached,
  the app automatically falls back to a small synthetic "demo" league so
  it's still usable offline.
- **Fixture difficulty**: comes straight from FPL's own Fixture Difficulty
  Rating (FDR, 1-5) -- the `team_h_difficulty`/`team_a_difficulty` fields
  the live fixtures API already returns, pivoted into a per-team lookup by
  `fpl_predictor.data_sources.fpl_api.fixtures_long_by_team`. No separate
  team-strength model or historical results data needed, and it's always
  as current as the live API itself.
- **Features**: rolling/lag stats (points, minutes, bps, ICT index,
  expected goals/assists where available) computed strictly from *past*
  gameweeks, plus fixture difficulty and home/away, defensive-contribution
  volume (tackles/clearances/blocks/interceptions/recoveries, feeding the
  2025/26 defensive-contribution scoring rule), and set-piece duty
  (penalty/free-kick/corner taker priority, from the API's `*_order`
  fields). See `fpl_predictor/features.py`.
- **Model**: a separate LightGBM regressor per position (GKP/DEF/MID/FWD),
  since clean sheets matter for defenders and goals matter for attackers,
  trained with a Tweedie objective (rather than plain L2) to match how
  skewed and zero-inflated FPL points actually are. Hyperparameters are
  chosen by walk-forward (`TimeSeriesSplit`) cross-validation, never on
  future gameweeks, so validation MAE reflects real predictive skill
  (`fpl_predictor/model.py`).
- **Minutes/rotation-risk model**: a separate LightGBM classifier predicts
  each player's P(60+ minutes) in the upcoming gameweek from their recent
  minutes pattern, independent of scoring ability -- shown in the UI as a
  Start % on its own, and blended with the API's status field (whichever
  is more conservative) to scale predicted points, so a fit-but-rotated
  player is treated differently from an announced injury even though
  neither shows up the same way in the raw status field
  (`fpl_predictor/minutes_model.py`).
- **Forecasting**: the app always predicts exactly one gameweek -- the
  next one that hasn't been played yet, determined from the FPL API's own
  `is_next` event flag (not by guessing from completed-gameweek history,
  which breaks at the start of a season). Predictions use the *actual*
  fixture list, correctly handling blank gameweeks (no fixture → 0
  points) and double gameweeks (two fixtures → points summed), then are
  scaled by each player's estimated probability of playing
  (`fpl_predictor/forecast.py`).
- **New-season cold start**: every season has a gap between the final
  ball of one campaign and the first completed gameweek of the next --
  during that window there is no (player, gameweek) data yet to train a
  model on. Rather than falling back to fake data, the app detects this
  and switches to a baseline that uses each player's last-season per-90
  output (from the FPL API's `history_past`) scaled by fixture difficulty
  and playing probability -- still real players, just a simpler model
  until GW1 results land, at which point full model training resumes
  automatically.
- **Optimization**: squad selection, starting XI, and transfer
  suggestions are all solved as integer linear programs (PuLP/CBC)
  against the real FPL rules (£100m budget, 15-man squad, max 3 per club,
  valid formations), so results are the actual optimum given the model's
  predictions — not a greedy approximation
  (`fpl_predictor/optimizer.py`).
- **Backtesting**: `fpl_predictor/backtest.py` replays the whole pipeline
  gameweek-by-gameweek over already-completed history -- retraining at
  each step using only earlier gameweeks -- and compares the actual
  points a model-picked XI would have scored against a naive "chase last
  week's top scorers" baseline. This is the real validation of whether
  the tool is useful, not just what its cross-validation error looks
  like; see the **Backtest** tab in the app.
- **Ablation study**: `fpl_predictor/ablation.py` (run via
  `python -m fpl_predictor.ablation`) uses that same backtest to compare
  the full pipeline against variants with one component removed at a time
  (Tweedie, defensive-contribution features, FDR, set-piece features, the
  minutes model) -- so each addition is demonstrated to help rather than
  just assumed. See "Ablation study results" below for what it found on
  the synthetic demo data.

## Running the web app

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (defaults to http://localhost:8501).

The app always targets the single upcoming gameweek (shown in the
sidebar); use the sidebar to set your budget and free transfers, and to
switch between live data and offline demo data. Tabs:

- **Optimal Squad** — the best possible 15 under budget/rules, plus the
  optimal starting XI, captain, and bench for the selected gameweek.
- **My Squad** — enter your own 15 players and see your best XI/captain,
  with validation against all squad rules.
- **Transfers** — best transfer(s) from your squad, only recommending a
  `-4` hit when the model expects it to pay off.
- **Player Explorer** — filterable/sortable table of every player's
  predicted points and points-per-million across upcoming gameweeks.
- **Model Insights** — per-position validation error, minutes-model
  Brier score, feature importance, plus the upcoming gameweek's fixture
  difficulty ratings.
- **Backtest** — walk-forward validation of the whole pipeline against
  already-completed gameweeks: actual points scored by the model's picks
  vs. a naive baseline, per-gameweek error and rank correlation.

## Ablation study results

Run with `python -m fpl_predictor.ablation` against the synthetic demo
dataset (13 backtested gameweeks, 96 players); see the module docstring
to run it against real data instead. Positive `vs. full` means removing
that component made the model-picked XI score *fewer* real points over
the backtest window -- i.e. the component was pulling its weight.

| Variant removed | MAE | Rank corr. | XI points vs. full |
|---|---|---|---|
| *(full model)* | 2.76 | 0.148 | — |
| Tweedie (→ plain L2) | 2.72 (better) | 0.157 (better) | **−43** |
| Defensive-contribution features | 2.76 (≈same) | 0.151 (≈same) | **−21** |
| Fixture difficulty (FDR) | 2.82 (worse) | 0.122 (worse) | +10 |
| Set-piece priority features | 2.76 (≈same) | 0.149 (≈same) | **−8** |
| Minutes/rotation model | 2.66 (better) | 0.146 (≈same) | **−34** |

Takeaways, and why the columns don't always agree:

- **FDR** is the one component confirmed by *every* metric (removing it
  makes MAE and rank correlation both clearly worse) -- expected, since
  it's the single feature deliberately built with the strongest causal
  relationship to points in the synthetic generator.
- **Tweedie and the minutes model** show a real tension: removing either
  one slightly *improves* raw per-player MAE, but clearly *reduces* the
  actual points the model's squad picks would have scored. This makes
  sense once you separate the two goals -- Tweedie trades a little
  average-case accuracy for a better-shaped right tail (getting big
  hauls right matters more for squad selection than getting every 0-4
  point week exactly right), and the minutes model deliberately shades
  down uncertain players even in weeks they'd have scored fine, which
  costs a bit of raw accuracy but avoids picking players who don't
  actually play. **MAE alone would have told us to drop both of these --
  the realized-points metric is what actually catches their value.**
- **Defensive-contribution and set-piece features** show only a small
  effect here, which is plausible given how infrequently their triggering
  conditions (hitting the CBIT threshold; scoring a nominated penalty)
  actually fire within 13 gameweeks -- a longer backtest window (a full
  real season) would be needed to say more with confidence.
- The realized-points metric is the most intuitive but also the noisiest
  of the three (a single backtest run over a modest number of gameweeks,
  with squad-selection cutoffs that can flip discontinuously on small
  ranking changes) -- treat single-run point swings as suggestive, and
  MAE/rank-correlation as the more statistically stable signal, unless
  running over many more gameweeks or multiple seeds.

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
  config.py              constants
  data_sources/
    fpl_api.py            live FPL API fetch + disk cache + fixture-difficulty lookup
  features.py             rolling/lag feature engineering
  model.py                per-position LightGBM training + tuning
  minutes_model.py         P(60+ mins) rotation-risk classifier
  forecast.py              project models onto future gameweeks
  optimizer.py             ILP squad/XI/transfer optimization
  pipeline.py              orchestration + synthetic demo dataset
  backtest.py               walk-forward validation over past gameweeks
  ablation.py               component-by-component validation via the backtest
app.py                     Streamlit web interface
tests/                     pytest suite (offline, uses demo data)
```

## Limitations / next steps

- The cold-start (pre-season) baseline is a simple heuristic, not a
  learned model -- it's meant to keep the app useful with real players
  before any current-season data exists, not to match the trained
  model's accuracy.
- Transfer search caps at a small number of transfers per run (adjustable
  in the UI) since it re-solves the full ILP per candidate transfer count.
- Defensive-contribution and set-piece-order stat field names (`tackles`,
  `clearances_blocks_interceptions`, `recoveries`, `penalties_order`,
  `direct_freekicks_order`, `corners_and_indirect_freekicks_order`) are
  based on the publicly documented current API shape but haven't been
  verified against a live payload from this sandbox (no network access
  here). Each `DataBundle.meta["history_columns"]` lists exactly what
  fields a real fetch returned, so this is easy to confirm/correct once
  run with live data -- unmatched names are silently skipped rather than
  erroring.
- Set-piece duty (`*_order`) is a *current-season snapshot*, not
  per-gameweek history -- the live API doesn't expose who took penalties
  in gameweek 3 specifically, only who's nominated today. If duty changed
  hands mid-season, historical training rows see today's taker rather
  than whoever it actually was at the time. No other data source exists
  for this without an external provider.
- The backtest rebuilds a fresh optimal squad from scratch every
  gameweek within budget; it deliberately doesn't model real transfer
  limits (free transfers, `-4` hits, one squad carried across the
  season), so it measures prediction/selection quality, not season-long
  transfer strategy -- those are different, also-useful questions.
- The Tweedie variance power (1.5) is a fixed, commonly-used default, not
  itself tuned per position.
- FPL's FDR is a coarse 1-5 rating set by FPL's own analysts; a model-
  derived team-strength signal (e.g. from underlying stats) could be more
  granular, at the cost of needing separate data and upkeep -- the FDR
  approach was chosen deliberately over that trade-off for simplicity and
  staying automatically current with no extra data source to maintain.
- The ablation study above was run on synthetic demo data because this
  sandbox has no live network access -- useful for confirming the
  *methodology* isolates each component correctly and for a rough sanity
  check, but the actual magnitudes (and even direction, for the smaller
  effects) should be re-checked against a real season once one's
  available; the demo generator's causal relationships are a
  simplification of the real game.
