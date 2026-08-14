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
  output (from the FPL API's `history_past`), scaled by fixture
  difficulty **and by their share of last season's available minutes**
  (`fpl_predictor/forecast.py`, `predict_cold_start_gameweek`) -- the
  latter matters a lot: a player with a great scoring rate from a
  handful of substitute cameos must not outrank a nailed-on regular just
  because their small-sample rate-when-played happens to be higher, and
  per-90 rate alone can't tell the two apart. Real players, just a
  simpler model, until GW1 results land and full model training resumes
  automatically.
- **Prediction intervals for captaincy**: alongside the main point
  estimate, a separate LightGBM quantile model predicts each player's
  10th/90th percentile range (`fpl_predictor/quantile_model.py`) --
  captaincy is then a genuine choice between a safe, tightly-bunched
  pick and a high-ceiling differential that happens to share the same
  average, not a guess. See the **Captaincy** panel in Optimal Squad.
- **Team playing style**: `fpl_predictor/team_style.py` classifies each
  club as Attacking, Balanced, or Defensive from real match results
  (goals scored/conceded per game, z-scored against the rest of the
  league) -- a *style* axis, not a quality one, so a team that's simply
  bad at both ends doesn't get mislabeled "Attacking" just for being
  relatively less bad at scoring than conceding. See the **Team playing
  styles** panel in Model Insights.
- **Underlying stats vs. points**: `fpl_predictor/stats_correlation.py`
  quantifies how well FPL's underlying match-performance stats -- xG/xA,
  ICT index and its components, all themselves derived from Opta match
  event data licensed to the Premier League/FPL -- actually correlate
  with points scored that gameweek. A raw Opta feed integration would
  need a separate commercial license this project doesn't have; this
  answers the same question ("does strong match performance predict FPL
  points?") using the Opta-derived stats already available via the FPL
  API. See the **Underlying stats vs. actual points** panel in Model
  Insights.
- **Optimization**: squad selection, starting XI, and transfer
  suggestions are all solved as integer linear programs (PuLP/CBC)
  against the real FPL rules (£100m budget, 15-man squad, max 3 per club,
  valid formations), so results are the actual optimum given the model's
  predictions — not a greedy approximation
  (`fpl_predictor/optimizer.py`).
- **Multiple transfer strategies**: `suggest_transfer_options` returns
  three differently-motivated plans side by side rather than one number to
  take or leave — **Optimal** (best net predicted points, hits included
  when they pay off), **Safe** (best plan using only your free
  transfer(s), never risks a −4), and **Differential** (biases toward
  lower-ownership players, for climbing rank against a mini-league full of
  the same template picks rather than just maximizing raw expected
  points). See the **Transfers** tab.
- **Fixture ticker**: every player table shows a compact "next 5"
  fixture string (opponent + home/away), and the Season Planner tab has a
  full team × gameweek grid colour-coded by FPL's FDR, so squad/transfer
  decisions aren't made blind to what's actually coming up
  (`fpl_predictor.data_sources.fpl_api.team_fixture_strings`/
  `fixture_ticker_table`).
- **Season planner**: `fpl_predictor/season_planner.py` rolls the same
  trained models forward across several future gameweeks (rather than
  just the single next one the main pipeline targets) to chart a squad's
  points trajectory and heuristically flag the best window for each chip
  — **Wildcard** (biggest gap between an unconstrained optimal squad and
  yours), **Free Hit** (worst blank gameweek), **Bench Boost** (best
  double-gameweek coverage), **Triple Captain** (single highest predicted
  ceiling). These are transparent, stated-reason heuristics meant to
  prompt a closer look, not another ILP claiming one provably-correct
  answer — chip timing is a judgement call. See the **Season Planner** tab.
- **Top-manager learnings**: `fpl_predictor/manager_insights.py` samples
  the top-ranked managers in FPL's public "Overall" classic league (fully
  public data, no login needed) for their chip-usage and captaincy
  patterns. Important constraint: manager-level pick/chip history only
  exists for the *currently in-progress* season — it resets at each
  season boundary, unlike the player-level stats the historical backtest
  above uses — so this tracks the current season's leaders as it unfolds
  rather than pulling last year's winners retrospectively, which the live
  API no longer exposes once a new season has started. See the **Top
  managers this season** panel in Model Insights.
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
- **Historical backtesting**: `fpl_predictor/historical_data.py` +
  `fpl_predictor/historical_backtest.py` (run via
  `python -m fpl_predictor.historical_backtest`) replay the same
  walk-forward backtest against *real* past FPL seasons, fetched from the
  public [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
  archive (not an official FPL data source; the live API only exposes the
  current season's per-gameweek history, so this is the only way to
  validate against real prior seasons). See "Historical backtest results"
  below.

## Running the web app

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (defaults to http://localhost:8501).

The app always targets the single upcoming gameweek (shown in the
sidebar). The budget is fixed at the real FPL squad budget (£100m, not
user-adjustable); use the sidebar to set your free transfers and to
switch between live data and offline demo data. Tabs:

- **Optimal Squad** — the best possible 15 under budget/rules, plus the
  optimal starting XI, captain, and bench for the selected gameweek, and
  a floor-vs-ceiling captaincy comparison for the squad.
- **My Squad** — enter your own 15 players and see your best XI/captain,
  with validation against all squad rules.
- **Transfers** — three side-by-side transfer strategies (Optimal / Safe
  / Differential) from your squad, each with its own stated rationale,
  and each in/out player shown with their next-5 fixture run.
- **Season Planner** — a multi-gameweek look-ahead: your squad's
  predicted-points trend over the next several gameweeks, heuristic chip
  timing suggestions (Wildcard/Free Hit/Bench Boost/Triple Captain), and
  a colour-coded fixture-difficulty ticker across the same window.
- **Player Explorer** — filterable/sortable table of every player's
  predicted points, start probability, points-per-million, and next-5
  fixture run across upcoming gameweeks.
- **Model Insights** — per-position validation error, minutes-model
  Brier score, feature importance, the upcoming gameweek's fixture
  difficulty ratings, each team's playing-style classification, how well
  the underlying (Opta-derived) match-performance stats actually
  correlate with points, and (opt-in) the current season's top real
  managers' chip-usage patterns.
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

## Historical backtest results

Real answer to "has this been validated against previous years' results":
`python -m fpl_predictor.historical_backtest`, walk-forward against the
last 5 completed FPL seasons fetched from the vaastav archive (retraining
at every gameweek using only earlier data, same harness as the ablation
study above, `min_train_gws=4`). "Baseline" is the same naive "chase last
week's top scorers" comparison the backtest tab uses.

| Season | Gameweeks | MAE | Rank corr. | Model XI pts | Baseline XI pts | Advantage |
|---|---|---|---|---|---|---|
| 2025-26 | 34 | 0.962 | 0.692 | 1733 | 1424 | **+309** |
| 2024-25 | 34 | 1.033 | 0.634 | 1747 | 1452 | **+295** |
| 2023-24 | 34 | 0.919 | 0.658 | 1670 | 1429 | **+241** |
| 2022-23 | 33 | 1.014 | 0.678 | 1661 | 1269 | **+392** |
| 2021-22 | 34 | 1.086 | 0.643 | 1674 | 1419 | **+255** |
| **Weighted avg / total** | **169** | **1.003** | **0.661** | **8485** | **6993** | **+1492** |

Takeaways:

- **This is the real validation, and it holds up.** Across 169 real
  gameweeks over 5 seasons, the model's picked XI outscored the naive
  baseline by +1492 points in total -- roughly **+8.8 points/gameweek**
  on average, consistently positive in every single season, not just on
  average. That's the difference between the two strategies over a full
  season played out for real.
- **MAE of ~1.0 point/player/gameweek** on real data is well within a
  sensible range for a game this variance-heavy (a single goal, assist,
  clean sheet, or bonus-point swing is worth 2-6+ points on its own) --
  and notably *tighter* than the ~2.76 MAE on the synthetic demo data,
  because the demo generator's noise is deliberately heavier than a real
  season's to make sure the ablation study had something to detect.
- **Rank correlation of ~0.66** means the model reliably orders players
  by who'll actually score well that gameweek, not just gets the average
  magnitude right -- which is what selection (not just point-estimate
  accuracy) actually depends on.
- **Consistency across 5 different seasons** (2021-22 through 2025-26,
  spanning real squad turnover, rule changes like the 2025/26 defensive-
  contribution points, and very different title/relegation races) is the
  important signal here -- a single good season could be luck, five in a
  row with the same walk-forward harness and no lookahead is not.
- Every season here ran with `tune=False` (the harness's default for
  speed -- retraining every gameweek across 5 seasons already means
  hundreds of model fits without also re-running a hyperparameter search
  at each one), so these numbers are a *lower bound*: a per-season tuned
  run (`tune=True`, slower) is one of the more promising remaining levers
  if this ever needs squeezing further.

### A note on network access

This project needs outbound HTTPS access to `fantasy.premierleague.com`
to fetch live data. Some sandboxed/CI environments block that host — the
app detects this and switches to the bundled synthetic demo league
automatically (with a banner explaining why), so the UI and model
pipeline are still fully exercisable offline.

`fpl_predictor/historical_backtest.py` and `historical_data.py` have a
separate network dependency: `raw.githubusercontent.com` (the public
vaastav/Fantasy-Premier-League archive), independent of the live FPL
API. An environment can have one reachable without the other.

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
  quantile_model.py         10th/90th percentile floor-ceiling model
  forecast.py              project models onto future gameweeks
  optimizer.py             ILP squad/XI/transfer optimization
  pipeline.py              orchestration + synthetic demo dataset
  backtest.py               walk-forward validation over past gameweeks
  ablation.py               component-by-component validation via the backtest
  historical_data.py        real past-season data from the vaastav archive
  historical_backtest.py    backtest replayed against real past seasons
  team_style.py             Attacking/Balanced/Defensive classification per team
  stats_correlation.py      underlying (Opta-derived) stats vs. actual points
  season_planner.py          multi-gameweek forecast + chip-timing heuristics
  manager_insights.py        top-manager chip/captain patterns (public Overall league)
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
- Team playing style is inferred purely from goals scored/conceded --
  real tactical identity (shot volume, possession, pressing intensity)
  needs shot/event-level data this project doesn't have access to
  (see the Opta note above). Goals are a noisier proxy: a team can look
  "Balanced" simply because a small sample of games hasn't yet separated
  its scoring and defending form, not because it's tactically balanced.
- The underlying-stats correlation panel reports *same-gameweek*
  correlation (did strong performance and points coincide), which is a
  different question from *predictive* value (does this gameweek's stat
  predict next gameweek's points, which is what the trained model's
  lagged/rolling features actually use) -- read the two alongside each
  other, not as substitutes.
- Chip-timing suggestions in the Season Planner are transparent
  heuristics (stated blank/double-gameweek counts, squad-quality gaps,
  ceiling comparisons), not another ILP with one provably-optimal answer
  -- real chip strategy also depends on things this model doesn't see
  (price-rise timing, a manager's own risk tolerance, rank position in a
  private league), so treat these as a prompt to look closer at a
  gameweek, not an instruction to follow blindly.
- Top-manager insights only ever reflect the *current* season as it's
  being played -- the live FPL API doesn't retain individual managers'
  gameweek-by-gameweek picks or chip timing once a season ends, so
  there's no way to retrospectively pull last year's Overall-league
  winners' decisions the way `historical_data.py` can for player-level
  stats. Early in a season (or between seasons) this panel will have
  little or nothing to show, by construction rather than a bug.
- The Differential transfer strategy uses a fixed ownership-penalty
  weight (`DIFFERENTIAL_OWNERSHIP_WEIGHT = 0.04` predicted points per %
  owned) rather than one tuned against real rank-climbing outcomes --
  reasonable by inspection, not empirically validated the way the main
  model's hyperparameters are.
