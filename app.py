"""FPL Predictor -- Streamlit web interface.

Run with:  streamlit run app.py

Ties together the data/feature/model/optimizer pipeline in
``fpl_predictor/`` into an interactive tool for picking the best possible
squad, analysing your own team, and finding the best transfers, for any
upcoming gameweek.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fpl_predictor.backtest import BacktestReport, run_backtest
from fpl_predictor.config import MAX_SQUAD_COST, POS_MAP
from fpl_predictor.optimizer import build_starting_xi, find_optimal_squad, suggest_transfers, validate_squad
from fpl_predictor.pipeline import PipelineResult, build_demo_data, fetch_live_data, needs_demo_fallback, run_pipeline
from fpl_predictor.stats_correlation import compute_stat_correlations
from fpl_predictor.team_style import compute_team_styles

st.set_page_config(page_title="FPL Predictor", page_icon="⚽", layout="wide")

# Fixed categorical colour order (blue / orange / aqua / yellow), reused
# everywhere a position is colour-coded so identity mapping never shifts.
POSITION_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
SERIES_BLUE, SERIES_ORANGE = "#2a78d6", "#eb6834"
# Diverging pair (blue <-> red) for the attacking/defensive team-style axis,
# neutral gray for "Balanced" -- same diverging convention as the rest of
# the app's charts, just applied to a style spectrum instead of a delta.
STYLE_COLORS = {"Attacking": "#e34948", "Balanced": "#8a8a8a", "Defensive": "#2a78d6"}


# ---------------------------------------------------------------------------
# Cached data / model layer -- Streamlit reruns the whole script on every
# widget interaction, so the expensive parts (network fetch, LightGBM
# training) must be cached or a gameweek-slider drag would retrain the model.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_pipeline(mode: str, budget_unused: float, _cache_bust: int) -> tuple[PipelineResult, dict]:
    if mode == "demo":
        data = build_demo_data()
    else:
        data = fetch_live_data()
        if needs_demo_fallback(data):
            demo = build_demo_data()
            demo.meta["fallback_reason"] = data.meta.get("error", "live FPL API unavailable")
            return run_pipeline(demo), demo.meta
    return run_pipeline(data), data.meta


def _player_label(row: pd.Series, points_col: str) -> str:
    pts = row.get(points_col, 0.0)
    return f"{row['web_name']} ({row['team_name']}) £{row['now_cost']:.1f}m • {pts:.1f}p"


def _position_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, height: int = 340) -> go.Figure:
    colors = [POSITION_COLORS.get(e, "#8a8a8a") for e in df["element_type"]]
    fig = go.Figure(go.Bar(x=df[x_col], y=df[y_col], orientation="h", marker_color=colors, hovertext=df["team_name"]))
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed"))
    return fig


def _importance_chart(imp_df: pd.DataFrame, position_label: str) -> go.Figure:
    top = imp_df.head(12).sort_values("importance")
    fig = go.Figure(go.Bar(x=top["importance"], y=top["feature"], orientation="h", marker_color=SERIES_BLUE))
    fig.update_layout(title=f"Feature importance — {position_label}", height=380, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def _comparison_chart(your_pts: float, optimal_pts: float, gw: int) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=["Your XI", "Optimal XI"], y=[your_pts, optimal_pts],
        marker_color=[SERIES_ORANGE, SERIES_BLUE], text=[f"{your_pts:.1f}", f"{optimal_pts:.1f}"], textposition="outside",
    ))
    fig.update_layout(title=f"Predicted points — GW{gw}", height=340, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return fig


@st.cache_data(show_spinner=False)
def _cached_backtest(_data, min_train_gws: int, tune: bool, cache_key: str) -> BacktestReport:
    # _data (leading underscore) is deliberately excluded from Streamlit's
    # hashing -- cache_key (built from data_meta) stands in for it instead.
    return run_backtest(_data, min_train_gws=min_train_gws, tune=tune)


def _backtest_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["gw"], y=df["model_xi_points"], name="Model XI (actual pts)", marker_color=SERIES_BLUE))
    fig.add_trace(go.Bar(x=df["gw"], y=df["baseline_xi_points"], name="Naive baseline XI (actual pts)", marker_color=SERIES_ORANGE))
    fig.update_layout(barmode="group", title="Actual points scored per gameweek: model-picked XI vs. naive baseline",
                       xaxis_title="Gameweek", yaxis_title="Points", height=380, margin=dict(l=10, r=10, t=40, b=10),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


def _backtest_mae_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=df["gw"], y=df["mae"], marker_color=SERIES_BLUE))
    fig.update_layout(title="Prediction error per gameweek (lower is better)", xaxis_title="Gameweek",
                       yaxis_title="MAE (points/player)", height=320, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return fig


def _team_style_chart(styles_df: pd.DataFrame) -> go.Figure:
    df = styles_df.sort_values("style_score")
    colors = [STYLE_COLORS.get(s, "#8a8a8a") for s in df["style"]]
    fig = go.Figure(go.Bar(x=df["style_score"], y=df["team_name"], orientation="h", marker_color=colors))
    fig.add_vline(x=0, line_width=1, line_color="rgba(128,128,128,0.5)")
    fig.update_layout(title="Team playing style (attack lean vs. defence lean)", xaxis_title="← more defensive · more attacking →",
                       height=max(320, 28 * len(df)), margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return fig


def _stat_correlation_chart(corr_df: pd.DataFrame) -> go.Figure:
    df = corr_df.sort_values("pearson_r")
    fig = go.Figure(go.Bar(x=df["pearson_r"], y=df["stat"], orientation="h", marker_color=SERIES_BLUE))
    fig.update_layout(title="Correlation with actual points scored (same gameweek)", xaxis_title="Pearson r",
                       height=max(280, 32 * len(df)), margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return fig


def _lineup_table(df: pd.DataFrame, gw_col: str, captain_id=None, vice_id=None) -> pd.DataFrame:
    cols = ["web_name", "team_name", "element_type", "now_cost", gw_col]
    has_start_prob = "start_probability" in df.columns
    if has_start_prob:
        cols.append("start_probability")
    out = df[cols].copy()
    out["POS"] = out["element_type"].map(POS_MAP)
    out = out.drop(columns=["element_type"])
    rename = {"web_name": "Player", "team_name": "Team", "now_cost": "£m", gw_col: "Pred. Pts"}
    if has_start_prob:
        out["start_probability"] = (out["start_probability"] * 100).round(0).astype(int).astype(str) + "%"
        rename["start_probability"] = "Start %"
    out = out.rename(columns=rename)
    if captain_id is not None:
        out.insert(0, "", ["\U0001f451" if pid == captain_id else ("\U0001f948" if pid == vice_id else "") for pid in df["player_id"]])
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚽ FPL Predictor")
data_mode_choice = st.sidebar.radio(
    "Data source", ["Live FPL API (auto-fallback to demo)", "Demo data (offline)"],
    help="Demo data is a small synthetic league used when the live FPL API can't be reached, e.g. from a network-restricted environment.",
)
mode = "demo" if data_mode_choice.startswith("Demo") else "live"

if "cache_bust" not in st.session_state:
    st.session_state.cache_bust = 0
if st.sidebar.button("\U0001f504 Refresh data & retrain model"):
    st.session_state.cache_bust += 1
    st.cache_resource.clear()

budget = MAX_SQUAD_COST  # the real FPL budget -- fixed, not user-adjustable
st.sidebar.metric("Budget", f"£{budget:.0f}m")
free_transfers = st.sidebar.number_input("Free transfers", min_value=0, max_value=5, value=1, step=1)

with st.spinner("Fetching data and training the model (walk-forward CV + hyperparameter search)..."):
    result, data_meta = _load_pipeline(mode, budget, st.session_state.cache_bust)

if data_meta.get("demo") or data_meta.get("fallback_reason"):
    reason = data_meta.get("fallback_reason")
    msg = "Running on **synthetic demo data**" + (f" (live API unavailable: {reason})" if reason else " (offline demo mode selected).")
    st.sidebar.warning(msg)
else:
    st.sidebar.success(f"Live data loaded ({data_meta.get('n_players', 0)} players, source: {data_meta.get('bootstrap_source')})")

if result.mode == "season_over":
    st.error("No upcoming gameweek found -- the season may be over, or fixture data is unavailable.")
    st.stop()
if result.mode == "no_data":
    st.error("No player data available at all. Try refreshing, or switch to demo data.")
    st.stop()

# The app always predicts exactly one gameweek: the next one that hasn't
# been played yet (per the FPL API's own is_next flag where available).
gw = result.future_gws[0]
gw_col = result.gw_cols[0]

st.sidebar.markdown("---")
st.sidebar.metric("Predicting", f"Gameweek {gw}")
if result.mode == "trained":
    st.sidebar.metric("Model MAE (points/player, avg)", f"{result.trained_model.overall_mae:.2f}")
    st.sidebar.caption(f"Trained on {result.last_completed_gw} completed gameweek(s) this season.")
elif result.mode == "cold_start":
    st.sidebar.info(
        "**Early-season baseline mode.** No gameweeks have been played yet this season, so there's nothing to "
        "train a model on -- predictions use each player's last-season per-90 output, scaled by fixture difficulty "
        "and playing probability, instead.",
        icon="ℹ️",
    )

pred_df = result.pred_df
if pred_df.empty:
    st.error(f"No predictions available for GW{gw} (e.g. a blank gameweek with no fixtures). Try refreshing.")
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_optimal, tab_mine, tab_transfers, tab_explorer, tab_model, tab_backtest = st.tabs(
    ["\U0001f3c6 Optimal Squad", "\U0001f464 My Squad", "\U0001f504 Transfers", "\U0001f50d Player Explorer",
     "\U0001f4ca Model Insights", "\U0001f9ea Backtest"]
)

# --- Optimal squad ----------------------------------------------------------
with tab_optimal:
    squad = find_optimal_squad(pred_df, budget=budget, points_col="pred_points_total_adj")
    if squad.empty:
        st.error("No feasible squad found under the current budget/constraints.")
    else:
        lineup, bench, captain, vice = build_starting_xi(squad, gw_col)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Squad cost", f"£{squad['now_cost'].sum():.1f}m", help=f"Budget: £{budget:.0f}m")
        c2.metric(f"Predicted GW{gw} points (XI)", f"{lineup[gw_col].sum():.1f}")
        c3.metric(f"Predicted GW{gw} points (full squad)", f"{squad[gw_col].sum():.1f}")
        c4.metric("Captain", f"{captain['web_name']} ({captain[gw_col]*2:.1f}p)")

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Starting XI")
            st.dataframe(_lineup_table(lineup, gw_col, captain["player_id"], vice["player_id"]), hide_index=True, use_container_width=True)
            st.subheader("Bench")
            st.dataframe(_lineup_table(bench, gw_col), hide_index=True, use_container_width=True)
        with right:
            st.plotly_chart(_position_bar(squad.sort_values("pred_points_total", ascending=False), "pred_points_total", "web_name", "Squad by total predicted points"), use_container_width=True, theme="streamlit")

# --- My squad ---------------------------------------------------------------
with tab_mine:
    st.caption("Pick your current 15-man squad to check it's valid and see your optimal starting XI + captain.")
    if "my_squad" not in st.session_state:
        st.session_state.my_squad = {pos: [] for pos in POS_MAP}

    cols = st.columns(4)
    for col, (pos_id, pos_label) in zip(cols, POS_MAP.items()):
        pos_pool = pred_df[pred_df["element_type"] == pos_id].sort_values("web_name")
        options = pos_pool["player_id"].tolist()
        label_map = {pid: _player_label(pos_pool[pos_pool["player_id"] == pid].iloc[0], "pred_points_total_adj") for pid in options}
        from fpl_predictor.config import POS_COUNTS
        with col:
            st.markdown(f"**{pos_label} ({POS_COUNTS[pos_id]})**")
            selected = st.multiselect(
                pos_label, options=options, default=[p for p in st.session_state.my_squad[pos_id] if p in options],
                format_func=lambda pid: label_map.get(pid, str(pid)), max_selections=POS_COUNTS[pos_id],
                key=f"multiselect_{pos_id}", label_visibility="collapsed",
            )
            st.session_state.my_squad[pos_id] = selected

    all_ids = [pid for ids in st.session_state.my_squad.values() for pid in ids]
    my_squad_df = pred_df[pred_df["player_id"].isin(all_ids)].drop_duplicates(subset=["player_id"])

    if len(all_ids) < 15:
        st.info(f"Select {15 - len(all_ids)} more player(s) to complete your squad ({len(all_ids)}/15).")
    else:
        v = validate_squad(my_squad_df, budget=budget)
        if not v["valid"]:
            st.error("Squad rule violations:\n\n" + "\n".join(f"- {i}" for i in v["issues"]))
        else:
            st.success(f"Valid squad — £{v['total_cost']:.1f}m / £{budget:.0f}m")
            my_lineup, my_bench, my_captain, my_vice = build_starting_xi(my_squad_df, gw_col)
            optimal_squad = find_optimal_squad(pred_df, budget=budget, points_col="pred_points_total_adj")
            optimal_lineup, *_ = build_starting_xi(optimal_squad, gw_col) if not optimal_squad.empty else (pd.DataFrame(),)

            left, right = st.columns([3, 2])
            with left:
                st.subheader("Your Starting XI")
                st.dataframe(_lineup_table(my_lineup, gw_col, my_captain["player_id"], my_vice["player_id"]), hide_index=True, use_container_width=True)
                st.caption(f"\U0001f451 Captain: {my_captain['web_name']} ×2 = {my_captain[gw_col]*2:.1f} pts")
            with right:
                if not optimal_lineup.empty:
                    st.plotly_chart(_comparison_chart(my_lineup[gw_col].sum(), optimal_lineup[gw_col].sum(), gw), use_container_width=True, theme="streamlit")

# --- Transfers ---------------------------------------------------------------
with tab_transfers:
    st.caption("Uses your squad from the 'My Squad' tab as the starting point.")
    all_ids = [pid for ids in st.session_state.get("my_squad", {}).values() for pid in ids]
    my_squad_df = pred_df[pred_df["player_id"].isin(all_ids)].drop_duplicates(subset=["player_id"])
    if len(my_squad_df) != 15:
        st.info("Complete a valid 15-man squad in the 'My Squad' tab first.")
    else:
        max_hits = st.slider("Max transfers to consider", 0, 5, 3)
        with st.spinner("Searching transfer combinations..."):
            plan = suggest_transfers(my_squad_df, pred_df, free_transfers=free_transfers, max_transfers_considered=max_hits, budget=budget, points_col="pred_points_total_adj")
        if plan is None:
            st.error("No feasible transfer plan found.")
        elif plan.n_transfers == 0:
            st.success(f"No transfers recommended — your squad is already optimal (or close to it) for GW{gw}.")
        else:
            hit_note = f" (−{plan.hit_cost:.0f} pt hit)" if plan.hit_cost else ""
            st.success(f"Recommended: {plan.n_transfers} transfer(s){hit_note} — net gain {plan.net_points - my_squad_df['pred_points_total_adj'].sum():.1f} pts for GW{gw}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Out**")
                for name in plan.transfers_out:
                    st.markdown(f"\U0001f53b {name}")
            with c2:
                st.markdown("**In**")
                for name in plan.transfers_in:
                    st.markdown(f"\U0001f53a {name}")

# --- Player explorer ----------------------------------------------------------
with tab_explorer:
    fc1, fc2, fc3 = st.columns(3)
    pos_filter = fc1.multiselect("Position", options=list(POS_MAP.values()), default=list(POS_MAP.values()))
    team_filter = fc2.multiselect("Team", options=sorted(pred_df["team_name"].dropna().unique().tolist()))
    search = fc3.text_input("Search player name")

    view = pred_df.copy()
    view["POS"] = view["element_type"].map(POS_MAP)
    view = view[view["POS"].isin(pos_filter)]
    if team_filter:
        view = view[view["team_name"].isin(team_filter)]
    if search:
        view = view[view["web_name"].str.contains(search, case=False, na=False)]
    view["PPM"] = (view["pred_points_total_adj"] / view["now_cost"]).round(2)

    if "start_probability" in view.columns:
        view["Start %"] = (view["start_probability"] * 100).round(0).astype(int)

    show_cols = ["web_name", "team_name", "POS", "now_cost"] + result.gw_cols + ["pred_points_total", "pred_points_total_adj", "PPM"]
    if "Start %" in view.columns:
        show_cols.append("Start %")
    show_cols = [c for c in show_cols if c in view.columns]
    st.dataframe(
        view[show_cols].rename(columns={"web_name": "Player", "team_name": "Team", "now_cost": "£m", "pred_points_total": "Total Pred.", "pred_points_total_adj": "Adj. Total"})
        .sort_values("Adj. Total", ascending=False),
        hide_index=True, use_container_width=True, height=560,
    )

# --- Model insights ----------------------------------------------------------
with tab_model:
    if result.mode == "cold_start":
        st.info(
            "No LightGBM model is trained right now — there are zero completed gameweeks this season to learn "
            "from yet. Predictions for the upcoming gameweek use last-season per-90 output scaled by fixture "
            "difficulty instead; the full model kicks back in automatically once GW1 results are in.",
            icon="ℹ️",
        )
    else:
        st.caption("Per-position LightGBM models, tuned via walk-forward (time-series) cross-validation.")
        mcols = st.columns(5)
        for col, (pos_id, pos_label) in zip(mcols, POS_MAP.items()):
            pm = result.trained_model.positions.get(pos_id)
            with col:
                if pm is None:
                    st.metric(pos_label, "n/a")
                else:
                    mae_display = f"{pm.mae:.2f}" if pm.mae == pm.mae else "n/a"  # NaN check
                    st.metric(f"{pos_label} MAE", mae_display, help=f"{pm.n_train_rows} training rows")
        with mcols[4]:
            mm = result.minutes_model
            if mm is not None and mm.model is not None:
                st.metric("Minutes model Brier", f"{mm.brier_score:.3f}", help="P(60+ mins next GW) classifier — lower Brier score is better (0 = perfect, 0.25 = coin-flip-level)")
            else:
                st.metric("Minutes model", "n/a", help="Not enough data yet to train a rotation-risk model this season")

        imp_cols = st.columns(2)
        shown = 0
        for pos_id, pos_label in POS_MAP.items():
            pm = result.trained_model.positions.get(pos_id)
            if pm is None or pm.feature_importance.empty:
                continue
            with imp_cols[shown % 2]:
                st.plotly_chart(_importance_chart(pm.feature_importance, pos_label), use_container_width=True, theme="streamlit")
            shown += 1

    with st.expander(f"Fixture difficulty ratings (FPL's own FDR) for GW{gw}"):
        fx = result.data.fixtures_df
        if fx.empty or "event" not in fx.columns:
            st.caption("No fixture data available.")
        else:
            gw_fx = fx[fx["event"] == gw]
            team_names = result.data.teams_df.set_index("id")["name"].to_dict() if not result.data.teams_df.empty else {}
            rows = []
            for r in gw_fx.itertuples(index=False):
                rows.append({"Home": team_names.get(r.team_h, r.team_h), "Away": team_names.get(r.team_a, r.team_a),
                             "Home FDR": getattr(r, "team_h_difficulty", None), "Away FDR": getattr(r, "team_a_difficulty", None)})
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            else:
                st.caption(f"No fixtures found for GW{gw} (blank gameweek).")

    with st.expander("Team playing styles (Attacking / Balanced / Defensive)"):
        st.caption(
            "From real match results (goals scored/conceded per game vs. the rest of the league) -- a *style* axis, "
            "not a quality one: a team excellent at both scoring and defending lands as Balanced (no lopsided "
            "trade-off), same as a team mediocre at both."
        )
        styles = compute_team_styles(result.data.fixtures_df, result.data.teams_df)
        if styles.empty:
            st.caption("No finished matches yet this season to classify styles from.")
        else:
            st.plotly_chart(_team_style_chart(styles), use_container_width=True, theme="streamlit")
            show = styles[["team_name", "style", "goals_scored_per_game", "goals_conceded_per_game", "games_played"]]
            st.dataframe(show.rename(columns={"team_name": "Team", "style": "Style", "goals_scored_per_game": "Scored/gm", "goals_conceded_per_game": "Conceded/gm", "games_played": "Games"}), hide_index=True, use_container_width=True)

    with st.expander("Underlying stats vs. actual points (Opta-derived)"):
        st.caption(
            "How well FPL's underlying match-performance stats -- themselves derived from Opta match event data -- "
            "actually correlate with points scored that same gameweek. A raw Opta feed integration would need a "
            "separate commercial license this project doesn't have; these aggregate stats are already Opta-derived "
            "and already available via the FPL API."
        )
        corr = compute_stat_correlations(result.data.history_df)
        if corr.empty:
            st.caption("Not enough completed-gameweek history yet to compute correlations.")
        else:
            st.plotly_chart(_stat_correlation_chart(corr), use_container_width=True, theme="streamlit")
            st.dataframe(corr.rename(columns={"stat": "Stat", "pearson_r": "Pearson r", "spearman_r": "Spearman r", "n": "N"}), hide_index=True, use_container_width=True)

# --- Backtest ------------------------------------------------------------
with tab_backtest:
    st.caption(
        "Walk-forward validation: for each past gameweek, retrains using only earlier gameweeks (no lookahead), "
        "predicts it, and compares the actual points scored by the model-picked XI against a naive baseline "
        "(a squad picked by chasing the *previous* gameweek's top scorers). Both squads are freely rebuilt each "
        "gameweek under budget -- this measures prediction/selection quality, not real transfer-limited strategy."
    )
    n_completed_gws = int(result.data.history_df["round"].nunique()) if not result.data.history_df.empty else 0
    if n_completed_gws < 4:
        st.info(
            f"Only {n_completed_gws} completed gameweek(s) of history so far this season -- need at least 4 "
            "to run a walk-forward backtest. Check back once more gameweeks have been played.",
            icon="ℹ️",
        )
    else:
        bt_col1, bt_col2, bt_col3 = st.columns(3)
        min_train_gws = bt_col1.number_input("Warm-up gameweeks (min. training data)", min_value=1, max_value=max(1, n_completed_gws - 1), value=min(3, n_completed_gws - 1))
        tune = bt_col2.checkbox("Full hyperparameter tuning (slower, more representative)", value=False)
        run_bt = bt_col3.button("\U0001f9ea Run backtest", type="primary")

        cache_key = f"{data_meta.get('bootstrap_source')}|{data_meta.get('n_history_rows')}|{n_completed_gws}|{min_train_gws}|{tune}"
        if run_bt:
            st.session_state["backtest_cache_key"] = cache_key
        if st.session_state.get("backtest_cache_key") == cache_key:
            with st.spinner("Replaying past gameweeks (retraining at each step)..."):
                bt_report = _cached_backtest(result.data, int(min_train_gws), bool(tune), cache_key)
            if not bt_report.per_gw:
                st.warning("Backtest produced no comparable gameweeks (not enough history after the warm-up period).")
            else:
                bt_df = bt_report.as_dataframe()
                advantage = bt_report.total_model_xi_points - bt_report.total_baseline_xi_points
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Overall MAE", f"{bt_report.overall_mae:.2f}")
                m2.metric("Mean rank correlation", f"{bt_report.mean_spearman:.2f}", help="Spearman correlation between predicted and actual points each gameweek")
                m3.metric("Model XI total (actual pts)", f"{bt_report.total_model_xi_points:.0f}")
                m4.metric("vs. naive baseline", f"{advantage:+.0f}", help="Model-picked XI total minus naive-baseline XI total, summed over all backtested gameweeks")
                st.plotly_chart(_backtest_chart(bt_df), use_container_width=True, theme="streamlit")
                st.plotly_chart(_backtest_mae_chart(bt_df), use_container_width=True, theme="streamlit")
                with st.expander("Per-gameweek detail"):
                    st.dataframe(bt_df.rename(columns={
                        "gw": "GW", "n_players": "Players", "mae": "MAE", "spearman_corr": "Rank corr.",
                        "model_xi_points": "Model XI pts", "baseline_xi_points": "Baseline XI pts",
                    }), hide_index=True, use_container_width=True)
        else:
            st.caption("Click **Run backtest** to compute this (retrains the model once per historical gameweek, so it takes a little while).")
