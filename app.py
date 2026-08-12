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

from fpl_predictor.config import MAX_SQUAD_COST, POS_MAP
from fpl_predictor.optimizer import build_starting_xi, find_optimal_squad, suggest_transfers, validate_squad
from fpl_predictor.pipeline import PipelineResult, build_demo_data, fetch_live_data, run_pipeline

st.set_page_config(page_title="FPL Predictor", page_icon="⚽", layout="wide")

# Fixed categorical colour order (blue / orange / aqua / yellow), reused
# everywhere a position is colour-coded so identity mapping never shifts.
POSITION_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
SERIES_BLUE, SERIES_ORANGE = "#2a78d6", "#eb6834"


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
        if not data.meta.get("ok") or data.history_df.empty:
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


def _lineup_table(df: pd.DataFrame, gw_col: str, captain_id=None, vice_id=None) -> pd.DataFrame:
    out = df[["web_name", "team_name", "element_type", "now_cost", gw_col]].copy()
    out["POS"] = out["element_type"].map(POS_MAP)
    out = out.drop(columns=["element_type"]).rename(columns={"web_name": "Player", "team_name": "Team", "now_cost": "£m", gw_col: "Pred. Pts"})
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

budget = st.sidebar.number_input("Budget (£m)", min_value=50.0, max_value=150.0, value=MAX_SQUAD_COST, step=0.5)
free_transfers = st.sidebar.number_input("Free transfers", min_value=0, max_value=5, value=1, step=1)

with st.spinner("Fetching data and training the model (walk-forward CV + hyperparameter search)..."):
    result, data_meta = _load_pipeline(mode, budget, st.session_state.cache_bust)

if data_meta.get("demo") or data_meta.get("fallback_reason"):
    reason = data_meta.get("fallback_reason")
    msg = "Running on **synthetic demo data**" + (f" (live API unavailable: {reason})" if reason else " (offline demo mode selected).")
    st.sidebar.warning(msg)
else:
    st.sidebar.success(f"Live data loaded ({data_meta.get('n_players', 0)} players, source: {data_meta.get('bootstrap_source')})")

if result.feat_df.empty:
    st.error("No data available to build features or train a model. Try refreshing, or switch to demo data.")
    st.stop()

gw_options = result.future_gws
gw = st.sidebar.select_slider("Gameweek to optimise for", options=gw_options, value=gw_options[0]) if gw_options else None
gw_col = f"GW{gw}_Points" if gw else None

st.sidebar.markdown("---")
st.sidebar.metric("Model MAE (points/player, avg)", f"{result.trained_model.overall_mae:.2f}")
st.sidebar.caption(f"Trained through GW{result.last_completed_gw} • projecting GW{gw_options[0] if gw_options else '-'}-{gw_options[-1] if gw_options else '-'}")

pred_df = result.pred_df
if pred_df.empty or gw_col is None:
    st.error("No upcoming fixtures found to predict -- the season may be over, or fixture data is unavailable.")
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_optimal, tab_mine, tab_transfers, tab_explorer, tab_model = st.tabs(
    ["\U0001f3c6 Optimal Squad", "\U0001f464 My Squad", "\U0001f504 Transfers", "\U0001f50d Player Explorer", "\U0001f4ca Model Insights"]
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
        c3.metric(f"Total pred. points (GW{gw_options[0]}-{gw_options[-1]})", f"{squad['pred_points_total'].sum():.1f}")
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
            st.success("No transfers recommended — your squad is already optimal (or close to it) for the projection horizon.")
        else:
            hit_note = f" (−{plan.hit_cost:.0f} pt hit)" if plan.hit_cost else ""
            st.success(f"Recommended: {plan.n_transfers} transfer(s){hit_note} — net gain {plan.net_points - my_squad_df['pred_points_total_adj'].sum():.1f} pts over the projection horizon")
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

    show_cols = ["web_name", "team_name", "POS", "now_cost"] + result.gw_cols + ["pred_points_total", "pred_points_total_adj", "PPM"]
    show_cols = [c for c in show_cols if c in view.columns]
    st.dataframe(
        view[show_cols].rename(columns={"web_name": "Player", "team_name": "Team", "now_cost": "£m", "pred_points_total": "Total Pred.", "pred_points_total_adj": "Adj. Total"})
        .sort_values("Adj. Total", ascending=False),
        hide_index=True, use_container_width=True, height=560,
    )

# --- Model insights ----------------------------------------------------------
with tab_model:
    st.caption("Per-position LightGBM models, tuned via walk-forward (time-series) cross-validation.")
    mcols = st.columns(4)
    for col, (pos_id, pos_label) in zip(mcols, POS_MAP.items()):
        pm = result.trained_model.positions.get(pos_id)
        with col:
            if pm is None:
                st.metric(pos_label, "n/a")
            else:
                mae_display = f"{pm.mae:.2f}" if pm.mae == pm.mae else "n/a"  # NaN check
                st.metric(f"{pos_label} MAE", mae_display, help=f"{pm.n_train_rows} training rows")

    imp_cols = st.columns(2)
    shown = 0
    for pos_id, pos_label in POS_MAP.items():
        pm = result.trained_model.positions.get(pos_id)
        if pm is None or pm.feature_importance.empty:
            continue
        with imp_cols[shown % 2]:
            st.plotly_chart(_importance_chart(pm.feature_importance, pos_label), use_container_width=True, theme="streamlit")
        shown += 1

    with st.expander("Team strength (Elo) ratings, from historical results"):
        from fpl_predictor.data_sources.team_strength import build_team_strength_table
        st.dataframe(build_team_strength_table(result.elo_ratings), hide_index=True, use_container_width=True)
