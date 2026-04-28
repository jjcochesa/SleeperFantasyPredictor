"""
Sleeper Fantasy FPL Predictor — Streamlit Web App
"""

import streamlit as st
import pandas as pd
import requests

from sleeper_predictor import (
    FPLDataClient,
    load_current_season_data,
    engineer_features,
    train_component_models,
    predict_next_gw,
)

st.set_page_config(
    page_title="Sleeper FPL Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SLEEPER_LEAGUE_ID = "1255585419836260352"
SLEEPER_API       = "https://api.sleeper.app/v1"

_DISPLAY_COLS = [
    "display_name", "team", "opp", "ha", "fdr", "position", "form", "avail",
    "exp_goals", "exp_assists", "exp_sot", "exp_kp", "exp_tkl", "exp_int",
    "exp_saves", "exp_cs", "sleeper_pts",
]

_COL_CONFIG = {
    "display_name": st.column_config.TextColumn("Player"),
    "team":         st.column_config.TextColumn("Team"),
    "opp":          st.column_config.TextColumn("Opp"),
    "ha":           st.column_config.TextColumn("H/A"),
    "fdr":          st.column_config.NumberColumn("FDR"),
    "position":     st.column_config.TextColumn("Pos"),
    "form":         st.column_config.TextColumn("Form"),
    "avail":        st.column_config.TextColumn("Avail"),
    "exp_goals":    st.column_config.NumberColumn("xG",    format="%.2f"),
    "exp_assists":  st.column_config.NumberColumn("xA",    format="%.2f"),
    "exp_sot":      st.column_config.NumberColumn("SoT",   format="%.2f"),
    "exp_kp":       st.column_config.NumberColumn("KP",    format="%.2f"),
    "exp_tkl":      st.column_config.NumberColumn("Tkl",   format="%.2f"),
    "exp_int":      st.column_config.NumberColumn("Int",   format="%.2f"),
    "exp_saves":    st.column_config.NumberColumn("Saves", format="%.2f"),
    "exp_cs":       st.column_config.NumberColumn("CS%",   format="%.2f"),
    "sleeper_pts":  st.column_config.NumberColumn("Pts",   format="%.1f"),
}


# ── Sleeper roster fetch ──────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def get_drafted_names(league_id: str) -> set[str]:
    """Return a set of lowercase player names already on a roster in the league."""
    try:
        # All rosters in the league
        rosters_r = requests.get(f"{SLEEPER_API}/league/{league_id}/rosters", timeout=10)
        rosters_r.raise_for_status()
        rosters = rosters_r.json()

        # Collect all player IDs across every roster
        player_ids: list[str] = []
        for roster in rosters:
            player_ids.extend(roster.get("players") or [])
            player_ids.extend(roster.get("reserve") or [])
            player_ids.extend(roster.get("taxi") or [])

        if not player_ids:
            return set()

        # Sleeper soccer player database — maps id → {full_name, ...}
        players_r = requests.get(f"{SLEEPER_API}/players/epl", timeout=30)
        if players_r.status_code != 200:
            # Fallback sport key
            players_r = requests.get(f"{SLEEPER_API}/players/soccer", timeout=30)
        players_r.raise_for_status()
        all_players: dict = players_r.json()

        drafted: set[str] = set()
        for pid in player_ids:
            p = all_players.get(str(pid), {})
            full = p.get("full_name") or f"{p.get('first_name','')} {p.get('last_name','')}".strip()
            if full:
                drafted.add(full.lower())

        return drafted

    except Exception:
        return set()


def _is_available(name: str, drafted: set[str]) -> bool:
    if not drafted:
        return True
    n = name.lower()
    # Exact match or partial — Sleeper names may differ slightly from FPL
    if n in drafted:
        return False
    parts = n.split()
    return not any(
        all(p in d for p in parts) or all(p in parts for p in d.split())
        for d in drafted
    )


# ── FPL predictions ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def get_predictions() -> pd.DataFrame:
    fpl_client = FPLDataClient()
    boot       = fpl_client.bootstrap()
    fixtures   = fpl_client.fixtures()
    df, ts     = load_current_season_data(fpl_client, boot, fixtures)
    feat       = engineer_features(df)
    bundle, feature_cols = train_component_models(feat)
    return predict_next_gw(feat, bundle, feature_cols, ts, fixtures, boot)


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("⚽ Sleeper Fantasy Predictor")

with st.spinner("Loading predictions — first run takes ~3 min..."):
    try:
        predictions = get_predictions()
    except Exception as e:
        st.error(f"Failed to load predictions: {e}")
        st.stop()

gw = int(predictions["GW"].iloc[0])
st.markdown(f"### Gameweek {gw} Predictions")

# Filters row
c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
with c1:
    pos_filter = st.selectbox("Position", ["All", "GK", "DEF", "MID", "FWD"])
with c2:
    teams = ["All"] + sorted(predictions["team"].unique().tolist())
    team_filter = st.selectbox("Team", teams)
with c3:
    min_pts = st.number_input("Min Pts", min_value=0.0, value=0.0, step=1.0)
with c4:
    search = st.text_input("Search player", placeholder="e.g. Salah, Haaland")

# Draft availability toggle
with st.spinner("Checking your league roster..."):
    drafted = get_drafted_names(SLEEPER_LEAGUE_ID)

league_ok = bool(drafted)  # False means API didn't return data
available_only = st.toggle(
    "✅ Show available (undrafted) players only",
    value=False,
    help="Hides players already on a roster in your Sleeper league" if league_ok
         else "League roster data unavailable — toggle has no effect",
    disabled=not league_ok,
)

# Apply filters
view = predictions.copy()
if pos_filter != "All":
    view = view[view["position"] == pos_filter]
if team_filter != "All":
    view = view[view["team"] == team_filter]
if min_pts > 0:
    view = view[view["sleeper_pts"] >= min_pts]
if search:
    mask = (
        view["name"].str.lower().str.contains(search.lower(), na=False)
        | view["display_name"].str.lower().str.contains(search.lower(), na=False)
    )
    view = view[mask]
if available_only and league_ok:
    view = view[view["name"].apply(lambda n: _is_available(n, drafted))]

st.caption(f"{len(view)} players" + (" · league data live" if league_ok else " · league data unavailable"))

st.dataframe(
    view[_DISPLAY_COLS].head(100),
    use_container_width=True,
    hide_index=True,
    column_config=_COL_CONFIG,
)

# Position tabs
st.markdown("---")
st.subheader("Top 10 by Position")

tab_src = view if (available_only and league_ok) else predictions
tabs = st.tabs(["🧤 GK", "🛡️ DEF", "🎯 MID", "⚡ FWD"])
for tab, pos in zip(tabs, ["GK", "DEF", "MID", "FWD"]):
    with tab:
        sub = tab_src[tab_src["position"] == pos].head(10)
        st.dataframe(sub[_DISPLAY_COLS], use_container_width=True, hide_index=True,
                     column_config=_COL_CONFIG)

# Refresh
st.markdown("---")
if st.button("🔄 Refresh Predictions"):
    st.cache_data.clear()
    st.rerun()
