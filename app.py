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
def get_league_data(league_id: str) -> tuple[set[str], str]:
    """
    Returns (set_of_drafted_player_ids, status_message).
    Uses FPL player IDs directly since Sleeper FPL leagues use them as player keys.
    """
    try:
        r = requests.get(f"{SLEEPER_API}/league/{league_id}/rosters", timeout=15)
        r.raise_for_status()
        rosters = r.json()

        if not rosters:
            return set(), "No rosters found in league"

        # Collect all player IDs (Sleeper FPL leagues use FPL element IDs as strings)
        player_ids: set[str] = set()
        for roster in rosters:
            for field in ("players", "reserve", "taxi"):
                for pid in (roster.get(field) or []):
                    player_ids.add(str(pid))

        if not player_ids:
            return set(), "Rosters exist but no players found yet"

        return player_ids, f"✅ League synced — {len(player_ids)} players on rosters"

    except requests.HTTPError as e:
        return set(), f"Sleeper API error: {e.response.status_code}"
    except Exception as e:
        return set(), f"Could not reach Sleeper: {e}"


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

# Sleeper league sync
drafted_ids, league_status = get_league_data(SLEEPER_LEAGUE_ID)
league_ok = bool(drafted_ids)

st.caption(league_status)

available_only = st.toggle(
    "✅ Show available (undrafted) players only",
    value=False,
    disabled=not league_ok,
    help="Hides players already on a roster in your Sleeper league",
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
    view = view[~view["player_id"].astype(str).isin(drafted_ids)]

st.caption(f"{len(view)} players shown")

with st.expander("🔍 Debug: ID matching", expanded=False):
    sample_sleeper = sorted(drafted_ids)[:10]
    sample_fpl    = predictions["player_id"].astype(str).head(10).tolist()
    st.write("**Sleeper roster IDs (first 10):**", sample_sleeper)
    st.write("**FPL player IDs in predictions (first 10):**", sample_fpl)
    overlap = drafted_ids & set(predictions["player_id"].astype(str))
    st.write(f"**Matching IDs:** {len(overlap)} overlap out of {len(drafted_ids)} drafted")

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
