"""
Sleeper Fantasy FPL Predictor — Streamlit Web App
"""

import json
import time
import unicodedata
from pathlib import Path
import streamlit as st
import pandas as pd
import requests

from sleeper_predictor import (
    FPLDataClient,
    load_current_season_data,
    engineer_features,
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
    "avg_pts_5", "exp_goals", "exp_assists", "exp_sot", "exp_kp", "exp_tkl", "exp_int",
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
    "avg_pts_5":    st.column_config.NumberColumn("Avg Pts 5", format="%.1f"),
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

@st.cache_data(ttl=3600, show_spinner=False)
def get_sleeper_name_map() -> tuple[dict[str, str], list[str]]:
    """Fetch Sleeper's player database. Returns ({id: name}, [debug_lines])."""
    debug = []
    # First, get the league info to find the correct sport key
    try:
        lr = requests.get(f"{SLEEPER_API}/league/{SLEEPER_LEAGUE_ID}", timeout=10)
        sport_key = lr.json().get("sport", "") if lr.status_code == 200 else ""
        debug.append(f"League sport key: '{sport_key}' (status {lr.status_code})")
    except Exception as e:
        sport_key = ""
        debug.append(f"League fetch error: {e}")

    sport_keys = []
    if sport_key:
        sport_keys.append(sport_key)
    sport_keys += [k for k in ("epl", "soccer", "pl", "football", "nfl") if k != sport_key]

    for sport in sport_keys:
        try:
            r = requests.get(f"{SLEEPER_API}/players/{sport}", timeout=30)
            debug.append(f"  /players/{sport} → {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                result = {}
                for pid, p in data.items():
                    full = (p.get("full_name")
                            or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip())
                    if full.strip():
                        result[str(pid)] = full.strip()
                if result:
                    debug.append(f"  ✅ Got {len(result)} players from /players/{sport}")
                    return result, debug
        except Exception as e:
            debug.append(f"  /players/{sport} → error: {e}")

    return {}, debug


def _normalise(name: str) -> str:
    """Lowercase and strip accents."""
    nfkd = unicodedata.normalize("NFKD", name.lower().strip())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


@st.cache_data(ttl=300, show_spinner=False)
def get_league_data(league_id: str) -> tuple[set[str], str]:
    """Returns (set_of_drafted_lowercase_names, status_message)."""
    try:
        r = requests.get(f"{SLEEPER_API}/league/{league_id}/rosters", timeout=15)
        r.raise_for_status()
        rosters = r.json()

        if not rosters:
            return set(), "No rosters found in league"

        player_ids: set[str] = set()
        for roster in rosters:
            for field in ("players", "reserve", "taxi"):
                for pid in (roster.get(field) or []):
                    player_ids.add(str(pid))

        if not player_ids:
            return set(), "Rosters exist but no players found yet"

        # Map Sleeper IDs → names
        name_map, name_debug = get_sleeper_name_map()
        drafted_names: set[str] = set()
        for pid in player_ids:
            name = name_map.get(pid, "")
            if name:
                drafted_names.add(_normalise(name))

        if drafted_names:
            return drafted_names, f"✅ League synced — {len(player_ids)} players on rosters"

        return player_ids, (
            f"⚠️ League synced ({len(player_ids)} players) — name lookup failed\n"
            + "\n".join(name_debug)
        )

    except requests.HTTPError as e:
        return set(), f"Sleeper API error: {e.response.status_code}"
    except Exception as e:
        return set(), f"Could not reach Sleeper: {e}"


@st.cache_data(ttl=300, show_spinner=False)
def get_my_roster(league_id: str, username: str) -> tuple[set[str], str]:
    """Fetch just this user's roster. Returns (set_of_player_names, status)."""
    try:
        ur = requests.get(f"{SLEEPER_API}/user/{username}", timeout=10)
        if ur.status_code != 200:
            return set(), f"Sleeper user '{username}' not found"
        user_id = str(ur.json().get("user_id", ""))
        if not user_id:
            return set(), "Could not retrieve user_id from Sleeper"

        rr = requests.get(f"{SLEEPER_API}/league/{league_id}/rosters", timeout=15)
        rr.raise_for_status()
        rosters = rr.json()

        my_roster = next(
            (ros for ros in rosters if str(ros.get("owner_id", "")) == user_id), None
        )
        if not my_roster:
            return set(), f"No roster found for '{username}' in this league"

        player_ids: set[str] = set()
        for field in ("players", "reserve", "taxi"):
            for pid in (my_roster.get(field) or []):
                player_ids.add(str(pid))

        name_map, _ = get_sleeper_name_map()
        my_names: set[str] = set()
        for pid in player_ids:
            name = name_map.get(pid, "")
            if name:
                my_names.add(_normalise(name))

        return my_names, f"✅ {len(player_ids)} players on your roster"

    except requests.HTTPError as e:
        return set(), f"Sleeper API error: {e.response.status_code}"
    except Exception as e:
        return set(), f"Error fetching roster: {e}"


def _is_drafted(fpl_name: str, display_name: str, drafted_names: set[str]) -> bool:
    """Return True if this player appears in the drafted set."""
    n = _normalise(fpl_name)
    d = _normalise(display_name)

    # 1. Exact full-name or display-name match
    if n in drafted_names or d in drafted_names:
        return True

    n_words = set(n.split())
    for dn in drafted_names:
        dn_words = dn.split()
        dn_set   = set(dn_words)

        # 2. Two or more words in common — catches "Bruno Borges Fernandes" vs "Bruno Fernandes"
        if len(n_words & dn_set) >= 2:
            return True

        # 3. Last word of FPL full name (≥5 chars) matches last word of a Sleeper name,
        #    AND first-name initials agree — avoids common-surname false positives
        #    (e.g. "callum wilson" must not match "harry wilson")
        n_parts = n.split()
        if n_parts and len(n_parts[-1]) >= 5 and n_parts[-1] == dn_words[-1]:
            if len(n_parts) < 2 or len(dn_words) < 2:
                return True  # single-name player, trust the match
            if n_parts[0][0] == dn_words[0][0]:
                return True

    return False


# ── FPL predictions ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def get_predictions() -> pd.DataFrame:
    fpl_client = FPLDataClient()
    boot       = fpl_client.bootstrap()
    fixtures   = fpl_client.fixtures()
    df, ts     = load_current_season_data(fpl_client, boot, fixtures)
    feat       = engineer_features(df)
    return predict_next_gw(feat, ts, fixtures, boot)


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
    view = view[~view.apply(
        lambda r: _is_drafted(r["name"], r["display_name"], drafted_ids), axis=1
    )]

st.caption(f"{len(view)} players shown")

with st.expander("🔍 Debug", expanded=False):
    _cache_dir = Path(".fpl_cache")

    # Sleeper historical stats API
    st.write("**Sleeper stats API (historical points source):**")
    _slp_cache = next(iter(sorted(_cache_dir.glob("sleeper_hist_gw*.json"))), None)
    _slp_per90 = next(iter(sorted(_cache_dir.glob("sleeper_per90_v8_*.json"))), None)
    _slp_keys  = _cache_dir / "sleeper_stat_keys.json"
    if _slp_cache:
        _slp_data = json.loads(_slp_cache.read_text())
        st.write(f"  ✅ {len(_slp_data)} players with Sleeper hist pts ({_slp_cache.stem})")
        if _slp_per90:
            _per90_data = json.loads(_slp_per90.read_text())
            # Show a GK's per-90 data to verify save field
            gk_sample = {k: v for k, v in list(_per90_data.items())[:3]}
            st.write(f"  Per-90 cache: {len(_per90_data)} players")
            st.write("  Sample per-90 (first 3 players):")
            st.json(gk_sample)
        if _slp_keys.exists():
            st.write("  All Sleeper API stat fields observed:")
            st.code(", ".join(json.loads(_slp_keys.read_text())))
    else:
        st.write("  ❌ Not fetched yet — click Refresh Predictions")

    # Sleeper name matching
    st.write("**Sleeper name matching:**")
    sample_drafted = sorted(drafted_ids)[:10]
    sample_fpl     = predictions[["name", "display_name"]].head(10).values.tolist()
    st.write("Drafted names from Sleeper (first 10):", sample_drafted)
    st.write("FPL names in predictions (first 10):", sample_fpl)
    if available_only:
        st.write(f"Players filtered out: {len(predictions) - len(view)}")
    _, name_debug = get_sleeper_name_map()
    st.write("**Sleeper API diagnostic:**")
    st.code("\n".join(name_debug))

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

# ── My Team ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("My Team")

username = st.text_input(
    "Your Sleeper username",
    placeholder="e.g. jjcochesa",
    help="Enter your Sleeper username to see lineup suggestions and waiver targets",
)

if username.strip():
    with st.spinner("Loading your roster..."):
        my_names, my_status = get_my_roster(SLEEPER_LEAGUE_ID, username.strip())
    st.caption(my_status)

    if my_names:
        my_players = predictions[
            predictions.apply(
                lambda r: _is_drafted(r["name"], r["display_name"], my_names), axis=1
            )
        ].sort_values("sleeper_pts", ascending=False).reset_index(drop=True)

        if len(my_players) == 0:
            st.warning("Roster found but no players matched predictions — name matching may need tuning.")
        else:
            col_left, col_right = st.columns([3, 2])

            with col_left:
                st.markdown("**Starting 11** *(sorted by predicted pts)*")
                starters = my_players.head(11)
                st.dataframe(
                    starters[_DISPLAY_COLS],
                    use_container_width=True,
                    hide_index=True,
                    column_config=_COL_CONFIG,
                )
                if len(my_players) > 11:
                    st.markdown("**Bench**")
                    bench = my_players.iloc[11:]
                    st.dataframe(
                        bench[_DISPLAY_COLS],
                        use_container_width=True,
                        hide_index=True,
                        column_config=_COL_CONFIG,
                    )

            with col_right:
                st.markdown("**Top Waiver Targets**")
                if league_ok:
                    available = predictions[
                        ~predictions.apply(
                            lambda r: _is_drafted(r["name"], r["display_name"], drafted_ids),
                            axis=1,
                        )
                    ]
                    wtabs = st.tabs(["🧤 GK", "🛡️ DEF", "🎯 MID", "⚡ FWD"])
                    for wtab, pos in zip(wtabs, ["GK", "DEF", "MID", "FWD"]):
                        with wtab:
                            top_avail = available[available["position"] == pos].head(5)
                            st.dataframe(
                                top_avail[_DISPLAY_COLS],
                                use_container_width=True,
                                hide_index=True,
                                column_config=_COL_CONFIG,
                            )
                else:
                    st.info("League sync required for waiver targets.")

# Refresh
st.markdown("---")
if st.button("🔄 Refresh Predictions"):
    st.cache_data.clear()
    st.rerun()
