"""
Sleeper Fantasy Premier League Weekly Predictor
================================================

Data sources
  • FPL API  – goals, assists, clean sheets, saves, cards, goals conceded,
               minutes, xG, xA, ICT index, player availability
  • FBref    – shots on target, key passes, accurate crosses, tackles won,
               interceptions, blocked shots, clearances, dribbles, aerials

These are the same Opta stats Sleeper uses for scoring.

Usage:
    python sleeper_predictor.py

Requirements:
    pip install requests pandas lightgbm numpy scikit-learn soccerdata pyarrow

Author: Sleeper Predictor for Juan
"""

import difflib
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger()

FPL_API = "https://fantasy.premierleague.com/api"
POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# ============================================================================
# SLEEPER SCORING TABLE
# ============================================================================

SLEEPER_SCORING = {
    "goals":                {"FWD": 9,    "MID": 9,    "DEF": 10,   "GK": 10},
    "assists":              {"FWD": 6,    "MID": 6,    "DEF": 7,    "GK": 7},
    "shots_on_target":      2.0,
    "key_passes":           {"FWD": 2,    "MID": 2,    "DEF": 2,    "GK": 0},
    "successful_dribbles":  1.0,
    "accurate_crosses":     1.0,
    "yellow_card":         -2.0,
    "red_card":            -7.0,
    "aerials_won":          {"FWD": 0.5,  "MID": 0.5,  "DEF": 1.0,  "GK": 1.0},
    "effective_clearances": {"FWD": 0,    "MID": 0,    "DEF": 0.25, "GK": 0.25},
    "saves":                2.0,
    "clean_sheet_60plus":   {"FWD": 0,    "MID": 1,    "DEF": 6,    "GK": 8},
    "tackles_won":          1.0,
    "interceptions":        1.0,
    "blocked_shots":        1.0,
    "goals_against":        {"FWD": 0,    "MID": 0,    "DEF": -2,   "GK": -2},
}

# Stats that come exclusively from FBref (not in FPL API)
FBREF_STATS = [
    "shots_on_target",
    "key_passes",
    "accurate_crosses",
    "tackles_won",
    "interceptions",
    "blocked_shots",
    "effective_clearances",
    "successful_dribbles",
    "aerials_won",
]

# FBref stat_type → {our_name: [candidate column substrings to search for]}
FBREF_PULL = {
    "shooting":   {
        "shots_on_target":      ["sot", "shots_on_target"],
    },
    "passing":    {
        "key_passes":           ["kp", "key_pass"],
        "accurate_crosses":     ["crs", "crosses"],
    },
    "defense":    {
        "tackles_won":          ["tklw", "tackles_won"],
        "interceptions":        ["int", "interceptions"],
        "blocked_shots":        ["blocks_sh", "sh"],   # shots blocked, under Blocks group
        "effective_clearances": ["clr", "clearances"],
    },
    "possession": {
        "successful_dribbles":  ["succ", "dribbles_succ", "successful_dribbles"],
    },
    "misc":       {
        "aerials_won":          ["won", "aerials_won", "aerial_won"],
    },
}

# Stats the model is trained to predict
TRAIN_TARGETS = [
    "goals_scored", "assists", "expected_goals", "expected_assists",
    "clean_sheets", "saves", "yellow_cards", "red_cards",
    "goals_conceded",
    # FBref targets
    "shots_on_target", "key_passes", "accurate_crosses",
    "tackles_won", "interceptions", "blocked_shots",
    "effective_clearances", "successful_dribbles", "aerials_won",
]

NON_FEATURE = {"name", "team", "position", "GW", "minutes", "player_id",
               "status", "chance_of_playing"}

FPL_ROLLING_STATS = [
    "goals_scored", "assists", "expected_goals", "expected_assists",
    "clean_sheets", "saves", "total_points", "influence", "creativity",
    "threat", "goals_conceded", "expected_goals_conceded",
]


# ============================================================================
# HELPERS
# ============================================================================

def _current_season_str() -> str:
    """Return soccerdata season string, e.g. '2526' for 2025-26."""
    now = datetime.now()
    y = now.year if now.month >= 8 else now.year - 1
    return f"{str(y)[2:]}{str(y + 1)[2:]}"


def _pos_score(key: str, pos: str) -> float:
    v = SLEEPER_SCORING[key]
    return v.get(pos, 0) if isinstance(v, dict) else float(v)


# ============================================================================
# FPL CLIENT
# ============================================================================

class FPLDataClient:
    """Fetch FPL API data with local JSON caching."""

    def __init__(self, cache_dir: str = ".fpl_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-GB,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://fantasy.premierleague.com/",
            "Origin": "https://fantasy.premierleague.com",
        })

    def _get_json(self, url: str, cache_key: str, force: bool = False) -> dict:
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists() and not force:
            return json.loads(cache_path.read_text())
        log.info(f"🌐 Fetching {cache_key}...")
        r = self.session.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        cache_path.write_text(json.dumps(data))
        return data

    def bootstrap(self) -> dict:
        return self._get_json(f"{FPL_API}/bootstrap-static/", "bootstrap", force=True)

    def fixtures(self) -> list:
        return self._get_json(f"{FPL_API}/fixtures/", "fixtures", force=True)

    def element_summary(self, player_id: int) -> dict:
        return self._get_json(
            f"{FPL_API}/element-summary/{player_id}/", f"element_{player_id}"
        )


# ============================================================================
# FBREF CLIENT  (soccerdata wrapper)
# ============================================================================

class FBrefDataClient:
    """
    Fetches per-player per-match Opta stats from FBref via soccerdata.
    Results are cached as Parquet so subsequent runs are instant.
    """

    def __init__(self, season: str | None = None, cache_dir: str = ".fbref_cache") -> None:
        self.season = season or _current_season_str()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    # -- internal helpers ---------------------------------------------------

    def _flatten(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collapse MultiIndex columns to single lowercase strings."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join(
                    p.strip().lower()
                    for p in col
                    if p.strip() and "unnamed" not in p.lower()
                ) or str(col[-1]).strip().lower()
                for col in df.columns
            ]
        else:
            df.columns = [str(c).strip().lower() for c in df.columns]
        return df

    def _find_col(self, df: pd.DataFrame, candidates: list[str]) -> str | None:
        """Return the first column whose name contains any candidate substring."""
        for cand in candidates:
            hits = [c for c in df.columns if cand in c]
            if hits:
                return hits[0]
        return None

    # -- public API ---------------------------------------------------------

    def load_match_stats(self) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
            player, GW, shots_on_target, key_passes, accurate_crosses,
            tackles_won, interceptions, blocked_shots, effective_clearances,
            successful_dribbles, aerials_won
        One row per player per gameweek.
        """
        cache_path = self.cache_dir / f"fbref_{self.season}.parquet"
        if cache_path.exists():
            log.info("✓ FBref stats loaded from cache")
            return pd.read_parquet(cache_path)

        try:
            from soccerdata import FBref  # type: ignore
        except ImportError:
            log.warning("⚠  soccerdata not installed — run: pip install soccerdata")
            return pd.DataFrame()

        try:
            fbref = FBref(
                leagues="ENG-Premier League",
                seasons=self.season,
                data_dir=self.cache_dir,
            )
        except Exception as e:
            log.warning(f"⚠  FBref init failed: {e}")
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []

        for stat_type, targets in FBREF_PULL.items():
            try:
                log.info(f"🌐 FBref: {stat_type}...")
                raw = fbref.read_player_match_stats(stat_type).reset_index()
                raw = self._flatten(raw)

                player_col = self._find_col(raw, ["player"])
                gw_col     = self._find_col(raw, ["round", "matchweek", "gameweek"])
                if not player_col:
                    log.warning(f"  ↳ no player column in {stat_type}, skipping")
                    continue

                rename = {player_col: "player"}
                if gw_col:
                    rename[gw_col] = "gw_raw"

                for stat, cands in targets.items():
                    col = self._find_col(raw, cands)
                    if col:
                        rename[col] = stat
                    else:
                        log.warning(f"  ↳ {stat_type}/{stat}: column not found")

                sub = raw[list(rename)].rename(columns=rename)

                for stat in targets:
                    if stat in sub.columns:
                        sub[stat] = pd.to_numeric(sub[stat], errors="coerce").fillna(0.0)

                frames.append(sub)
                found = [s for s in targets if s in sub.columns]
                log.info(f"  ↳ captured: {found}")

            except AttributeError:
                log.warning(f"  ↳ read_player_match_stats not available for {stat_type}")
            except Exception as e:
                log.warning(f"  ↳ FBref {stat_type} failed: {e}")

        if not frames:
            log.warning("⚠  All FBref fetches failed — advanced stats will be 0")
            return pd.DataFrame()

        merged = frames[0]
        for frame in frames[1:]:
            on = [c for c in ["player", "gw_raw"] if c in merged.columns and c in frame.columns]
            merged = merged.merge(frame, on=on, how="outer")

        # Parse "Matchweek 34" → 34
        if "gw_raw" in merged.columns:
            merged["GW"] = pd.to_numeric(
                merged["gw_raw"].astype(str).str.extract(r"(\d+)")[0],
                errors="coerce",
            )

        for stat in FBREF_STATS:
            if stat not in merged.columns:
                merged[stat] = 0.0

        log.info(f"✓ FBref merged: {merged.shape[0]} player-GW rows")

        try:
            merged.to_parquet(cache_path)
        except Exception:
            pass  # pyarrow missing — just skip caching

        return merged


# ============================================================================
# PLAYER NAME MATCHING
# ============================================================================

def fuzzy_match_names(fpl_names: list[str], fbref_names: list[str]) -> dict[str, str]:
    """
    Map FPL display names → FBref player names.
    Strategy: exact → last-name-only → fuzzy (cutoff 0.76).
    """
    fb_lower = {n.lower(): n for n in fbref_names}
    mapping: dict[str, str] = {}

    for name in fpl_names:
        low = name.lower()

        # 1. Exact
        if low in fb_lower:
            mapping[name] = fb_lower[low]
            continue

        # 2. Last name only (unambiguous)
        last = low.split()[-1]
        hits = [orig for k, orig in fb_lower.items() if k.endswith(last)]
        if len(hits) == 1:
            mapping[name] = hits[0]
            continue

        # 3. Fuzzy
        close = difflib.get_close_matches(low, list(fb_lower), n=1, cutoff=0.76)
        if close:
            mapping[name] = fb_lower[close[0]]

    return mapping


# ============================================================================
# DATA LOADING
# ============================================================================

def load_fpl_data(client: FPLDataClient, boot: dict) -> pd.DataFrame:
    """Pull per-player per-GW history from FPL API. Also captures availability."""
    elements = pd.DataFrame(boot["elements"])
    team_map = dict(zip(
        pd.DataFrame(boot["teams"])["id"],
        pd.DataFrame(boot["teams"])["short_name"],
    ))

    # Availability snapshot (current week)
    avail = elements.set_index("id")[["status", "chance_of_playing_next_round"]].to_dict("index")

    rows = []
    for _, el in elements.iterrows():
        if el["minutes"] == 0:
            continue
        try:
            summary = client.element_summary(int(el["id"]))
        except requests.HTTPError:
            continue

        pid = int(el["id"])
        av  = avail.get(pid, {})

        for gw in summary.get("history", []):
            rows.append({
                "name":             f"{el['first_name']} {el['second_name']}",
                "player_id":        pid,
                "team":             team_map.get(el["team"], str(el["team"])),
                "position":         POSITION_MAP.get(el["element_type"]),
                "GW":               gw["round"],
                "minutes":          gw["minutes"],
                "goals_scored":     gw["goals_scored"],
                "assists":          gw["assists"],
                "clean_sheets":     gw["clean_sheets"],
                "goals_conceded":   gw["goals_conceded"],
                "saves":            gw["saves"],
                "yellow_cards":     gw["yellow_cards"],
                "red_cards":        gw["red_cards"],
                "was_home":         int(gw["was_home"]),
                "expected_goals":   float(gw.get("expected_goals", 0) or 0),
                "expected_assists": float(gw.get("expected_assists", 0) or 0),
                "influence":        float(gw["influence"]),
                "creativity":       float(gw["creativity"]),
                "threat":           float(gw["threat"]),
                "ict_index":        float(gw["ict_index"]),
                "total_points":            gw["total_points"],
                "expected_goals_conceded": float(gw.get("expected_goals_conceded", 0) or 0),
                "status":               av.get("status", "a"),
                "chance_of_playing":    av.get("chance_of_playing_next_round") or 100,
            })

    df = pd.DataFrame(rows)
    log.info(f"✓ FPL: {len(df)} player-GW records")
    return df


def build_team_gw_stats(df: pd.DataFrame) -> pd.DataFrame:
    gk = df[df["position"] == "GK"]
    def_stats = gk.groupby(["team", "GW"]).agg(
        team_goals_conceded=("goals_conceded", "first"),
        team_xg_conceded=("expected_goals_conceded", "first"),
    ).reset_index()
    att_stats = df.groupby(["team", "GW"]).agg(
        team_goals_scored=("goals_scored", "sum"),
        team_xg_scored=("expected_goals", "sum"),
    ).reset_index()
    ts = att_stats.merge(def_stats, on=["team", "GW"], how="left").sort_values(["team", "GW"])
    for w in [3, 5]:
        for col in ["team_goals_conceded", "team_xg_conceded", "team_goals_scored", "team_xg_scored"]:
            if col in ts.columns:
                ts[f"{col}_avg{w}"] = ts.groupby("team")[col].transform(
                    lambda s, ww=w: s.shift(1).rolling(ww, min_periods=1).mean())
    return ts


def enrich_with_fixture_context(df: pd.DataFrame, ts: pd.DataFrame,
                                fixtures: list, boot: dict) -> pd.DataFrame:
    teams_df = pd.DataFrame(boot["teams"])
    tid2name = dict(zip(teams_df["id"], teams_df["short_name"]))
    str_df = teams_df[["short_name", "strength_attack_home", "strength_attack_away",
                        "strength_defence_home", "strength_defence_away"]].rename(
                            columns={"short_name": "opp"})
    fix_rows = []
    for f in fixtures:
        gw = f.get("event")
        if not gw:
            continue
        h, a = tid2name.get(f["team_h"], ""), tid2name.get(f["team_a"], "")
        fix_rows += [{"team": h, "GW": gw, "opp": a, "was_home": 1},
                     {"team": a, "GW": gw, "opp": h, "was_home": 0}]
    fdf = pd.DataFrame(fix_rows)
    opp_cols = ["team", "GW", "team_goals_conceded_avg5", "team_xg_conceded_avg5",
                "team_goals_scored_avg5", "team_xg_scored_avg5"]
    odf = ts[[c for c in opp_cols if c in ts.columns]].rename(
        columns={"team": "opp", "team_goals_conceded_avg5": "opp_gc_avg5",
                 "team_xg_conceded_avg5": "opp_xgc_avg5",
                 "team_goals_scored_avg5": "opp_gs_avg5",
                 "team_xg_scored_avg5": "opp_xgs_avg5"})
    fdf = fdf.merge(odf, on=["opp", "GW"], how="left")
    fdf = fdf.merge(str_df, on="opp", how="left")
    fdf["opp_att_str"] = fdf.apply(lambda r: r["strength_attack_away"] if r["was_home"] == 1
                                   else r["strength_attack_home"], axis=1)
    fdf["opp_def_str"] = fdf.apply(lambda r: r["strength_defence_away"] if r["was_home"] == 1
                                   else r["strength_defence_home"], axis=1)
    fdf = fdf.drop(columns=["opp", "strength_attack_home", "strength_attack_away",
                             "strength_defence_home", "strength_defence_away"])
    for c in ["opp_gc_avg5", "opp_xgc_avg5", "opp_gs_avg5", "opp_xgs_avg5"]:
        fdf[c] = fdf[c].fillna(1.2)
    fdf[["opp_att_str", "opp_def_str"]] = fdf[["opp_att_str", "opp_def_str"]].fillna(1000)
    df = df.drop(columns=["was_home"], errors="ignore")
    return df.merge(fdf, on=["team", "GW"], how="left")


def load_current_season_data(
    fpl_client: FPLDataClient,
    fbref_client: FBrefDataClient,
    boot: dict,
    fixtures: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fpl_df   = load_fpl_data(fpl_client, boot)
    fbref_df = fbref_client.load_match_stats()

    if fbref_df.empty or "player" not in fbref_df.columns or "GW" not in fbref_df.columns:
        log.warning("⚠  FBref data unavailable — advanced stats defaulted to 0")
        for col in FBREF_STATS:
            fpl_df[col] = 0.0
        df = fpl_df
    else:
        name_map = fuzzy_match_names(
            fpl_df["name"].unique().tolist(),
            fbref_df["player"].unique().tolist(),
        )
        pct = 100 * len(name_map) / fpl_df["name"].nunique()
        log.info(f"✓ Name match: {len(name_map)}/{fpl_df['name'].nunique()} ({pct:.0f}%)")
        fpl_df["_fb"] = fpl_df["name"].map(name_map)
        stat_cols  = [c for c in FBREF_STATS if c in fbref_df.columns]
        fbref_slim = fbref_df[["player", "GW"] + stat_cols].rename(columns={"player": "_fb"})
        df = fpl_df.merge(fbref_slim, on=["_fb", "GW"], how="left")
        for col in FBREF_STATS:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)
        df.drop(columns=["_fb"], inplace=True)
        log.info(f"✓ Merged dataset: {df.shape}")

    ts = build_team_gw_stats(df)
    df = enrich_with_fixture_context(df, ts, fixtures, boot)
    return df, ts


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lagged rolling averages for every stat — no future leakage."""
    df = df.sort_values(["name", "GW"]).reset_index(drop=True)

    all_stats = FPL_ROLLING_STATS + FBREF_STATS

    for window in [3, 5, 10]:
        for stat in all_stats:
            if stat not in df.columns:
                continue
            df[f"{stat}_avg{window}"] = (
                df.groupby("name")[stat]
                .transform(lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean())
            )

        df[f"minutes_avg{window}"] = (
            df.groupby("name")["minutes"]
            .transform(lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean())
        )
        df[f"starts_rate{window}"] = (
            df.groupby("name")["minutes"]
            .transform(lambda s, w=window: (s.shift(1) >= 60).rolling(w, min_periods=1).mean())
        )

    for span in [3, 5, 10]:
        for stat in all_stats:
            if stat not in df.columns:
                continue
            df[f"{stat}_ewm{span}"] = (
                df.groupby("name")[stat]
                .transform(lambda s, sp=span: s.shift(1).ewm(span=sp, min_periods=1).mean())
            )

    df["games_played"] = df.groupby("name").cumcount()
    df = df.dropna(subset=["minutes_avg5"])

    log.info(f"✓ Features engineered: {df.shape}")
    return df


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_component_models(df: pd.DataFrame) -> tuple[dict, list]:
    """One LGBMRegressor per (position, target stat), time-series CV."""
    feature_cols = [c for c in df.columns if c not in NON_FEATURE]

    bundle: dict = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        sub = df[df["position"] == pos].copy()
        if len(sub) < 50:
            continue

        bundle[pos] = {}
        log.info(f"Training {pos}...")

        for target in TRAIN_TARGETS:
            if target not in sub.columns:
                continue
            y = sub[target].fillna(0)
            if y.sum() == 0:
                continue

            X = sub[feature_cols].fillna(0)

            tscv = TimeSeriesSplit(n_splits=2)
            for tr, te in tscv.split(X):
                m = LGBMRegressor(n_estimators=200, learning_rate=0.1, num_leaves=31,
                                  verbose=-1, random_state=42)
                m.fit(X.iloc[tr], y.iloc[tr])

            m = LGBMRegressor(n_estimators=200, learning_rate=0.1, num_leaves=31,
                              verbose=-1, random_state=42)
            m.fit(X, y)
            bundle[pos][target] = m

    return bundle, feature_cols


# ============================================================================
# PREDICTION  +  FULL SLEEPER SCORING
# ============================================================================

def predict_next_gw(df: pd.DataFrame, bundle: dict, feature_cols: list,
                    ts: pd.DataFrame, fixtures: list, boot: dict) -> pd.DataFrame:
    next_gw = int(df["GW"].max()) + 1
    base = df.sort_values("GW").groupby("name").tail(1).copy()
    base["GW"] = next_gw

    teams_df = pd.DataFrame(boot["teams"])
    tid2name = dict(zip(teams_df["id"], teams_df["short_name"]))
    str_df = teams_df.set_index("short_name")[
        ["strength_attack_home", "strength_attack_away",
         "strength_defence_home", "strength_defence_away"]].to_dict("index")
    next_fix = {}
    for f in fixtures:
        if f.get("event") == next_gw:
            h = tid2name.get(f["team_h"], "")
            a = tid2name.get(f["team_a"], "")
            next_fix[h] = {"opp": a, "is_home": 1}
            next_fix[a] = {"opp": h, "is_home": 0}
    latest_ts = ts.sort_values("GW").groupby("team").last()
    for idx, row in base.iterrows():
        fi   = next_fix.get(row["team"], {})
        ih   = fi.get("is_home", 1)
        opp  = fi.get("opp", "")
        os   = latest_ts.loc[opp].to_dict() if opp in latest_ts.index else {}
        side = "away" if ih else "home"
        st   = str_df.get(opp, {})
        base.at[idx, "was_home"]     = ih
        base.at[idx, "opp_gc_avg5"]  = os.get("team_goals_conceded_avg5", 1.2)
        base.at[idx, "opp_xgc_avg5"] = os.get("team_xg_conceded_avg5", 1.2)
        base.at[idx, "opp_gs_avg5"]  = os.get("team_goals_scored_avg5", 1.2)
        base.at[idx, "opp_xgs_avg5"] = os.get("team_xg_scored_avg5", 1.2)
        base.at[idx, "opp_att_str"]  = st.get(f"strength_attack_{side}", 1000)
        base.at[idx, "opp_def_str"]  = st.get(f"strength_defence_{side}", 1000)

    rows = []
    for _, row in base.iterrows():
        pos = row["position"]
        if pos not in bundle:
            continue

        # Availability multiplier
        status = row.get("status", "a")
        chance = float(row.get("chance_of_playing", 100))
        if status in ("i", "s") or chance == 0:
            avail_mult = 0.0
        elif status == "d":
            avail_mult = chance / 100
        else:
            avail_mult = 1.0

        x = row[feature_cols].fillna(0).to_frame().T.astype(float)

        s: dict[str, float] = {
            t: max(0.0, float(bundle[pos][t].predict(x)[0]))
            for t in bundle[pos]
        }

        exp_min = float(row.get("minutes_avg5", 60))
        prob_cs = s.get("clean_sheets", 0) * min(1.0, exp_min / 60)

        # Full Sleeper point formula — every scoring category
        pts = (
            s.get("goals_scored",        0) * _pos_score("goals", pos)
          + s.get("assists",             0) * _pos_score("assists", pos)
          + s.get("shots_on_target",     0) * SLEEPER_SCORING["shots_on_target"]
          + s.get("key_passes",          0) * _pos_score("key_passes", pos)
          + s.get("successful_dribbles", 0) * SLEEPER_SCORING["successful_dribbles"]
          + s.get("accurate_crosses",    0) * SLEEPER_SCORING["accurate_crosses"]
          + s.get("yellow_cards",        0) * SLEEPER_SCORING["yellow_card"]
          + s.get("red_cards",           0) * SLEEPER_SCORING["red_card"]
          + s.get("aerials_won",         0) * _pos_score("aerials_won", pos)
          + s.get("effective_clearances",0) * _pos_score("effective_clearances", pos)
          + s.get("saves",               0) * SLEEPER_SCORING["saves"]
          + prob_cs                         * _pos_score("clean_sheet_60plus", pos)
          + s.get("tackles_won",         0) * SLEEPER_SCORING["tackles_won"]
          + s.get("interceptions",       0) * SLEEPER_SCORING["interceptions"]
          + s.get("blocked_shots",       0) * SLEEPER_SCORING["blocked_shots"]
          + s.get("goals_conceded",      0) * _pos_score("goals_against", pos)
        ) * avail_mult

        rows.append({
            "name":         row["name"],
            "team":         row["team"],
            "position":     pos,
            "GW":           next_gw,
            "avail":        f"{int(chance)}%" if status != "a" else "OK",
            "exp_min":      round(exp_min, 1),
            "exp_goals":    round(s.get("goals_scored",        0), 2),
            "exp_assists":  round(s.get("assists",             0), 2),
            "exp_sot":      round(s.get("shots_on_target",     0), 2),
            "exp_kp":       round(s.get("key_passes",          0), 2),
            "exp_tkl":      round(s.get("tackles_won",         0), 2),
            "exp_int":      round(s.get("interceptions",       0), 2),
            "exp_saves":    round(s.get("saves",               0), 2),
            "exp_cs":       round(prob_cs,                         2),
            "sleeper_pts":  round(pts,                             2),
        })

    result = pd.DataFrame(rows).sort_values("sleeper_pts", ascending=False)
    log.info(f"✓ Predicted {len(result)} players for GW{next_gw}")
    return result


# ============================================================================
# INTERACTIVE CLI
# ============================================================================

_DISPLAY_COLS = ["name", "team", "position", "avail", "exp_min",
                 "exp_goals", "exp_assists", "exp_sot", "exp_tkl", "exp_int",
                 "exp_saves", "exp_cs", "sleeper_pts"]


def print_menu(predictions: pd.DataFrame) -> None:
    while True:
        print("\n" + "=" * 90)
        print(f"⚽  SLEEPER FANTASY  GW{predictions['GW'].iloc[0]}  PREDICTOR")
        print("=" * 90)
        print("\nOptions:")
        print("  [1] Top 50 all positions")
        print("  [2] Top 10 by position")
        print("  [3] Filter by team")
        print("  [4] Filter by position + min expected points")
        print("  [5] Show stat breakdown for a player")
        print("  [6] Export to CSV")
        print("  [7] Exit")
        print()

        choice = input("Choose (1-7): ").strip()

        if choice == "1":
            print("\n🏆 TOP 50 SLEEPER SCORERS\n")
            print(predictions.head(50)[_DISPLAY_COLS].to_string(index=False))

        elif choice == "2":
            for pos in ["GK", "DEF", "MID", "FWD"]:
                sub = predictions[predictions["position"] == pos].head(10)
                if not sub.empty:
                    print(f"\n── {pos} ──")
                    print(sub[_DISPLAY_COLS].to_string(index=False))

        elif choice == "3":
            team = input("Team short code (e.g. ARS, LIV, MCI): ").upper()
            sub = predictions[predictions["team"] == team].head(20)
            if not sub.empty:
                print(f"\n{team} PLAYERS:\n")
                print(sub[_DISPLAY_COLS].to_string(index=False))
            else:
                print(f"No predictions found for {team}")

        elif choice == "4":
            pos     = input("Position (GK/DEF/MID/FWD): ").upper()
            min_pts = float(input("Minimum expected points: "))
            sub = predictions[
                (predictions["position"] == pos) & (predictions["sleeper_pts"] >= min_pts)
            ]
            if not sub.empty:
                print(f"\n{pos} players ≥ {min_pts} pts:\n")
                print(sub[_DISPLAY_COLS].to_string(index=False))
            else:
                print(f"No {pos} players found above {min_pts} pts")

        elif choice == "5":
            name = input("Player name (partial OK): ").lower()
            sub = predictions[predictions["name"].str.lower().str.contains(name)]
            if not sub.empty:
                print()
                print(sub[_DISPLAY_COLS].to_string(index=False))
            else:
                print("Player not found")

        elif choice == "6":
            fname = f"sleeper_gw{predictions['GW'].iloc[0]}.csv"
            predictions.to_csv(fname, index=False)
            log.info(f"✓ Saved to {fname}")

        elif choice == "7":
            print("\nGoodbye! ⚽")
            break


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    log.info("🚀 SLEEPER FANTASY PREMIER LEAGUE PREDICTOR")
    log.info("=" * 50)

    fpl_client   = FPLDataClient()
    fbref_client = FBrefDataClient()
    boot         = fpl_client.bootstrap()
    fixtures     = fpl_client.fixtures()
    df, ts       = load_current_season_data(fpl_client, fbref_client, boot, fixtures)
    feat         = engineer_features(df)
    bundle, feature_cols = train_component_models(feat)
    predictions  = predict_next_gw(feat, bundle, feature_cols, ts, fixtures, boot)
    print_menu(predictions)


if __name__ == "__main__":
    main()
