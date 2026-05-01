"""
Sleeper Fantasy Premier League Weekly Predictor
================================================

Data source: FPL API — goals, assists, clean sheets, saves, cards,
goals conceded, minutes, xG, xA, ICT index, player availability,
own goals, penalties missed/saved.

Usage:
    python sleeper_predictor.py

Requirements:
    pip install requests pandas lightgbm numpy scikit-learn pyarrow
"""

import json
import logging
import time
import unicodedata
import warnings
from datetime import datetime
from pathlib import Path

import re

import numpy as np
import pandas as pd
import requests

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
    "own_goals":            -5.0,
    "penalties_missed":     -4.0,
    "penalties_saved":       8.0,
    # Additional Sleeper scoring categories
    "smothers":             1.0,
    "high_claims":          1.0,
    "dispossessed":        -0.5,
    "penalty_kicks_drawn":  2.0,
    "second_yellow":       -5.0,
}


FPL_ROLLING_STATS = [
    "goals_scored", "assists", "expected_goals", "expected_assists",
    "clean_sheets", "saves", "total_points", "influence", "creativity",
    "threat", "goals_conceded", "expected_goals_conceded",
    "own_goals", "penalties_missed", "penalties_saved",
]


# ============================================================================
# HELPERS
# ============================================================================

def _current_season_str() -> str:
    """Return soccerdata season string, e.g. '2526' for 2025-26."""
    now = datetime.now()
    y = now.year if now.month >= 8 else now.year - 1
    return f"{str(y)[2:]}{str(y + 1)[2:]}"


def _sleeper_season_year() -> int:
    """Return start year of current soccer season (e.g. 2025 for 2025-26)."""
    now = datetime.now()
    return now.year if now.month >= 8 else now.year - 1


def _pos_score(key: str, pos: str) -> float:
    v = SLEEPER_SCORING[key]
    return v.get(pos, 0) if isinstance(v, dict) else float(v)


def _norm_name(name: str) -> str:
    """Lowercase, strip accents — for FBref↔FPL name matching."""
    nfkd = unicodedata.normalize("NFKD", name.lower().strip())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# ============================================================================
# SLEEPER STATS API  (per-GW real points)
# ============================================================================

SLEEPER_API = "https://api.sleeper.app/v1"

# Sleeper API stat field names (from api.sleeper.app/v1/stats/clubsoccer:epl)
_SLEEPER_FIELD = {
    "goals":                ["gs"],
    "assists":              ["ast", "asts"],
    "shots_on_target":      ["sot"],
    "key_passes":           ["kp"],
    "successful_dribbles":  ["drb"],
    "accurate_crosses":     ["acnc"],
    "aerials_won":          ["aer"],
    "effective_clearances": ["clr"],
    "saves":                ["svs", "saves"],
    "clean_sheets":         ["cos"],
    "tackles_won":          ["tkl"],
    "interceptions":        ["int"],
    "blocked_shots":        ["blk"],
    "goals_against":        ["ga"],
    "own_goals":            ["og"],
    "penalties_missed":     ["pm"],
    "penalties_saved":      ["ps"],
    "yellow_card":          ["yc"],
    "red_card":             ["rc"],
    "minutes":              ["min"],
}


def _stat(stats: dict, key: str) -> float:
    for field in _SLEEPER_FIELD.get(key, [key]):
        v = stats.get(field)
        if v is not None:
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
    return 0.0


def _sleeper_pts_from_api(stats: dict, pos: str) -> float:
    """Compute Sleeper points from a Sleeper API stats dict (full Opta stats)."""
    mins = _stat(stats, "minutes") or 90
    cs   = _stat(stats, "clean_sheets") if mins >= 60 else 0
    return (
        _stat(stats, "goals")               * _pos_score("goals",               pos)
      + _stat(stats, "assists")              * _pos_score("assists",             pos)
      + _stat(stats, "shots_on_target")      * SLEEPER_SCORING["shots_on_target"]
      + _stat(stats, "key_passes")           * _pos_score("key_passes",         pos)
      + _stat(stats, "successful_dribbles")  * SLEEPER_SCORING["successful_dribbles"]
      + _stat(stats, "accurate_crosses")     * SLEEPER_SCORING["accurate_crosses"]
      + _stat(stats, "aerials_won")          * _pos_score("aerials_won",        pos)
      + _stat(stats, "effective_clearances") * _pos_score("effective_clearances",pos)
      + _stat(stats, "saves")                * SLEEPER_SCORING["saves"]
      + _stat(stats, "tackles_won")          * SLEEPER_SCORING["tackles_won"]
      + _stat(stats, "interceptions")        * SLEEPER_SCORING["interceptions"]
      + _stat(stats, "blocked_shots")        * SLEEPER_SCORING["blocked_shots"]
      + _stat(stats, "goals_against")        * _pos_score("goals_against",      pos)
      + _stat(stats, "own_goals")            * SLEEPER_SCORING["own_goals"]
      + _stat(stats, "penalties_missed")     * SLEEPER_SCORING["penalties_missed"]
      + _stat(stats, "penalties_saved")      * SLEEPER_SCORING["penalties_saved"]
      + _stat(stats, "yellow_card")          * SLEEPER_SCORING["yellow_card"]
      + _stat(stats, "red_card")             * SLEEPER_SCORING["red_card"]
      + cs                                   * _pos_score("clean_sheet_60plus", pos)
    )


def load_sleeper_hist_pts(
    current_gw: int,
    name_map: dict[str, str],
    pos_map: dict[str, str],
    sport: str = "clubsoccer:epl",
    cache_dir: str = ".fpl_cache",
    n_weeks: int = 7,      # GWs to fetch for per-90 stat calculations
    n_avg: int = 5,        # GWs to use for the displayed avg_pts_5 column
    min_mins: int = 45,    # Minimum minutes to include a GW in per-90 (filters cameos)
) -> tuple[dict[str, float], dict[str, dict], list[str]]:
    """
    Fetch last n_weeks of stats from Sleeper API.
    Returns:
      - {norm_name: avg_pts over last n_avg GWs}  — display column only
      - {norm_name: {stat: per90}}                 — per-90 over last n_weeks GWs
      - [debug_lines]
    """
    cache_dir_p = Path(cache_dir)
    cache_dir_p.mkdir(exist_ok=True)
    start_gw    = max(1, current_gw - n_weeks + 1)
    cache_file  = cache_dir_p / f"sleeper_hist_gw{start_gw}_{current_gw}.json"
    per90_file  = cache_dir_p / f"sleeper_per90_v6_gw{start_gw}_{current_gw}.json"

    debug: list[str] = []

    if (cache_file.exists() and per90_file.exists() and
            (time.time() - cache_file.stat().st_mtime) < 3600):
        debug.append("  ✅ Loaded from cache")
        return (json.loads(cache_file.read_text()),
                json.loads(per90_file.read_text()),
                debug)

    year = _sleeper_season_year()

    player_pts:   dict[str, list[float]] = {}  # norm_name -> [pts per gw]
    player_stats: dict[str, list[dict]]  = {}  # norm_name -> [stats dict per gw]
    all_seen_keys: set[str] = set()            # for diagnostic

    for gw in range(max(1, current_gw - n_weeks + 1), current_gw + 1):
        url = f"{SLEEPER_API}/stats/{sport}/regular/{year}/{gw}"
        try:
            r = requests.get(url, timeout=15)
            debug.append(f"  GW{gw}: {r.status_code}")
            if r.status_code != 200:
                continue
            week_stats = r.json()
            if not week_stats:
                continue
            for pid, stats in week_stats.items():
                if not isinstance(stats, dict):
                    continue
                # Collect all non-zero field names for diagnostics
                all_seen_keys.update(k for k, v in stats.items() if v)
                norm = _norm_name(name_map.get(str(pid), ""))
                if not norm:
                    continue
                pts = float(stats.get("pts_std", 0.0))
                player_pts.setdefault(norm, []).append(pts)
                player_stats.setdefault(norm, []).append(stats)
        except Exception as e:
            debug.append(f"  GW{gw}: error — {e}")

    # Save all observed field names for debugging in the UI
    if all_seen_keys:
        (cache_dir_p / "sleeper_stat_keys.json").write_text(
            json.dumps(sorted(all_seen_keys))
        )

    # avg_pts_5: last n_avg GWs only (display column — not used in prediction)
    avg_pts = {
        name: round(sum(pts[-n_avg:]) / len(pts[-n_avg:]), 1)
        for name, pts in player_pts.items() if pts
    }

    # Compute per-90 averages over n_weeks GWs, skipping cameo appearances
    per90: dict[str, dict] = {}
    _per90_keys = [
        "g",                           # goals scored (gs = games started, not goals)
        "ast", "asts", "at",           # assists — "at" is the real Sleeper field
        "sot", "kp", "acnc", "drb",    # attacking
        "int", "tkw", "clr", "aer",    # defensive
        "bs",                          # blocked shots
        "sv",                          # GK saves
        "yc", "rc", "yc2",            # cards
        "ga", "og",                    # goals against, own goals
        "dis",                         # dispossessed
        "sm", "hcs",                   # smothers, high claims
        "pkd", "pkm", "ps",           # pen kicks drawn/missed/saved
    ]
    for norm, gw_list in player_stats.items():
        totals: dict[str, float] = {}
        total_min = 0.0
        for s in gw_list:
            mins = float(s.get("min", 0) or 0)
            if mins < min_mins:        # skip cameos — inflates per-90
                continue
            total_min += mins
            for k in _per90_keys:
                totals[k] = totals.get(k, 0.0) + float(s.get(k, 0) or 0)
        if total_min >= 90:
            n90 = total_min / 90
            p90 = {k: round(v / n90, 3) for k, v in totals.items()}
            # Normalise assist: "at" is the real Sleeper field; ast/asts are fallbacks
            p90["ast"] = max(p90.get("at", 0.0), p90.get("ast", 0.0), p90.get("asts", 0.0))
            per90[norm] = p90

    cache_file.write_text(json.dumps(avg_pts))
    per90_file.write_text(json.dumps(per90))
    debug.append(f"  ✅ {len(avg_pts)} players with historical pts, {len(per90)} with per-90 stats")
    return avg_pts, per90, debug


def load_understat_xg(cache_dir: str = ".fpl_cache") -> dict[str, dict]:
    """
    Fetch season xG/xA per player from Understat.
    Returns {norm_name: {xg_per90, xa_per90}}.
    xG/xA are better predictors of future scoring than raw goals/assists.
    """
    cache_dir_p = Path(cache_dir)
    cache_dir_p.mkdir(exist_ok=True)
    year = _sleeper_season_year()
    cache_file = cache_dir_p / f"understat_xg_{year}.json"

    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 3600 * 6:
        return json.loads(cache_file.read_text())

    url = f"https://understat.com/league/EPL/{year}"
    try:
        r = requests.get(url, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0 (compatible; stats-fetcher/1.0)"})
        if r.status_code != 200:
            log.warning(f"Understat returned {r.status_code}")
            return {}

        m = re.search(r"playersData\s*=\s*JSON\.parse\('(.+?)'\)", r.text)
        if not m:
            log.warning("Understat: playersData not found in page")
            return {}

        # Understat encodes the JSON string with unicode escapes
        raw = m.group(1).encode("utf-8").decode("unicode_escape")
        players = json.loads(raw)

        result: dict[str, dict] = {}
        for p in players:
            name = _norm_name(str(p.get("player_name", "")))
            mins = float(p.get("time", 0) or 0)
            if not name or mins < 90:
                continue
            n90 = mins / 90
            result[name] = {
                "xg_per90": round(float(p.get("xG", 0) or 0) / n90, 3),
                "xa_per90": round(float(p.get("xA", 0) or 0) / n90, 3),
            }

        cache_file.write_text(json.dumps(result))
        log.info(f"✅ Understat: {len(result)} players with xG/xA")
        return result

    except Exception as e:
        log.warning(f"Understat fetch failed: {e}")
        return {}


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
# DATA LOADING
# ============================================================================


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

        pos = POSITION_MAP.get(el["element_type"], "MID")
        for gw in summary.get("history", []):
            mins = gw["minutes"]
            cs   = gw["clean_sheets"] if mins >= 60 else 0
            slp  = (
                gw["goals_scored"]          * _pos_score("goals",               pos)
              + gw["assists"]               * _pos_score("assists",             pos)
              + gw["saves"]                 * SLEEPER_SCORING["saves"]
              + gw["goals_conceded"]        * _pos_score("goals_against",       pos)
              + gw.get("own_goals", 0)      * SLEEPER_SCORING["own_goals"]
              + gw.get("penalties_missed",0)* SLEEPER_SCORING["penalties_missed"]
              + gw.get("penalties_saved", 0)* SLEEPER_SCORING["penalties_saved"]
              + gw["yellow_cards"]          * SLEEPER_SCORING["yellow_card"]
              + gw["red_cards"]             * SLEEPER_SCORING["red_card"]
              + cs                          * _pos_score("clean_sheet_60plus",  pos)
            )
            rows.append({
                "name":             f"{el['first_name']} {el['second_name']}",
                "display_name":     el.get("web_name", el["second_name"]),
                "player_id":        pid,
                "team":             team_map.get(el["team"], str(el["team"])),
                "position":         pos,
                "GW":               gw["round"],
                "minutes":          mins,
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
                "sleeper_pts_hist":        round(slp, 2),
                "expected_goals_conceded": float(gw.get("expected_goals_conceded", 0) or 0),
                "own_goals":            gw.get("own_goals", 0),
                "penalties_missed":     gw.get("penalties_missed", 0),
                "penalties_saved":      gw.get("penalties_saved", 0),
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
    boot: dict,
    fixtures: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_fpl_data(fpl_client, boot)
    ts = build_team_gw_stats(df)
    df = enrich_with_fixture_context(df, ts, fixtures, boot)
    return df, ts


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lagged rolling averages for every stat — no future leakage."""
    df = df.sort_values(["name", "GW"]).reset_index(drop=True)

    all_stats = FPL_ROLLING_STATS

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
# PREDICTION  +  FULL SLEEPER SCORING
# ============================================================================

def predict_next_gw(df: pd.DataFrame,
                    ts: pd.DataFrame, fixtures: list, boot: dict) -> pd.DataFrame:
    data_gw = int(df["GW"].max())
    # Switch to the following GW as soon as the current one has kicked off.
    # This means predictions are ready before the last game ends — important
    # for waiver deadlines which run ~2 hours after the final whistle.
    current_gw_started = any(
        f.get("event") == data_gw + 1 and f.get("started")
        for f in fixtures
    )
    next_gw = data_gw + 2 if current_gw_started else data_gw + 1
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
            next_fix[h] = {"opp": a, "is_home": 1,
                           "fdr": f.get("team_h_difficulty", 3)}
            next_fix[a] = {"opp": h, "is_home": 0,
                           "fdr": f.get("team_a_difficulty", 3)}
    latest_ts = ts.sort_values("GW").groupby("team").last()
    for idx, row in base.iterrows():
        fi   = next_fix.get(row["team"], {})
        ih   = fi.get("is_home", 1)
        opp  = fi.get("opp", "")
        os   = latest_ts.loc[opp].to_dict() if opp in latest_ts.index else {}
        side = "away" if ih else "home"
        st   = str_df.get(opp, {})
        base.at[idx, "opp"]          = opp or "TBC"
        base.at[idx, "ha"]           = "H" if ih else "A"
        base.at[idx, "fdr"]          = fi.get("fdr", 3)
        base.at[idx, "was_home"]     = ih
        base.at[idx, "opp_gc_avg5"]  = os.get("team_goals_conceded_avg5", 1.2)
        base.at[idx, "opp_xgc_avg5"] = os.get("team_xg_conceded_avg5", 1.2)
        base.at[idx, "opp_gs_avg5"]  = os.get("team_goals_scored_avg5", 1.2)
        base.at[idx, "opp_xgs_avg5"] = os.get("team_xg_scored_avg5", 1.2)
        base.at[idx, "opp_att_str"]  = st.get(f"strength_attack_{side}", 1000)
        base.at[idx, "opp_def_str"]  = st.get(f"strength_defence_{side}", 1000)

    # Position map for Sleeper stats scoring: norm_name -> position
    pos_map = (
        df.groupby("name")["position"].last()
        .reset_index()
        .assign(norm=lambda d: d["name"].map(_norm_name))
        .set_index("norm")["position"]
        .to_dict()
    )

    # Fetch last-5 GW points from Sleeper stats API (full Opta stats included)
    # Need the Sleeper name_map; build a lightweight version from FPL names as fallback
    try:
        _r = requests.get(f"{SLEEPER_API}/players/clubsoccer:epl", timeout=30)
        _name_map = {}
        if _r.status_code == 200:
            for pid, p in _r.json().items():
                full = (p.get("full_name") or
                        f"{p.get('first_name','')} {p.get('last_name','')}".strip())
                if full.strip():
                    _name_map[str(pid)] = full.strip()
    except Exception:
        _name_map = {}

    sleeper_hist, sleeper_per90, _slp_debug = load_sleeper_hist_pts(
        current_gw=next_gw - 1,
        name_map=_name_map,
        pos_map=pos_map,
    )

    # Fallback: FPL-derived partial Sleeper pts (missing Opta stats)
    fpl_last5 = (
        df.sort_values("GW")
        .groupby("name")["sleeper_pts_hist"]
        .apply(lambda s: round(s.tail(5).mean(), 1))
        .to_dict()
    )

    def _avg5(player_name: str) -> float:
        norm = _norm_name(player_name)
        return sleeper_hist.get(norm) or fpl_last5.get(player_name, 0.0)

    understat_xg = load_understat_xg()
    log.info(f"Understat xG/xA loaded for {len(understat_xg)} players")

    GOAL_CAPS    = {"GK": 0.03, "DEF": 0.12, "MID": 0.40, "FWD": 0.80}
    ASSIST_CAPS  = {"GK": 0.02, "DEF": 0.25, "MID": 0.50, "FWD": 0.35}
    GOAL_FLOOR   = {"GK": 0.005, "DEF": 0.01, "MID": 0.03, "FWD": 0.05}
    ASSIST_FLOOR = {"GK": 0.002, "DEF": 0.005, "MID": 0.01, "FWD": 0.01}

    rows = []
    for _, row in base.iterrows():
        pos = row["position"]
        if pos not in ("GK", "DEF", "MID", "FWD"):
            continue

        status = row.get("status", "a")
        chance = float(row.get("chance_of_playing", 100))
        avail_mult = 0.0 if (status in ("i", "s") or chance == 0) else 1.0

        exp_min   = float(row.get("minutes_avg5", 60))
        min_scale = min(1.0, exp_min / 90)

        # ── Fixture adjustments ───────────────────────────────────────────────
        fdr    = int(row.get("fdr", 3))
        opp_gs = max(0.3, float(row.get("opp_gs_avg5", 1.3)))
        opp_gc = max(0.3, float(row.get("opp_gc_avg5", 1.3)))

        fdr_att        = {1: 1.6, 2: 1.3, 3: 1.0, 4: 0.72, 5: 0.48}[fdr]
        opp_def_factor = min(1.6, max(0.5, opp_gc / 1.3))
        att_mult       = fdr_att * opp_def_factor
        fdr_def        = {1: 0.65, 2: 0.82, 3: 1.0, 4: 1.25, 5: 1.55}[fdr]
        is_home        = int(row.get("was_home", 1))
        ha_mult        = 1.08 if is_home else 0.93

        # Poisson clean sheet: P(CS) = e^(-λ)
        lambda_opp = opp_gs * fdr_def * ha_mult
        prob_cs    = float(np.exp(-lambda_opp)) if exp_min >= 60 else 0.0
        prob_cs    = min(0.85, prob_cs)

        # Form indicator (FPL rolling avg — display only)
        pts3  = float(row.get("total_points_avg3",  0))
        pts10 = float(row.get("total_points_avg10", 0))
        form  = "~"
        if pts10 > 0.5:
            ratio = pts3 / pts10
            form  = "🔥" if ratio >= 1.3 else ("❄️" if ratio <= 0.7 else "~")

        # ── Stat sources ──────────────────────────────────────────────────────
        slp_key = _norm_name(str(row["name"]))
        alt_key = _norm_name(str(row.get("display_name", "")))
        sp = sleeper_per90.get(slp_key) or sleeper_per90.get(alt_key) or {}
        us = understat_xg.get(slp_key)  or understat_xg.get(alt_key)  or {}

        # Fallback: match on 2+ shared words — catches "David Raya" → "David Raya Martin"
        if not sp:
            slp_words = set(slp_key.split())
            for k, v in sleeper_per90.items():
                if len(slp_words & set(k.split())) >= 2:
                    sp = v
                    break
        if not us:
            slp_words = set(slp_key.split())
            for k, v in understat_xg.items():
                if len(slp_words & set(k.split())) >= 2:
                    us = v
                    break
        avg5 = _avg5(row["name"])

        if not sp:
            # No Sleeper per-90 data — pure historical avg (no stat blend possible)
            fixture_mult = (att_mult + fdr_def) / 2.0 * ha_mult
            pts = avg5 * fixture_mult * min_scale
            adj_goals = adj_assists = est_sot = est_kp = 0.0
            est_tkl = est_int = adj_saves = 0.0
            prob_cs_out = 0.0
            pts *= avail_mult
            rows.append({
                "name":         row["name"],
                "display_name": row.get("display_name", row["name"]),
                "player_id":    row.get("player_id", ""),
                "team":         row["team"],
                "opp":          row.get("opp", "TBC"),
                "ha":           row.get("ha", "?"),
                "fdr":          int(row.get("fdr", 3)),
                "position":     pos,
                "form":         form,
                "GW":           next_gw,
                "avail":        f"{int(chance)}%" if status != "a" else "OK",
                "avg_pts_5":    avg5,
                "exp_min":      round(exp_min, 1),
                "exp_goals":    0.0, "exp_assists": 0.0,
                "exp_sot":      0.0, "exp_kp":      0.0,
                "exp_tkl":      0.0, "exp_int":     0.0,
                "exp_saves":    0.0, "exp_cs":      0.0,
                "sleeper_pts":  round(pts, 2),
            })
            continue
        else:
            # Goals: blend Sleeper g_per90 (actual) with Understat xG (expected).
            # NOTE: gs = games started (not goals). Goals field is "g".
            # xG regresses lucky/unlucky periods toward true quality.
            g_per90   = sp.get("g",   0.0)
            mins90 = max(0.5, float(row.get("minutes_avg5", 60)) / 90)
            fpl_xg_per90 = float(row.get("expected_goals_avg5",  0) or 0) / mins90
            fpl_xa_per90 = float(row.get("expected_assists_avg5", 0) or 0) / mins90
            xg_per90  = us.get("xg_per90", fpl_xg_per90)
            raw_goals = 0.4 * g_per90 + 0.6 * xg_per90
            adj_goals = min(
                max(raw_goals * att_mult * ha_mult * min_scale, GOAL_FLOOR[pos]),
                GOAL_CAPS[pos] * min_scale,
            )

            # Assists: same blend
            ast_per90   = sp.get("ast", 0.0)
            xa_per90    = us.get("xa_per90", fpl_xa_per90)
            raw_assists = 0.4 * ast_per90 + 0.6 * xa_per90
            adj_assists = min(
                max(raw_assists * att_mult * ha_mult * min_scale, ASSIST_FLOOR[pos]),
                ASSIST_CAPS[pos] * min_scale,
            )

            # Attacking stats (scale with attacking fixture difficulty + H/A)
            est_sot = sp.get("sot",  0.0) * att_mult * ha_mult * min_scale
            est_kp  = sp.get("kp",   0.0) * att_mult * ha_mult * min_scale
            est_crs = sp.get("acnc", 0.0) * att_mult * ha_mult * min_scale
            est_drb = sp.get("drb",  0.0) * att_mult * ha_mult * min_scale

            # Defensive stats (scale with defensive fixture difficulty + H/A)
            # tkw = tackles won (real Sleeper field, separate from interceptions)
            est_tkl = sp.get("tkw", 0.0) * fdr_def * ha_mult * min_scale
            est_int = sp.get("int", 0.0) * fdr_def * ha_mult * min_scale
            est_clr = sp.get("clr", 0.0) * fdr_def * ha_mult * min_scale
            est_aer = sp.get("aer", 0.0) * fdr_def * ha_mult * min_scale
            est_blk = sp.get("bs",  0.0) * fdr_def * ha_mult * min_scale  # bs = blocked shots

            # GK saves (field: sv)
            adj_saves = sp.get("sv", 0.0) * fdr_def * ha_mult * min_scale

            # Goals against (GK: -2, DEF: -2, MID/FWD: scoring weight is 0)
            adj_gc = sp.get("ga", 0.0) * fdr_def * ha_mult * min_scale

            # Cards (intrinsic — no fixture scaling)
            adj_yc  = sp.get("yc",  0.0) * min_scale
            adj_rc  = sp.get("rc",  0.0) * min_scale
            adj_yc2 = sp.get("yc2", 0.0) * min_scale

            # Other Sleeper scoring stats (intrinsic to player style)
            adj_smo = sp.get("sm",  0.0) * min_scale  # smothers
            adj_hcs = sp.get("hcs", 0.0) * min_scale  # high claims succeeded
            adj_dis = sp.get("dis", 0.0) * min_scale  # dispossessed
            adj_pkd = sp.get("pkd", 0.0) * min_scale  # penalty kicks drawn
            adj_pkm = sp.get("pkm", 0.0) * min_scale  # penalty kicks missed (field: pkm)
            adj_og  = sp.get("og",  0.0) * min_scale  # own goals
            adj_ps  = sp.get("ps",  0.0) * min_scale  # penalty kick saves

            prob_cs_out = prob_cs

            stat_pts = (
                adj_goals   * _pos_score("goals", pos)
              + adj_assists * _pos_score("assists", pos)
              + adj_saves   * SLEEPER_SCORING["saves"]
              + adj_gc      * _pos_score("goals_against", pos)
              + adj_yc      * SLEEPER_SCORING["yellow_card"]
              + adj_rc      * SLEEPER_SCORING["red_card"]
              + adj_yc2     * SLEEPER_SCORING["second_yellow"]
              + prob_cs     * _pos_score("clean_sheet_60plus", pos)
              + est_sot     * SLEEPER_SCORING["shots_on_target"]
              + est_kp      * _pos_score("key_passes", pos)
              + est_crs     * SLEEPER_SCORING["accurate_crosses"]
              + est_drb     * SLEEPER_SCORING["successful_dribbles"]
              + est_tkl     * SLEEPER_SCORING["tackles_won"]
              + est_int     * SLEEPER_SCORING["interceptions"]
              + est_blk     * SLEEPER_SCORING["blocked_shots"]
              + est_clr     * _pos_score("effective_clearances", pos)
              + est_aer     * _pos_score("aerials_won", pos)
              + adj_smo     * SLEEPER_SCORING["smothers"]
              + adj_hcs     * SLEEPER_SCORING["high_claims"]
              + adj_dis     * SLEEPER_SCORING["dispossessed"]
              + adj_pkd     * SLEEPER_SCORING["penalty_kicks_drawn"]
              + adj_pkm     * SLEEPER_SCORING["penalties_missed"]
              + adj_og      * SLEEPER_SCORING["own_goals"]
              + adj_ps      * SLEEPER_SCORING["penalties_saved"]
            )
            # Blend 70% stat model + 30% fixture-adjusted historical avg.
            # The historical component anchors player quality — prevents hot-streak
            # inflation (Flemming ≈ Watkins) and cold-patch underrating (Van Hecke).
            hist_mult = (att_mult + fdr_def) / 2.0 * ha_mult * min_scale
            pts = 0.70 * stat_pts + 0.30 * (avg5 * hist_mult)

        pts *= avail_mult

        rows.append({
            "name":         row["name"],
            "display_name": row.get("display_name", row["name"]),
            "player_id":    row.get("player_id", ""),
            "team":         row["team"],
            "opp":          row.get("opp", "TBC"),
            "ha":           row.get("ha", "?"),
            "fdr":          int(row.get("fdr", 3)),
            "position":     pos,
            "form":         form,
            "GW":           next_gw,
            "avail":        f"{int(chance)}%" if status != "a" else "OK",
            "avg_pts_5":    avg5,
            "exp_min":      round(exp_min, 1),
            "exp_goals":    round(adj_goals,   2),
            "exp_assists":  round(adj_assists, 2),
            "exp_sot":      round(est_sot,     2),
            "exp_kp":       round(est_kp,      2),
            "exp_tkl":      round(est_tkl,     2),
            "exp_int":      round(est_int,     2),
            "exp_saves":    round(adj_saves,   2),
            "exp_cs":       round(prob_cs_out, 2),
            "sleeper_pts":  round(pts,         2),
        })

    result = pd.DataFrame(rows).sort_values("sleeper_pts", ascending=False)
    log.info(f"✓ Predicted {len(result)} players for GW{next_gw}")

    # Auto-save for later validation once GW completes
    save_path = Path(".fpl_cache") / f"predictions_gw{next_gw}.csv"
    try:
        result.to_csv(save_path, index=False)
        log.info(f"💾 Predictions saved → {save_path.name}")
    except Exception:
        pass

    return result


# ============================================================================
# GW VALIDATION
# ============================================================================

def _score_actual_gw(row: pd.Series, pos: str) -> float:
    """Approximate actual Sleeper pts from a completed GW row using real stats."""
    if float(row.get("minutes", 0)) == 0:
        return 0.0
    minutes = float(row.get("minutes", 0))
    cs = float(row.get("clean_sheets", 0)) if minutes >= 60 else 0.0

    thr = float(row.get("threat", 0))
    cre = float(row.get("creativity", 0))
    inf = float(row.get("influence", 0))

    if pos == "GK":
        est_sot, est_kp, est_crs, est_drb = 0.0, 0.0, cre/50, 0.0
        est_tkl, est_int, est_blk, est_clr, est_aer = inf/80, inf/90, 0.0, inf/12, inf/8
    elif pos == "DEF":
        est_sot, est_kp, est_crs, est_drb = thr/30, cre/25, cre/18, cre/22
        est_tkl, est_int, est_blk, est_clr, est_aer = inf/18, inf/22, inf/28, inf/10, inf/12
    elif pos == "MID":
        est_sot, est_kp, est_crs, est_drb = thr/18, cre/22, cre/22, cre/25
        est_tkl, est_int, est_blk, est_clr, est_aer = inf/25, inf/30, inf/30, inf/28, inf/18
    else:
        est_sot, est_kp, est_crs, est_drb = thr/18, cre/28, cre/25, cre/20
        est_tkl, est_int, est_blk, est_clr, est_aer = inf/55, inf/65, inf/55, 0.0, inf/14

    return (
        float(row.get("goals_scored",    0)) * _pos_score("goals", pos)
      + float(row.get("assists",         0)) * _pos_score("assists", pos)
      + float(row.get("saves",           0)) * SLEEPER_SCORING["saves"]
      + float(row.get("goals_conceded",  0)) * _pos_score("goals_against", pos)
      + float(row.get("own_goals",       0)) * SLEEPER_SCORING["own_goals"]
      + float(row.get("penalties_missed",0)) * SLEEPER_SCORING["penalties_missed"]
      + float(row.get("penalties_saved", 0)) * SLEEPER_SCORING["penalties_saved"]
      + float(row.get("yellow_cards",    0)) * SLEEPER_SCORING["yellow_card"]
      + float(row.get("red_cards",       0)) * SLEEPER_SCORING["red_card"]
      + cs                                    * _pos_score("clean_sheet_60plus", pos)
      + est_sot * SLEEPER_SCORING["shots_on_target"]
      + est_kp  * _pos_score("key_passes", pos)
      + est_crs * SLEEPER_SCORING["accurate_crosses"]
      + est_drb * SLEEPER_SCORING["successful_dribbles"]
      + est_tkl * SLEEPER_SCORING["tackles_won"]
      + est_int * SLEEPER_SCORING["interceptions"]
      + est_blk * SLEEPER_SCORING["blocked_shots"]
      + est_clr * _pos_score("effective_clearances", pos)
      + est_aer * _pos_score("aerials_won", pos)
    )


def validate_last_gw(df: pd.DataFrame) -> None:
    """Compare the most recent saved predictions against actual completed GW results."""
    cache_dir = Path(".fpl_cache")
    pred_files = sorted(cache_dir.glob("predictions_gw*.csv"))
    if not pred_files:
        print("  No saved predictions yet — run before a gameweek to start tracking.")
        return

    completed_gws = set(df["GW"].unique())
    valid_file = pred_gw = None
    for f in reversed(pred_files):
        try:
            gw = int(f.stem.replace("predictions_gw", ""))
            if gw in completed_gws:
                valid_file, pred_gw = f, gw
                break
        except ValueError:
            continue

    if valid_file is None:
        next_gw = int(df["GW"].max()) + 1
        print(f"  GW{next_gw} predictions saved — check back after the gameweek completes.")
        return

    saved  = pd.read_csv(valid_file)
    gw_df  = df[df["GW"] == pred_gw].copy()
    gw_df["actual_pts"] = gw_df.apply(lambda r: _score_actual_gw(r, r["position"]), axis=1)

    actual = gw_df[["name", "actual_pts"]].set_index("name")
    merged = (
        saved.set_index("name")[["display_name", "position", "sleeper_pts"]]
        .join(actual, how="inner")
        .reset_index()
        .rename(columns={"sleeper_pts": "pred_pts"})
    )
    merged["error"]     = merged["actual_pts"] - merged["pred_pts"]
    merged["abs_error"] = merged["error"].abs()

    mae      = merged["abs_error"].mean()
    corr     = merged[["pred_pts", "actual_pts"]].corr().iloc[0, 1]
    top10    = set(merged.nlargest(10, "pred_pts")["name"])
    top20act = set(merged.nlargest(20, "actual_pts")["name"])
    hit_rate = len(top10 & top20act)

    print(f"\n📊  GW{pred_gw} ACCURACY CHECK\n")
    print(f"  MAE:         {mae:.2f} pts  |  Correlation: {corr:.2f}  |  Top-10 hit: {hit_rate}/10 in actual top 20\n")

    display = merged.nlargest(20, "pred_pts")[
        ["display_name", "position", "pred_pts", "actual_pts", "error"]
    ].copy()
    display["error"] = display["error"].map(lambda x: f"+{x:.1f}" if x >= 0 else f"{x:.1f}")
    print(display.to_string(index=False))


# ============================================================================
# INTERACTIVE CLI
# ============================================================================

_DISPLAY_COLS = ["display_name", "team", "opp", "ha", "fdr", "position", "form", "avail",
                 "exp_goals", "exp_assists", "exp_sot", "exp_kp", "exp_tkl", "exp_int",
                 "exp_saves", "exp_cs", "sleeper_pts"]


def print_menu(predictions: pd.DataFrame, df: pd.DataFrame) -> None:
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
        print("  [8] Validate last GW accuracy")
        print()

        choice = input("Choose (1-8): ").strip()

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

        elif choice == "8":
            validate_last_gw(df)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    log.info("🚀 SLEEPER FANTASY PREMIER LEAGUE PREDICTOR")
    log.info("=" * 50)

    fpl_client   = FPLDataClient()
    boot         = fpl_client.bootstrap()
    fixtures     = fpl_client.fixtures()
    df, ts       = load_current_season_data(fpl_client, boot, fixtures)
    feat         = engineer_features(df)
    predictions  = predict_next_gw(feat, ts, fixtures, boot)
    print_menu(predictions, feat)


if __name__ == "__main__":
    main()
