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
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMRegressor, early_stopping as lgb_early_stopping
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
    # FPL API available
    "own_goals":            -5.0,
    "penalties_missed":     -4.0,
    "penalties_saved":       8.0,
}

# Stats the model is trained to predict
TRAIN_TARGETS = [
    "goals_scored", "assists", "expected_goals", "expected_assists",
    "clean_sheets", "saves", "yellow_cards", "red_cards",
    "goals_conceded", "own_goals", "penalties_missed", "penalties_saved",
]

NON_FEATURE = {"name", "team", "position", "GW", "minutes", "player_id",
               "status", "chance_of_playing"}

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

            # Use last time-series split to find best n_estimators via early stopping
            tscv  = TimeSeriesSplit(n_splits=3)
            tr, te = list(tscv.split(X))[-1]

            params = dict(
                n_estimators=2000,
                learning_rate=0.02,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=0.5,
                verbose=-1,
                random_state=42,
            )
            m_cv = LGBMRegressor(**params)
            m_cv.fit(
                X.iloc[tr], y.iloc[tr],
                eval_set=[(X.iloc[te], y.iloc[te])],
                callbacks=[lgb_early_stopping(50, verbose=False)],
            )
            best_n = m_cv.best_iteration_ or 300

            # Final model trained on all data with tuned n_estimators
            m = LGBMRegressor(**{**params, "n_estimators": best_n})
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
        base.at[idx, "opp"]          = opp or "TBC"
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

        exp_min   = float(row.get("minutes_avg5", 60))
        min_scale = min(1.0, exp_min / 90)  # scale per-event stats by expected minutes
        prob_cs   = s.get("clean_sheets", 0) * min(1.0, exp_min / 60)

        def sc(stat: str) -> float:
            return s.get(stat, 0.0) * min_scale

        # Full Sleeper point formula — every scoring category
        pts = (
            sc("goals_scored")        * _pos_score("goals", pos)
          + sc("assists")             * _pos_score("assists", pos)
          + sc("saves")               * SLEEPER_SCORING["saves"]
          + sc("goals_conceded")      * _pos_score("goals_against", pos)
          + sc("own_goals")           * SLEEPER_SCORING["own_goals"]
          + sc("penalties_missed")    * SLEEPER_SCORING["penalties_missed"]
          + sc("penalties_saved")     * SLEEPER_SCORING["penalties_saved"]
          + sc("yellow_cards")        * SLEEPER_SCORING["yellow_card"]
          + sc("red_cards")           * SLEEPER_SCORING["red_card"]
          + prob_cs                   * _pos_score("clean_sheet_60plus", pos)
        ) * avail_mult

        rows.append({
            "name":         row["name"],
            "team":         row["team"],
            "opp":          row.get("opp", "TBC"),
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

_DISPLAY_COLS = ["name", "team", "opp", "position", "avail", "exp_min",
                 "exp_goals", "exp_assists", "exp_sot", "exp_kp", "exp_tkl", "exp_int",
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
    boot         = fpl_client.bootstrap()
    fixtures     = fpl_client.fixtures()
    df, ts       = load_current_season_data(fpl_client, boot, fixtures)
    feat         = engineer_features(df)
    bundle, feature_cols = train_component_models(feat)
    predictions  = predict_next_gw(feat, bundle, feature_cols, ts, fixtures, boot)
    print_menu(predictions)


if __name__ == "__main__":
    main()
