"""
Sleeper Fantasy Premier League Weekly Predictor
================================================

Run this script every week to:
1. Pull live FPL API data (current season history)
2. Train LightGBM component models per position
3. Predict expected Sleeper points for next gameweek
4. Interactive CLI to explore, filter, and pick best players

Usage:
    python sleeper_predictor.py

Requirements:
    pip install requests pandas lightgbm numpy scikit-learn

Author: Sleeper Predictor for Juan (GW34 onwards)
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

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
# SLEEPER SCORING TABLE (Standard)
# ============================================================================

SLEEPER_SCORING = {
    "goals": {"FWD": 9, "MID": 9, "DEF": 10, "GK": 10},
    "assists": {"FWD": 6, "MID": 6, "DEF": 7, "GK": 7},
    "shots_on_target": 2.0,
    "key_passes": {"FWD": 2, "MID": 2, "DEF": 2, "GK": 0},
    "successful_dribbles": 1.0,
    "accurate_crosses": 1.0,
    "yellow_card": -2.0,
    "red_card": -7.0,
    "aerials_won": {"FWD": 0.5, "MID": 0.5, "DEF": 1.0, "GK": 1.0},
    "effective_clearances": {"FWD": 0, "MID": 0, "DEF": 0.25, "GK": 0.25},
    "saves": 2.0,
    "clean_sheet_60plus": {"FWD": 0, "MID": 1, "DEF": 6, "GK": 8},
    "tackles_won": 1.0,
    "interceptions": 1.0,
    "blocked_shots": 1.0,
    "goals_against": {"FWD": 0, "MID": 0, "DEF": -2, "GK": -2},
}


class FPLDataClient:
    """Fetch FPL API data with caching."""

    def __init__(self, cache_dir: str = ".fpl_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "sleeper-predictor/1.0"})

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
        """Main FPL: players, teams, current status."""
        return self._get_json(f"{FPL_API}/bootstrap-static/", "bootstrap", force=True)

    def fixtures(self) -> list:
        return self._get_json(f"{FPL_API}/fixtures/", "fixtures", force=True)

    def element_summary(self, player_id: int) -> dict:
        """Per-player history across all GWs."""
        return self._get_json(
            f"{FPL_API}/element-summary/{player_id}/", f"element_{player_id}"
        )


def load_current_season_data(client: FPLDataClient) -> pd.DataFrame:
    """Pull every player's GW history for current season."""
    boot = client.bootstrap()
    elements = pd.DataFrame(boot["elements"])
    teams = pd.DataFrame(boot["teams"])
    team_map = dict(zip(teams["id"], teams["short_name"]))

    rows = []
    for _, el in elements.iterrows():
        if el["minutes"] == 0:
            continue
        try:
            summary = client.element_summary(int(el["id"]))
        except requests.HTTPError:
            continue

        for gw in summary.get("history", []):
            rows.append(
                {
                    "name": f"{el['first_name']} {el['second_name']}",
                    "player_id": el["id"],
                    "team": team_map.get(el["team"], str(el["team"])),
                    "position": POSITION_MAP.get(el["element_type"]),
                    "GW": gw["round"],
                    "minutes": gw["minutes"],
                    "goals_scored": gw["goals_scored"],
                    "assists": gw["assists"],
                    "clean_sheets": gw["clean_sheets"],
                    "goals_conceded": gw["goals_conceded"],
                    "saves": gw["saves"],
                    "yellow_cards": gw["yellow_cards"],
                    "red_cards": gw["red_cards"],
                    "was_home": int(gw["was_home"]),
                    "expected_goals": float(gw.get("expected_goals", 0) or 0),
                    "expected_assists": float(gw.get("expected_assists", 0) or 0),
                    "influence": float(gw["influence"]),
                    "creativity": float(gw["creativity"]),
                    "threat": float(gw["threat"]),
                    "ict_index": float(gw["ict_index"]),
                    "total_points": gw["total_points"],
                }
            )

    df = pd.DataFrame(rows)
    log.info(f"✓ Loaded {len(df)} player-GW records")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build lagged rolling features (never leak future info)."""
    df = df.sort_values(["name", "GW"]).reset_index(drop=True)

    for window in [3, 5, 10]:
        for stat in [
            "goals_scored",
            "assists",
            "expected_goals",
            "expected_assists",
            "clean_sheets",
            "saves",
            "total_points",
            "influence",
            "creativity",
            "threat",
        ]:
            col = f"{stat}_avg{window}"
            df[col] = (
                df.groupby("name")[stat]
                .transform(
                    lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
                )
            )

        # Minutes reliability (critical for Sleeper)
        df[f"minutes_avg{window}"] = (
            df.groupby("name")["minutes"]
            .transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
            )
        )
        df[f"starts_rate{window}"] = (
            df.groupby("name")["minutes"]
            .transform(
                lambda s, w=window: (s.shift(1) >= 60).rolling(w, min_periods=1).mean()
            )
        )

    df["games_played"] = df.groupby("name").cumcount()
    df = df.dropna(subset=["minutes_avg5"])

    log.info(f"✓ Features engineered: {df.shape}")
    return df


def train_component_models(df: pd.DataFrame) -> dict:
    """Train one model per (position, stat) using time-series CV."""
    non_feature = {
        "name",
        "team",
        "position",
        "GW",
        "minutes",
        "player_id",
        "was_home",
    }
    feature_cols = [c for c in df.columns if c not in non_feature]

    targets = [
        "goals_scored",
        "assists",
        "expected_goals",
        "expected_assists",
        "clean_sheets",
        "saves",
        "yellow_cards",
        "red_cards",
    ]

    bundle = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        sub = df[df["position"] == pos].copy()
        if len(sub) < 50:
            continue

        bundle[pos] = {}
        log.info(f"Training {pos}...")

        for target in targets:
            if target not in sub.columns or sub[target].sum() == 0:
                continue

            X = sub[feature_cols].fillna(0)
            y = sub[target].fillna(0)

            # Quick time-series CV
            tscv = TimeSeriesSplit(n_splits=2)
            errs = []
            for tr_idx, te_idx in tscv.split(X):
                m = LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    num_leaves=31,
                    verbose=-1,
                    random_state=42,
                )
                m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                pred = m.predict(X.iloc[te_idx])
                errs.append(mean_absolute_error(y.iloc[te_idx], pred))

            # Train final model
            m = LGBMRegressor(
                n_estimators=200,
                learning_rate=0.1,
                num_leaves=31,
                verbose=-1,
                random_state=42,
            )
            m.fit(X, y)
            bundle[pos][target] = (m, feature_cols)

    return bundle, feature_cols


def predict_next_gw(df: pd.DataFrame, bundle: dict, feature_cols: list) -> pd.DataFrame:
    """Predict Sleeper points for next gameweek."""
    current_gw = df["GW"].max()
    next_gw = current_gw + 1

    # Use latest features per player
    base = df.sort_values("GW").groupby("name").tail(1).copy()
    base["GW"] = next_gw

    predictions = []
    for _, row in base.iterrows():
        pos = row["position"]
        if pos not in bundle:
            continue

        x = row[feature_cols].fillna(0).to_frame().T

        # Predict each stat
        stats = {}
        for target, (model, _) in bundle[pos].items():
            stats[target] = max(0.0, float(model.predict(x)[0]))

        exp_minutes = float(row.get("minutes_avg5", 60))

        # Sleeper scoring
        pts = 0.0
        pts += stats.get("goals_scored", 0) * SLEEPER_SCORING["goals"].get(pos, 0)
        pts += stats.get("assists", 0) * SLEEPER_SCORING["assists"].get(pos, 0)
        pts += stats.get("yellow_cards", 0) * SLEEPER_SCORING["yellow_card"]
        pts += stats.get("red_cards", 0) * SLEEPER_SCORING["red_card"]

        # Clean sheet (gates on minutes)
        prob_cs = stats.get("clean_sheets", 0) * min(1.0, exp_minutes / 60)
        pts += prob_cs * SLEEPER_SCORING["clean_sheet_60plus"].get(pos, 0)

        # Saves (mostly GK)
        pts += stats.get("saves", 0) * SLEEPER_SCORING["saves"]

        # Goals against (DEF/GK penalty)
        pts += stats.get("goals_conceded", 0) * SLEEPER_SCORING["goals_against"].get(
            pos, 0
        )

        # Expected goal involvement as proxy for advanced attacking potential
        pts += stats.get("expected_goals", 0) * 2.5  # bonus for underlying threat
        pts += stats.get("expected_assists", 0) * 2.5

        predictions.append(
            {
                "name": row["name"],
                "team": row["team"],
                "position": pos,
                "GW": next_gw,
                "exp_minutes": round(exp_minutes, 1),
                "exp_goals": round(stats.get("goals_scored", 0), 2),
                "exp_assists": round(stats.get("assists", 0), 2),
                "exp_saves": round(stats.get("saves", 0), 2),
                "exp_cs": round(prob_cs, 2),
                "sleeper_pts": round(pts, 2),
            }
        )

    result = pd.DataFrame(predictions)
    result = result.sort_values("sleeper_pts", ascending=False)
    log.info(f"✓ Predicted {len(result)} players for GW{next_gw}")
    return result


def print_menu(predictions: pd.DataFrame) -> None:
    """Interactive CLI to explore predictions."""
    while True:
        print("\n" + "=" * 80)
        print(f"⚽ SLEEPER FANTASY GW{predictions['GW'].iloc[0]} PREDICTOR")
        print("=" * 80)
        print("\nOptions:")
        print("  [1] Top 20 all positions")
        print("  [2] Top 10 by position (GK, DEF, MID, FWD)")
        print("  [3] Filter by team")
        print("  [4] Filter by position + min expected points")
        print("  [5] Export to CSV")
        print("  [6] Exit")
        print()

        choice = input("Choose (1-6): ").strip()

        if choice == "1":
            print("\n🏆 TOP 20 SLEEPER SCORERS\n")
            print(
                predictions.head(20)[
                    ["name", "team", "position", "exp_minutes", "sleeper_pts"]
                ]
                .to_string(index=False)
            )

        elif choice == "2":
            print()
            for pos in ["GK", "DEF", "MID", "FWD"]:
                sub = predictions[predictions["position"] == pos].head(10)
                if len(sub) > 0:
                    print(f"\n{pos}:")
                    print(
                        sub[
                            ["name", "team", "exp_minutes", "sleeper_pts"]
                        ].to_string(index=False)
                    )

        elif choice == "3":
            team = input("Team (e.g., ARS, MUN, LIV): ").upper()
            sub = predictions[predictions["team"] == team].head(20)
            if len(sub) > 0:
                print(f"\n{team} PLAYERS:\n")
                print(
                    sub[["name", "position", "exp_minutes", "sleeper_pts"]].to_string(
                        index=False
                    )
                )
            else:
                print(f"No predictions for {team}")

        elif choice == "4":
            pos = input("Position (GK/DEF/MID/FWD): ").upper()
            min_pts = float(input("Minimum expected points: "))
            sub = predictions[
                (predictions["position"] == pos) & (predictions["sleeper_pts"] >= min_pts)
            ]
            if len(sub) > 0:
                print(f"\n{pos} PLAYERS >= {min_pts} PTS:\n")
                print(
                    sub[["name", "team", "exp_minutes", "sleeper_pts"]].to_string(
                        index=False
                    )
                )
            else:
                print(f"No {pos} found with {min_pts}+ pts")

        elif choice == "5":
            fname = f"sleeper_gw{predictions['GW'].iloc[0]}.csv"
            predictions.to_csv(fname, index=False)
            log.info(f"✓ Saved to {fname}")

        elif choice == "6":
            print("\nGoodbye! ⚽")
            break


def main():
    log.info("🚀 SLEEPER FANTASY PREMIER LEAGUE PREDICTOR")
    log.info("=" * 50)

    client = FPLDataClient()
    df = load_current_season_data(client)
    feat = engineer_features(df)
    bundle, feature_cols = train_component_models(feat)
    predictions = predict_next_gw(feat, bundle, feature_cols)

    print_menu(predictions)


if __name__ == "__main__":
    main()
