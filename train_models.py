"""
Train projection models for hitter hits/total-bases and pitcher strikeouts.

Pulls Statcast game logs via pybaseball for a configurable pool of players,
builds per-game training rows (next-game stat as target, prior 5-game rolling
averages plus park context as features), and fits GradientBoostingRegressors.

Models, feature names and per-target residual std are saved to ./models/.
The Flask app loads them via src.model and uses them when available, falling
back to the existing heuristic projections when models are missing.

Usage:
    python train_models.py                          # default: last 120 days
    python train_models.py --days 200               # longer training window
    python train_models.py --hitters "Aaron Judge,Mookie Betts"
    python train_models.py --pitchers "Gerrit Cole,Tarik Skubal"
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from pybaseball import playerid_lookup, statcast_batter, statcast_pitcher

from src.mlb_data import STADIUMS

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

DEFAULT_HITTERS = [
    "Aaron Judge", "Mookie Betts", "Juan Soto", "Freddie Freeman",
    "Bobby Witt Jr.", "Shohei Ohtani", "Jose Ramirez", "Vladimir Guerrero Jr.",
    "Kyle Tucker", "Yordan Alvarez", "Corey Seager", "Bryce Harper",
    "Manny Machado", "Trea Turner", "Marcus Semien", "Rafael Devers",
    "Pete Alonso", "Matt Olson", "Francisco Lindor", "Gunnar Henderson",
]

DEFAULT_PITCHERS = [
    "Gerrit Cole", "Zack Wheeler", "Spencer Strider", "Corbin Burnes",
    "Tarik Skubal", "Kevin Gausman", "Logan Webb", "Pablo Lopez",
    "Yoshinobu Yamamoto", "Cole Ragans", "Aaron Nola", "Zac Gallen",
    "Sonny Gray", "George Kirby", "Logan Gilbert",
]

INDOOR_PARKS = {team for team, info in STADIUMS.items() if info["indoor"]}


def _split_name(full_name: str):
    parts = full_name.strip().split()
    if len(parts) < 2:
        return None, None
    suffixes = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}
    last = parts[-1].lower()
    if last in suffixes and len(parts) >= 3:
        return parts[0], parts[-2]
    return parts[0], parts[-1]


def _lookup_id(full_name: str):
    first, last = _split_name(full_name)
    if not first:
        return None
    try:
        df = playerid_lookup(last, first)
        if df.empty:
            return None
        return int(df.iloc[0]["key_mlbam"])
    except Exception as exc:
        print(f"  lookup failed for {full_name}: {exc}", file=sys.stderr)
        return None


def _batter_game_logs(name: str, days: int) -> pd.DataFrame | None:
    pid = _lookup_id(name)
    if pid is None:
        return None
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = statcast_batter(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), pid)
    except Exception as exc:
        print(f"  statcast_batter failed for {name}: {exc}", file=sys.stderr)
        return None
    if df is None or df.empty:
        return None

    grouped = df.groupby("game_date")
    hits = grouped["events"].apply(
        lambda x: x.isin(["single", "double", "triple", "home_run"]).sum()
    )
    tb = grouped["events"].apply(
        lambda x: (x == "single").sum()
        + (x == "double").sum() * 2
        + (x == "triple").sum() * 3
        + (x == "home_run").sum() * 4
    )
    home_team = grouped["home_team"].first()

    out = pd.DataFrame({
        "game_date": pd.to_datetime(hits.index),
        "hits": hits.values.astype(float),
        "total_bases": tb.values.astype(float),
        "home_team_code": home_team.values,
    }).sort_values("game_date").reset_index(drop=True)
    out["player"] = name
    return out


def _pitcher_game_logs(name: str, days: int) -> pd.DataFrame | None:
    pid = _lookup_id(name)
    if pid is None:
        return None
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = statcast_pitcher(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), pid)
    except Exception as exc:
        print(f"  statcast_pitcher failed for {name}: {exc}", file=sys.stderr)
        return None
    if df is None or df.empty:
        return None

    grouped = df.groupby("game_date")
    ks = grouped["events"].apply(
        lambda x: x.isin(["strikeout", "strikeout_double_play"]).sum()
    )
    hits_allowed = grouped["events"].apply(
        lambda x: x.isin(["single", "double", "triple", "home_run"]).sum()
    )
    home_team = grouped["home_team"].first()

    out = pd.DataFrame({
        "game_date": pd.to_datetime(ks.index),
        "strikeouts": ks.values.astype(float),
        "hits_allowed": hits_allowed.values.astype(float),
        "home_team_code": home_team.values,
    }).sort_values("game_date").reset_index(drop=True)
    out["player"] = name
    return out


# Statcast home_team field uses 3-letter codes; map a few common ones to our
# stadium dict so the indoor flag works out.
TEAM_CODE_TO_NAME = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "CHC": "Chicago Cubs", "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians", "COL": "Colorado Rockies",
    "DET": "Detroit Tigers", "HOU": "Houston Astros", "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Oakland Athletics", "ATH": "Athletics",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates", "SD": "San Diego Padres",
    "SF": "San Francisco Giants", "SEA": "Seattle Mariners", "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays", "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays",
    "WSH": "Washington Nationals",
}


def _is_indoor(home_code: str) -> int:
    name = TEAM_CODE_TO_NAME.get(str(home_code), "")
    return 1 if name in INDOOR_PARKS else 0


def build_batter_dataset(logs: pd.DataFrame) -> pd.DataFrame:
    """Per-game rows with prior-5-game rolling averages and next-game targets."""
    rows = []
    for player, g in logs.groupby("player"):
        g = g.sort_values("game_date").reset_index(drop=True)
        if len(g) < 6:
            continue
        for i in range(5, len(g)):
            window = g.iloc[i - 5:i]
            tgt = g.iloc[i]
            rows.append({
                "player": player,
                "hits_avg_5": float(window["hits"].mean()),
                "tb_avg_5": float(window["total_bases"].mean()),
                "hits_std_5": float(window["hits"].std() or 0.0),
                "tb_std_5": float(window["total_bases"].std() or 0.0),
                "indoor": _is_indoor(tgt["home_team_code"]),
                "target_hits": float(tgt["hits"]),
                "target_tb": float(tgt["total_bases"]),
            })
    return pd.DataFrame(rows)


def build_pitcher_dataset(logs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for player, g in logs.groupby("player"):
        g = g.sort_values("game_date").reset_index(drop=True)
        if len(g) < 4:
            continue
        for i in range(3, len(g)):
            window = g.iloc[max(0, i - 5):i]
            tgt = g.iloc[i]
            rows.append({
                "player": player,
                "k_avg_5": float(window["strikeouts"].mean()),
                "k_std_5": float(window["strikeouts"].std() or 0.0),
                "hits_allowed_avg_5": float(window["hits_allowed"].mean()),
                "indoor": _is_indoor(tgt["home_team_code"]),
                "target_k": float(tgt["strikeouts"]),
            })
    return pd.DataFrame(rows)


def _fit(features: pd.DataFrame, target: pd.Series, name: str):
    if len(features) < 25:
        print(f"  {name}: not enough samples ({len(features)}) — skipping")
        return None
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    residuals = y_test.values - preds
    std = float(np.std(residuals)) or 1.0
    print(f"  {name}: n={len(features)}, MAE={mae:.3f}, residual_std={std:.3f}")
    return {
        "model": model,
        "features": list(features.columns),
        "residual_std": std,
        "mae": mae,
        "n_samples": int(len(features)),
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }


def _save(bundle, filename):
    if bundle is None:
        return
    out = MODELS_DIR / filename
    joblib.dump(bundle, out)
    print(f"  saved → {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=120, help="Lookback window in days")
    p.add_argument("--hitters", type=str, default="", help="Comma-separated hitter names")
    p.add_argument("--pitchers", type=str, default="", help="Comma-separated pitcher names")
    args = p.parse_args()

    hitters = [h.strip() for h in args.hitters.split(",") if h.strip()] or DEFAULT_HITTERS
    pitchers = [p_.strip() for p_ in args.pitchers.split(",") if p_.strip()] or DEFAULT_PITCHERS

    print(f"Pulling batter logs ({args.days}d) for {len(hitters)} hitters …")
    batter_frames = []
    for name in hitters:
        print(f"  · {name}")
        df = _batter_game_logs(name, args.days)
        if df is not None and not df.empty:
            batter_frames.append(df)

    print(f"Pulling pitcher logs ({args.days}d) for {len(pitchers)} pitchers …")
    pitcher_frames = []
    for name in pitchers:
        print(f"  · {name}")
        df = _pitcher_game_logs(name, args.days)
        if df is not None and not df.empty:
            pitcher_frames.append(df)

    if not batter_frames and not pitcher_frames:
        print("No game logs were retrieved. Aborting.")
        sys.exit(1)

    summary = {"hitters_used": [], "pitchers_used": []}

    if batter_frames:
        all_batters = pd.concat(batter_frames, ignore_index=True)
        ds = build_batter_dataset(all_batters)
        summary["hitters_used"] = sorted(ds["player"].unique().tolist())
        feat_cols = ["hits_avg_5", "tb_avg_5", "hits_std_5", "tb_std_5", "indoor"]
        X = ds[feat_cols]

        print("\nFitting hitter_hits model …")
        _save(_fit(X, ds["target_hits"], "hitter_hits"), "hitter_hits.joblib")

        print("Fitting hitter_total_bases model …")
        _save(_fit(X, ds["target_tb"], "hitter_total_bases"), "hitter_total_bases.joblib")

    if pitcher_frames:
        all_pitchers = pd.concat(pitcher_frames, ignore_index=True)
        ds = build_pitcher_dataset(all_pitchers)
        summary["pitchers_used"] = sorted(ds["player"].unique().tolist())
        feat_cols = ["k_avg_5", "k_std_5", "hits_allowed_avg_5", "indoor"]
        X = ds[feat_cols]

        print("\nFitting pitcher_strikeouts model …")
        _save(_fit(X, ds["target_k"], "pitcher_strikeouts"), "pitcher_strikeouts.joblib")

    summary["trained_at"] = datetime.utcnow().isoformat() + "Z"
    summary["lookback_days"] = args.days
    (MODELS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Summary written to {MODELS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
