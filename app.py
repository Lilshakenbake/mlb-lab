import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from flask import Flask, render_template, redirect, url_for, session, request

from src.mlb_data import (
    get_todays_games,
    get_team_active_hitters,
    get_last5_hitter_profile,
    get_last5_pitcher_profile,
    get_game_weather,
)

from src.predict import (
    build_hitter_prop,
    build_pitcher_k_prop,
    build_spread_lean,
)

PROJECTED_ROSTER_SCAN_LIMIT = 10
PROFILE_FETCH_WORKERS = 8

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "super-secret-key")
APP_PASSWORD = os.getenv("APP_PASSWORD", "mlb123")

BOARD_CACHE = {}
BOARD_CACHE_TTL_SECONDS = 1800  # 30 minutes


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        password = request.form.get("password", "").strip()
        if password == APP_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("home"))
        error = "Wrong password"

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


def _fetch_hitter_profile_safe(name):
    try:
        profile, err = get_last5_hitter_profile(name)
        if err or not profile:
            return name, None
        return name, profile
    except Exception:
        return name, None


def get_projected_top_hitters(team_id, limit=3):
    hitters = get_team_active_hitters(team_id)[:PROJECTED_ROSTER_SCAN_LIMIT]
    if not hitters:
        return []

    scored = []
    with ThreadPoolExecutor(max_workers=PROFILE_FETCH_WORKERS) as pool:
        futures = [pool.submit(_fetch_hitter_profile_safe, name) for name in hitters]
        for fut in as_completed(futures):
            name, profile = fut.result()
            if not profile or profile.get("games_used", 0) < 3:
                continue
            score = profile["hits_avg"] * 0.9 + profile["tb_avg"] * 1.1
            scored.append((score, name, profile))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:limit]


def build_game_boards(game):
    top_hits = []
    top_total_bases = []
    top_strikeouts = []

    away_pitcher_name = game.get("away_pitcher", "").strip()
    home_pitcher_name = game.get("home_pitcher", "").strip()

    # Run weather, both team rosters+profiles, and both pitcher lookups
    # concurrently. Each of these used to block sequentially on slow
    # Baseball Savant scrapes.
    def _safe_pitcher(name):
        if not name or name == "TBD":
            return None
        try:
            profile, _ = get_last5_pitcher_profile(name)
            return profile
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=5) as pool:
        f_weather = pool.submit(get_game_weather, game)
        f_away_hitters = pool.submit(get_projected_top_hitters, game["away_team_id"], 3)
        f_home_hitters = pool.submit(get_projected_top_hitters, game["home_team_id"], 3)
        f_away_pitcher = pool.submit(_safe_pitcher, away_pitcher_name)
        f_home_pitcher = pool.submit(_safe_pitcher, home_pitcher_name)

        weather = f_weather.result()
        away_hitters = f_away_hitters.result()
        home_hitters = f_home_hitters.result()
        away_pitcher_profile = f_away_pitcher.result()
        home_pitcher_profile = f_home_pitcher.result()

    lineup_confirmed = False
    projected_mode = True

    away_team_score = 0.0
    home_team_score = 0.0

    if away_pitcher_profile and "strikeouts_avg" in away_pitcher_profile:
        top_strikeouts.append(
            build_pitcher_k_prop(
                pitcher_name=away_pitcher_name,
                line=5.5,
                projection=away_pitcher_profile["strikeouts_avg"],
                weather=weather,
                pitcher_profile=away_pitcher_profile,
            )
        )

    if home_pitcher_profile and "strikeouts_avg" in home_pitcher_profile:
        top_strikeouts.append(
            build_pitcher_k_prop(
                pitcher_name=home_pitcher_name,
                line=5.5,
                projection=home_pitcher_profile["strikeouts_avg"],
                weather=weather,
                pitcher_profile=home_pitcher_profile,
            )
        )

    for idx, (_, hitter_name, hitter_profile) in enumerate(away_hitters):
        try:
            pitcher_hits_allowed = home_pitcher_profile["hits_allowed_avg"] if home_pitcher_profile else None

            hit = build_hitter_prop(
                "hits",
                hitter_name,
                home_pitcher_name,
                0.5,
                hitter_profile["hits_avg"],
                pitcher_hits_allowed,
                idx,
                weather,
                hitter_profile=hitter_profile,
            )
            tb = build_hitter_prop(
                "total_bases",
                hitter_name,
                home_pitcher_name,
                1.5,
                hitter_profile["tb_avg"],
                pitcher_hits_allowed,
                idx,
                weather,
                hitter_profile=hitter_profile,
            )

            if hit["pick"] != "PASS":
                top_hits.append(hit)
            if tb["pick"] != "PASS":
                top_total_bases.append(tb)

            away_team_score += (
                hitter_profile["hits_avg"] * 0.8
                + hitter_profile["tb_avg"] * 1.0
            )
        except Exception:
            continue

    for idx, (_, hitter_name, hitter_profile) in enumerate(home_hitters):
        try:
            pitcher_hits_allowed = away_pitcher_profile["hits_allowed_avg"] if away_pitcher_profile else None

            hit = build_hitter_prop(
                "hits",
                hitter_name,
                away_pitcher_name,
                0.5,
                hitter_profile["hits_avg"],
                pitcher_hits_allowed,
                idx,
                weather,
                hitter_profile=hitter_profile,
            )
            tb = build_hitter_prop(
                "total_bases",
                hitter_name,
                away_pitcher_name,
                1.5,
                hitter_profile["tb_avg"],
                pitcher_hits_allowed,
                idx,
                weather,
                hitter_profile=hitter_profile,
            )

            if hit["pick"] != "PASS":
                top_hits.append(hit)
            if tb["pick"] != "PASS":
                top_total_bases.append(tb)

            home_team_score += (
                hitter_profile["hits_avg"] * 0.8
                + hitter_profile["tb_avg"] * 1.0
            )
        except Exception:
            continue

    spread_lean = build_spread_lean(
        game=game,
        home_team_score=home_team_score,
        away_team_score=away_team_score,
        home_pitcher_profile=home_pitcher_profile,
        away_pitcher_profile=away_pitcher_profile,
        weather=weather,
    )

    return {
        "top_hits": sorted(top_hits, key=lambda x: x["probability"], reverse=True),
        "top_total_bases": sorted(top_total_bases, key=lambda x: x["probability"], reverse=True),
        "top_strikeouts": sorted(top_strikeouts, key=lambda x: x["probability"], reverse=True),
        "spread_lean": spread_lean,
        "weather": weather,
        "lineup_confirmed": lineup_confirmed,
        "projected_mode": projected_mode,
    }


def get_cached_game_boards(game):
    game_pk = game["gamePk"]
    now = time.time()

    cached = BOARD_CACHE.get(game_pk)
    if cached and now - cached["ts"] < BOARD_CACHE_TTL_SECONDS:
        return cached["data"]

    data = build_game_boards(game)
    BOARD_CACHE[game_pk] = {"ts": now, "data": data}
    return data


@app.route("/", methods=["GET"])
@login_required
def home():
    games = get_todays_games()
    return render_template("index.html", games=games, best_plays=[])


@app.route("/game/<int:game_pk>")
@login_required
def game_detail(game_pk):
    games = get_todays_games()
    game = next((g for g in games if g["gamePk"] == game_pk), None)

    if not game:
        return render_template("game_detail.html", error="Game not found", game=None)

    try:
        boards = get_cached_game_boards(game)
    except Exception:
        boards = {
            "top_hits": [],
            "top_total_bases": [],
            "top_strikeouts": [],
            "spread_lean": {
                "ml_pick": "No side available",
                "ml_probability": 0,
                "run_line_pick": "No run line available",
                "run_line_probability": 0,
                "margin": 0,
                "confidence": "LOW",
                "note": "Board build failed for this request",
            },
            "weather": get_game_weather(game),
            "lineup_confirmed": False,
            "projected_mode": True,
        }

    return render_template(
        "game_detail.html",
        game=game,
        error=None,
        top_hits=boards["top_hits"],
        top_home_runs=[],
        top_total_bases=boards["top_total_bases"],
        top_rbis=[],
        top_strikeouts=boards["top_strikeouts"],
        spread_lean=boards["spread_lean"],
        weather=boards["weather"],
        lineup_confirmed=boards["lineup_confirmed"],
        projected_mode=boards["projected_mode"],
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)