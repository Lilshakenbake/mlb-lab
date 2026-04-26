import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from flask import Flask, render_template, redirect, url_for, session, request, jsonify

from src import tracker, grader
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

PROJECTED_ROSTER_SCAN_LIMIT = int(os.getenv("ROSTER_SCAN_LIMIT", "6"))
PROFILE_FETCH_WORKERS = int(os.getenv("PROFILE_FETCH_WORKERS", "2"))

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "super-secret-key")
APP_PASSWORD = os.getenv("APP_PASSWORD", "mlb123")

BOARD_CACHE = {}
BOARD_CACHE_TTL_SECONDS = int(os.getenv("BOARD_CACHE_TTL", "1800"))  # 30 min default

PLAYS_CACHE = {"ts": 0, "data": [], "computing": False}
PLAYS_CACHE_TTL_SECONDS = int(os.getenv("PLAYS_CACHE_TTL", "600"))  # 10 min default
PLAYS_GAME_CONCURRENCY = int(os.getenv("PLAYS_GAME_CONCURRENCY", "2"))
PLAYS_LIMIT = int(os.getenv("PLAYS_LIMIT", "12"))
BOARD_INNER_CONCURRENCY = int(os.getenv("BOARD_INNER_CONCURRENCY", "3"))
_PLAYS_LOCK = threading.Lock()

STAT_LABELS = {
    "hits": "Hits",
    "total_bases": "Total Bases",
    "home_runs": "Home Runs",
    "rbis": "RBIs",
    "pitcher_strikeouts": "Strikeouts",
}


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
    top_home_runs = []
    top_rbis = []
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

    with ThreadPoolExecutor(max_workers=BOARD_INNER_CONCURRENCY) as pool:
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

    park_name = (weather or {}).get("park")

    def _hitter_props(hitters, opposing_pitcher_name, opposing_profile, side_score):
        for idx, (_, hitter_name, hitter_profile) in enumerate(hitters):
            try:
                pitcher_hits_allowed = (
                    opposing_profile["hits_allowed_avg"] if opposing_profile else None
                )

                props = [
                    ("hits", 0.5, hitter_profile.get("hits_avg", 0.0), top_hits),
                    ("total_bases", 1.5, hitter_profile.get("tb_avg", 0.0), top_total_bases),
                    ("home_runs", 0.5, hitter_profile.get("hr_avg", 0.0), top_home_runs),
                    ("rbis", 0.5, hitter_profile.get("rbi_avg", 0.0), top_rbis),
                ]
                for stat_type, line, base, bucket in props:
                    prop = build_hitter_prop(
                        stat_type,
                        hitter_name,
                        opposing_pitcher_name,
                        line,
                        base,
                        pitcher_hits_allowed,
                        idx,
                        weather,
                        hitter_profile=hitter_profile,
                        opp_pitcher_profile=opposing_profile,
                        park_name=park_name,
                    )
                    if prop["pick"] != "PASS":
                        bucket.append(prop)

                side_score[0] += (
                    hitter_profile.get("hits_avg", 0.0) * 0.8
                    + hitter_profile.get("tb_avg", 0.0) * 1.0
                )
            except Exception:
                continue

    away_box = [away_team_score]
    home_box = [home_team_score]
    _hitter_props(away_hitters, home_pitcher_name, home_pitcher_profile, away_box)
    _hitter_props(home_hitters, away_pitcher_name, away_pitcher_profile, home_box)
    away_team_score = away_box[0]
    home_team_score = home_box[0]

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
        "top_home_runs": sorted(top_home_runs, key=lambda x: x["probability"], reverse=True),
        "top_rbis": sorted(top_rbis, key=lambda x: x["probability"], reverse=True),
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


def _build_plays_for_game(game):
    try:
        boards = get_cached_game_boards(game)
    except Exception:
        return []

    matchup = f"{game['away_team']} @ {game['home_team']}"
    out = []

    for stat_type_key, prop_list in (
        ("hits", boards.get("top_hits", [])),
        ("total_bases", boards.get("top_total_bases", [])),
        ("home_runs", boards.get("top_home_runs", [])),
        ("rbis", boards.get("top_rbis", [])),
    ):
        for prop in prop_list:
            if prop.get("pick") == "PASS":
                continue
            out.append({
                "kind": "hitter",
                "headline": prop.get("player", ""),
                "stat_label": STAT_LABELS.get(stat_type_key, stat_type_key),
                "pick": prop.get("pick"),
                "line": prop.get("line"),
                "projection": prop.get("projection"),
                "edge": prop.get("edge"),
                "probability": prop.get("probability", 0),
                "model_used": prop.get("model_used", False),
                "matchup": matchup,
                "game_pk": game.get("gamePk"),
            })

    for prop in boards.get("top_strikeouts", []):
        if prop.get("pick") == "PASS":
            continue
        out.append({
            "kind": "pitcher",
            "headline": prop.get("pitcher", ""),
            "stat_label": "Strikeouts",
            "pick": prop.get("pick"),
            "line": prop.get("line"),
            "projection": prop.get("projection"),
            "edge": prop.get("edge"),
            "probability": prop.get("probability", 0),
            "model_used": prop.get("model_used", False),
            "matchup": matchup,
            "game_pk": game.get("gamePk"),
        })

    spread = boards.get("spread_lean") or {}
    if spread.get("ml_pick") and spread.get("ml_probability", 0) >= 55:
        out.append({
            "kind": "moneyline",
            "headline": spread["ml_pick"],
            "stat_label": "Moneyline",
            "pick": "ML",
            "line": None,
            "projection": None,
            "edge": spread.get("margin"),
            "probability": spread.get("ml_probability", 0),
            "model_used": False,
            "matchup": matchup,
            "game_pk": game.get("gamePk"),
        })
    if spread.get("run_line_pick") and spread.get("run_line_probability", 0) >= 55:
        out.append({
            "kind": "runline",
            "headline": spread["run_line_pick"],
            "stat_label": "Run Line",
            "pick": "RL",
            "line": None,
            "projection": None,
            "edge": spread.get("margin"),
            "probability": spread.get("run_line_probability", 0),
            "model_used": False,
            "matchup": matchup,
            "game_pk": game.get("gamePk"),
        })

    return out


def _refresh_plays_blocking():
    import gc
    try:
        games = get_todays_games()
    except Exception:
        with _PLAYS_LOCK:
            PLAYS_CACHE["computing"] = False
        return

    all_plays = []
    if PLAYS_GAME_CONCURRENCY <= 1:
        # Sequential mode for memory-constrained hosts. Frees pandas
        # DataFrames between games via explicit GC.
        for g in games:
            try:
                all_plays.extend(_build_plays_for_game(g))
            except Exception:
                pass
            gc.collect()
    else:
        with ThreadPoolExecutor(max_workers=PLAYS_GAME_CONCURRENCY) as pool:
            futures = [pool.submit(_build_plays_for_game, g) for g in games]
            for fut in as_completed(futures):
                try:
                    all_plays.extend(fut.result())
                except Exception:
                    continue
                gc.collect()

    all_plays.sort(key=lambda p: p.get("probability", 0), reverse=True)
    with _PLAYS_LOCK:
        PLAYS_CACHE["ts"] = time.time()
        PLAYS_CACHE["data"] = all_plays[:PLAYS_LIMIT]
        PLAYS_CACHE["computing"] = False


def get_plays_of_day_snapshot():
    """Return what we have right now, kicking off a background refresh if stale."""
    now = time.time()
    with _PLAYS_LOCK:
        fresh = PLAYS_CACHE["data"] and now - PLAYS_CACHE["ts"] < PLAYS_CACHE_TTL_SECONDS
        already_running = PLAYS_CACHE["computing"]
        data = list(PLAYS_CACHE["data"])
        ts = PLAYS_CACHE["ts"]
        if not fresh and not already_running:
            PLAYS_CACHE["computing"] = True
            t = threading.Thread(target=_refresh_plays_blocking, daemon=True)
            t.start()
            already_running = True

    return {
        "plays": data,
        "computing": already_running and not fresh,
        "ts": ts,
    }


@app.route("/", methods=["GET"])
@login_required
def home():
    games = get_todays_games()
    return render_template("index.html", games=games, best_plays=[])


@app.route("/api/plays-of-day", methods=["GET"])
@login_required
def api_plays_of_day():
    return jsonify(get_plays_of_day_snapshot())


@app.route("/watchlist", methods=["GET"])
@login_required
def watchlist():
    grade_state = grader.trigger_background_grade(force=False)
    plays = tracker.list_plays()
    stats = tracker.summary_stats()
    grade_info = {
        "running": grade_state.get("running"),
        "last_run_label": grader.humanize_age(grade_state.get("last_run_ts") or 0),
        "last_result": grade_state.get("last_result"),
        "just_started": grade_state.get("started"),
    }
    return render_template("watchlist.html", plays=plays, stats=stats, grade_info=grade_info)


@app.route("/api/grade", methods=["POST"])
@login_required
def api_grade():
    state = grader.trigger_background_grade(force=True)
    if request.form:
        return redirect(url_for("watchlist"))
    return jsonify(state)


@app.route("/api/odds/<int:play_id>", methods=["POST"])
@login_required
def api_odds(play_id):
    odds_val = request.form.get("odds") if request.form else None
    if odds_val is None and request.is_json:
        odds_val = (request.get_json(silent=True) or {}).get("odds")
    ok = tracker.update_odds(play_id, odds_val)
    if request.form:
        return redirect(url_for("watchlist"))
    return jsonify({"ok": ok})


@app.route("/api/units/<int:play_id>", methods=["POST"])
@login_required
def api_units(play_id):
    units_val = request.form.get("units") if request.form else None
    if units_val is None and request.is_json:
        units_val = (request.get_json(silent=True) or {}).get("units")
    ok = tracker.update_units(play_id, units_val)
    if request.form:
        return redirect(url_for("watchlist"))
    return jsonify({"ok": ok})


@app.route("/api/track", methods=["POST"])
@login_required
def api_track():
    payload = request.get_json(silent=True) or {}
    result = tracker.add_play(payload)
    if not result.get("ok"):
        return jsonify(result), 400
    return jsonify(result)


@app.route("/api/settle/<int:play_id>", methods=["POST"])
@login_required
def api_settle(play_id):
    result = request.form.get("result") or (request.get_json(silent=True) or {}).get("result")
    actual_value = request.form.get("actual_value") or (request.get_json(silent=True) or {}).get("actual_value")
    notes = request.form.get("notes") or (request.get_json(silent=True) or {}).get("notes")
    ok = tracker.settle_play(play_id, result, actual_value=actual_value, notes=notes)
    if request.form:
        return redirect(url_for("watchlist"))
    return jsonify({"ok": ok})


@app.route("/api/reopen/<int:play_id>", methods=["POST"])
@login_required
def api_reopen(play_id):
    ok = tracker.reopen_play(play_id)
    if request.form:
        return redirect(url_for("watchlist"))
    return jsonify({"ok": ok})


@app.route("/api/untrack/<int:play_id>", methods=["POST"])
@login_required
def api_untrack(play_id):
    ok = tracker.delete_play(play_id)
    if request.form:
        return redirect(url_for("watchlist"))
    return jsonify({"ok": ok})


@app.route("/game/<int:game_pk>")
@login_required
def game_detail(game_pk):
    games = get_todays_games()
    game = next((g for g in games if g["gamePk"] == game_pk), None)

    if not game:
        return render_template("game_detail.html", error="Game not found", game=None)

    sort_mode = request.args.get("sort", "prob")
    if sort_mode not in ("prob", "edge"):
        sort_mode = "prob"

    try:
        boards = get_cached_game_boards(game)
    except Exception:
        boards = {
            "top_hits": [],
            "top_total_bases": [],
            "top_home_runs": [],
            "top_rbis": [],
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

    if sort_mode == "edge":
        sort_key = lambda p: abs(p.get("edge", 0) or 0)
    else:
        sort_key = lambda p: p.get("probability", 0) or 0

    def _sort(items):
        return sorted(items or [], key=sort_key, reverse=True)

    return render_template(
        "game_detail.html",
        game=game,
        error=None,
        top_hits=_sort(boards["top_hits"]),
        top_home_runs=_sort(boards["top_home_runs"]),
        top_total_bases=_sort(boards["top_total_bases"]),
        top_rbis=_sort(boards["top_rbis"]),
        top_strikeouts=_sort(boards["top_strikeouts"]),
        spread_lean=boards["spread_lean"],
        weather=boards["weather"],
        lineup_confirmed=boards["lineup_confirmed"],
        projected_mode=boards["projected_mode"],
        sort_mode=sort_mode,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)