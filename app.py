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
    get_confirmed_starting_hitters,
    get_last5_hitter_profile,
    get_last5_pitcher_profile,
    get_game_weather,
)

from src.predict import (
    build_hitter_prop,
    build_pitcher_k_prop,
    build_spread_lean,
    compute_hr_threat,
    compute_nrfi,
    build_hrr_combo,
)
from src import ai_review

PROJECTED_ROSTER_SCAN_LIMIT = int(os.getenv("ROSTER_SCAN_LIMIT", "16"))
PROFILE_FETCH_WORKERS = int(os.getenv("PROFILE_FETCH_WORKERS", "2"))
HITTERS_PER_TEAM = int(os.getenv("HITTERS_PER_TEAM", "5"))

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "super-secret-key")
APP_PASSWORD = os.getenv("APP_PASSWORD", "mlb123")

BOARD_CACHE = {}
BOARD_CACHE_TTL_SECONDS = int(os.getenv("BOARD_CACHE_TTL", "1800"))  # 30 min default

PLAYS_CACHE = {"ts": 0, "data": [], "computing": False}
PLAYS_CACHE_TTL_SECONDS = int(os.getenv("PLAYS_CACHE_TTL", "600"))  # 10 min default
PLAYS_GAME_CONCURRENCY = int(os.getenv("PLAYS_GAME_CONCURRENCY", "2"))
PLAYS_LIMIT = int(os.getenv("PLAYS_LIMIT", "20"))
PLAYS_PER_GAME_CAP = int(os.getenv("PLAYS_PER_GAME_CAP", "3"))
BOARD_INNER_CONCURRENCY = int(os.getenv("BOARD_INNER_CONCURRENCY", "3"))
_PLAYS_LOCK = threading.Lock()

# HR Threats Board — separate from Plays of the Day. Ranks tonight's hitters
# purely by probability of 1+ HR (not by line edge), since HR lines are almost
# always 0.5 and the standard edge logic biases toward UNDER.
HR_THREATS_CACHE = {"ts": 0, "data": []}
NRFI_CACHE = {"ts": 0, "data": []}
SPECIALS_CACHE = {"ts": 0, "data": {
    "run_line": None,
    "sgp": None,
    "cross_parlay": None,
    "hr_pair": None,
    "bases_parlay": None,
}}
HR_THREATS_LIMIT = int(os.getenv("HR_THREATS_LIMIT", "12"))
HRR_COMBO_CACHE = {"ts": 0, "data": []}
HRR_COMBO_LIMIT = int(os.getenv("HRR_COMBO_LIMIT", "12"))
# Locks: top N highest-probability single plays across the WHOLE slate,
# uncapped by per-game diversification. Used by the hero strip.
LOCKS_CACHE = {"ts": 0, "data": []}
LOCKS_LIMIT = int(os.getenv("LOCKS_LIMIT", "3"))

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


def get_projected_top_hitters(team_id, limit=HITTERS_PER_TEAM, game_pk=None, side=None):
    """Resolve which hitters from this team to grade.

    Priority:
    1. Confirmed starting lineup from MLB (if game_pk + side given) — gives us
       all 9 actual starters in batting order. This is what was missing before:
       even when MLB posted lineups, we ignored them and guessed off the roster.
    2. Fall back to scanning the active roster and ranking by recent production.

    `limit` caps how many hitters we return per team (default 5 — bumped from 3
    so stars buried in the order still show up). When confirmed lineups are
    available we return ALL nine starters so the dashboard mirrors the real
    batting order; ranking happens in the prop-builders downstream.
    """
    # ── Path 1: confirmed lineup ────────────────────────────────────────
    confirmed = []
    if game_pk and side:
        try:
            confirmed = get_confirmed_starting_hitters(game_pk, side) or []
        except Exception:
            confirmed = []

    if confirmed:
        scored = []
        with ThreadPoolExecutor(max_workers=PROFILE_FETCH_WORKERS) as pool:
            futures = [pool.submit(_fetch_hitter_profile_safe, name) for name in confirmed]
            results = {}
            for fut in as_completed(futures):
                name, profile = fut.result()
                results[name] = profile
        # Preserve batting-order sequence (MLB lineup ordering); skip blanks.
        for name in confirmed:
            profile = results.get(name)
            if not profile or profile.get("games_used", 0) < 2:
                continue
            score = profile.get("hits_avg", 0) * 0.9 + profile.get("tb_avg", 0) * 1.1
            scored.append((score, name, profile))
        if scored:
            return scored  # all confirmed starters with usable profiles

    # ── Path 2: active-roster fallback ──────────────────────────────────
    hitters = get_team_active_hitters(team_id)[:PROJECTED_ROSTER_SCAN_LIMIT]
    if not hitters:
        return []

    scored = []
    with ThreadPoolExecutor(max_workers=PROFILE_FETCH_WORKERS) as pool:
        futures = [pool.submit(_fetch_hitter_profile_safe, name) for name in hitters]
        for fut in as_completed(futures):
            name, profile = fut.result()
            # Loosen min-games filter from 3 → 2 so call-ups & recent activations
            # (which used to silently disappear) are eligible.
            if not profile or profile.get("games_used", 0) < 2:
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

    game_pk_lookup = game.get("gamePk")

    with ThreadPoolExecutor(max_workers=BOARD_INNER_CONCURRENCY) as pool:
        f_weather = pool.submit(get_game_weather, game)
        f_away_hitters = pool.submit(
            get_projected_top_hitters, game["away_team_id"], HITTERS_PER_TEAM,
            game_pk_lookup, "away",
        )
        f_home_hitters = pool.submit(
            get_projected_top_hitters, game["home_team_id"], HITTERS_PER_TEAM,
            game_pk_lookup, "home",
        )
        f_away_pitcher = pool.submit(_safe_pitcher, away_pitcher_name)
        f_home_pitcher = pool.submit(_safe_pitcher, home_pitcher_name)

        weather = f_weather.result()
        away_hitters = f_away_hitters.result()
        home_hitters = f_home_hitters.result()
        away_pitcher_profile = f_away_pitcher.result()
        home_pitcher_profile = f_home_pitcher.result()

    # If we got >= 5 hitters per side, MLB lineups are almost certainly posted
    # (vs roster fallback which we cap at HITTERS_PER_TEAM).
    lineup_confirmed = (len(away_hitters) >= 5 and len(home_hitters) >= 5)
    projected_mode = not lineup_confirmed

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

    def _hitter_props(hitters, opposing_pitcher_name, opposing_profile, side_score, opp_team_name=None):
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
                        opp_team=opp_team_name,
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
    # Pass opposing team name so bullpen factor can apply to late-inning ABs.
    _hitter_props(away_hitters, home_pitcher_name, home_pitcher_profile, away_box, game.get("home_team"))
    _hitter_props(home_hitters, away_pitcher_name, away_pitcher_profile, home_box, game.get("away_team"))
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

    # HR Threats — pure probability ranking, independent of line edge.
    matchup_label = f"{game['away_team']} @ {game['home_team']}"
    hr_threats = []
    for idx, (_, name, profile) in enumerate(away_hitters):
        t = compute_hr_threat(
            name, profile, home_pitcher_name, home_pitcher_profile,
            idx, weather, park_name,
        )
        if t:
            t["matchup"] = matchup_label
            t["game_pk"] = game.get("gamePk")
            hr_threats.append(t)
    for idx, (_, name, profile) in enumerate(home_hitters):
        t = compute_hr_threat(
            name, profile, away_pitcher_name, away_pitcher_profile,
            idx, weather, park_name,
        )
        if t:
            t["matchup"] = matchup_label
            t["game_pk"] = game.get("gamePk")
            hr_threats.append(t)

    nrfi = compute_nrfi(
        game, home_pitcher_profile, away_pitcher_profile, weather, park_name=park_name,
    )
    if nrfi:
        nrfi["matchup"] = matchup_label
        nrfi["game_pk"] = game.get("gamePk")
        nrfi["home_pitcher"] = home_pitcher_name
        nrfi["away_pitcher"] = away_pitcher_name

    # ── 1+ H/R/RBI prop board: one row per hitter with combined hits +
    # runs + RBIs >= 1 probability (the "any of three" high-floor prop).
    hrr_combo = []
    for idx, (_, name, profile) in enumerate(away_hitters):
        c = build_hrr_combo(
            name, profile, home_pitcher_profile, idx, weather,
            park_name=park_name, opp_team=game.get("home_team"),
        )
        if c and c["pick"] != "PASS":
            c["matchup"] = matchup_label
            c["game_pk"] = game.get("gamePk")
            c["pitcher"] = home_pitcher_name
            c["lineup_spot"] = idx + 1
            hrr_combo.append(c)
    for idx, (_, name, profile) in enumerate(home_hitters):
        c = build_hrr_combo(
            name, profile, away_pitcher_profile, idx, weather,
            park_name=park_name, opp_team=game.get("away_team"),
        )
        if c and c["pick"] != "PASS":
            c["matchup"] = matchup_label
            c["game_pk"] = game.get("gamePk")
            c["pitcher"] = away_pitcher_name
            c["lineup_spot"] = idx + 1
            hrr_combo.append(c)

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
        "hr_threats": hr_threats,
        "nrfi": nrfi,
        "hrr_combo": sorted(hrr_combo, key=lambda x: x["probability"], reverse=True),
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

    # 1+ H/R/RBI prop — flows through the same play pipeline as other hitter plays.
    for prop in boards.get("hrr_combo", []):
        if prop.get("pick") == "PASS":
            continue
        out.append({
            "kind": "hitter_combo",
            "headline": prop.get("player", ""),
            "stat_label": "1+ H/R/RBI",
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


def _prob_to_american(p):
    """Convert decimal probability to fair American odds string."""
    if not p or p <= 0 or p >= 1:
        return "—"
    if p >= 0.5:
        return f"-{round(p / (1 - p) * 100)}"
    return f"+{round((1 - p) / p * 100)}"


def _parlay_payout_units(probs, leg_odds=-110, stake_units=1.0):
    """Compute parlay payout in units for legs at given American odds.

    Each -110 leg = 1.909x decimal; multiply legs and subtract stake."""
    if leg_odds < 0:
        leg_decimal = 1 + (100 / abs(leg_odds))
    else:
        leg_decimal = 1 + (leg_odds / 100)
    parlay_decimal = leg_decimal ** len(probs)
    profit = (parlay_decimal - 1) * stake_units
    combined_prob = 1.0
    for p in probs:
        combined_prob *= (p / 100.0 if p > 1 else p)
    return round(profit, 2), round(combined_prob * 100, 1)


def _build_specials(sorted_plays, sorted_hr):
    """Compute the daily specials from already-collected data:
       1. Best Run Line of the Day
       2. Best 3-Leg Same-Game Parlay
       3. Best 3-Leg Cross-Game Parlay (any players, any stat)
       4. Best HR Pair (2-leg)"""
    specials = {
        "run_line": None,
        "sgp": None,
        "cross_parlay": None,
        "hr_pair": None,
        "bases_parlay": None,
    }

    # ── 1. Best Run Line ──────────────────────────────────────────────────
    run_lines = [p for p in sorted_plays if p.get("kind") == "runline"]
    if run_lines:
        best_rl = run_lines[0]  # already sorted by probability
        specials["run_line"] = {
            "headline": best_rl.get("headline"),
            "matchup": best_rl.get("matchup"),
            "probability": best_rl.get("probability"),
            "edge": best_rl.get("edge"),
            "fair_odds": _prob_to_american((best_rl.get("probability") or 0) / 100.0),
            "game_pk": best_rl.get("game_pk"),
        }

    # ── 2. Best 3-Leg SGP ─────────────────────────────────────────────────
    # Group hitter/pitcher props by game; pick the game whose top-3 plays
    # have the highest combined probability. Skip games with <3 quality plays.
    by_game = {}
    for p in sorted_plays:
        if p.get("kind") in ("moneyline", "runline"):
            continue  # SGP is props-only — cleaner correlation story
        if (p.get("probability") or 0) < 60:
            continue  # min quality bar per leg
        gpk = p.get("game_pk")
        if gpk is None:
            continue
        by_game.setdefault(gpk, []).append(p)

    best_sgp_score = 0
    for gpk, plays in by_game.items():
        if len(plays) < 3:
            continue
        # Pick top 3 by probability — and ensure they're DIFFERENT players when possible.
        seen_players = set()
        chosen = []
        for p in plays:
            player = p.get("headline")
            if player in seen_players:
                continue
            seen_players.add(player)
            chosen.append(p)
            if len(chosen) == 3:
                break
        if len(chosen) < 3:
            continue
        probs = [(p.get("probability") or 0) / 100.0 for p in chosen]
        combined = probs[0] * probs[1] * probs[2]
        # Score = combined prob × small bonus per "premium" leg (>=70%)
        score = combined * (1 + 0.05 * sum(1 for p in probs if p >= 0.70))
        if score > best_sgp_score:
            best_sgp_score = score
            payout, combined_pct = _parlay_payout_units(probs)
            specials["sgp"] = {
                "matchup": chosen[0].get("matchup"),
                "game_pk": gpk,
                "legs": [
                    {
                        "player": p.get("headline"),
                        "stat": p.get("stat_label"),
                        "pick": f'{p.get("pick")} {p.get("line")}',
                        "probability": p.get("probability"),
                    }
                    for p in chosen
                ],
                "combined_probability": combined_pct,
                "fair_odds": _prob_to_american(combined),
                "payout_units": payout,  # profit on 1u stake at -110 each leg
            }

    # ── 3. Best 3-Leg CROSS-Game Parlay ───────────────────────────────────
    # Pick the 3 highest-probability props from THREE DIFFERENT games. No
    # same-game correlation requirement — pure best-available across the
    # whole slate. Keeps player diversity too.
    cross_legs = []
    cross_games_used = set()
    cross_players_used = set()
    for p in sorted_plays:
        if p.get("kind") in ("moneyline", "runline"):
            continue
        if (p.get("probability") or 0) < 60:
            continue
        gpk = p.get("game_pk")
        player = p.get("headline")
        if gpk in cross_games_used:
            continue  # different game requirement
        if player in cross_players_used:
            continue
        cross_legs.append(p)
        cross_games_used.add(gpk)
        cross_players_used.add(player)
        if len(cross_legs) == 3:
            break

    if len(cross_legs) == 3:
        probs = [(p.get("probability") or 0) / 100.0 for p in cross_legs]
        combined = probs[0] * probs[1] * probs[2]
        payout, combined_pct = _parlay_payout_units(probs)
        specials["cross_parlay"] = {
            "legs": [
                {
                    "player": p.get("headline"),
                    "stat": p.get("stat_label"),
                    "pick": f'{p.get("pick")} {p.get("line")}',
                    "matchup": p.get("matchup"),
                    "probability": p.get("probability"),
                    "game_pk": p.get("game_pk"),
                }
                for p in cross_legs
            ],
            "combined_probability": combined_pct,
            "fair_odds": _prob_to_american(combined),
            "payout_units": payout,
        }

    # ── 4. Best Bases Parlay (3 legs, OVER total-bases only) ────────────
    tb_legs = []
    tb_games_used = set()
    tb_players_used = set()
    for p in sorted_plays:
        if p.get("stat_label") != "Total Bases":
            continue
        if p.get("pick") != "OVER":  # OVER-only — UNDER TB defeats the high-floor intent
            continue
        if (p.get("probability") or 0) < 60:
            continue
        gpk = p.get("game_pk")
        player = p.get("headline")
        if gpk in tb_games_used or player in tb_players_used:
            continue
        tb_legs.append(p)
        tb_games_used.add(gpk)
        tb_players_used.add(player)
        if len(tb_legs) == 3:
            break

    if len(tb_legs) == 3:
        probs = [(p.get("probability") or 0) / 100.0 for p in tb_legs]
        combined = probs[0] * probs[1] * probs[2]
        payout, combined_pct = _parlay_payout_units(probs)
        specials["bases_parlay"] = {
            "legs": [
                {
                    "player": p.get("headline"),
                    "stat": p.get("stat_label"),
                    "pick": f'{p.get("pick")} {p.get("line")}',
                    "matchup": p.get("matchup"),
                    "probability": p.get("probability"),
                    "game_pk": p.get("game_pk"),
                }
                for p in tb_legs
            ],
            "combined_probability": combined_pct,
            "fair_odds": _prob_to_american(combined),
            "payout_units": payout,
        }

    # ── 5. Best HR Pair ───────────────────────────────────────────────────
    if len(sorted_hr) >= 2:
        # Top two HR threats — could be same or different games.
        h1, h2 = sorted_hr[0], sorted_hr[1]
        p1, p2 = (h1.get("probability") or 0) / 100.0, (h2.get("probability") or 0) / 100.0
        combined = p1 * p2
        # HR props are typically +400 each (≈20% implied) — much juicier.
        avg_fair_odds = (h1.get("fair_odds") or 400)
        try:
            leg_odds = int(avg_fair_odds) if isinstance(avg_fair_odds, (int, float)) else 400
        except (ValueError, TypeError):
            leg_odds = 400
        payout, combined_pct = _parlay_payout_units([p1, p2], leg_odds=leg_odds)
        specials["hr_pair"] = {
            "leg_a": {"player": h1.get("player"), "vs": h1.get("vs"), "park": h1.get("park"), "probability": h1.get("probability"), "fair_odds": h1.get("fair_odds")},
            "leg_b": {"player": h2.get("player"), "vs": h2.get("vs"), "park": h2.get("park"), "probability": h2.get("probability"), "fair_odds": h2.get("fair_odds")},
            "combined_probability": combined_pct,
            "fair_odds": _prob_to_american(combined),
            "payout_units": payout,
        }

    return specials


def _run_ai_review_pass():
    """Batch-review the top locks, top HRR combos, and top HR threats with
    GPT, then atomically swap the cache lists with reviewed copies.

    Single API call per refresh keeps cost low. We deep-copy each pick before
    attaching the AI review so that API handlers (which copy the list under
    the lock and then JSON-encode outside the lock) never observe a dict
    being mutated mid-serialization.
    """
    if not ai_review.is_enabled():
        return
    import copy
    with _PLAYS_LOCK:
        locks_snapshot = [copy.deepcopy(p) for p in LOCKS_CACHE["data"]]
        hrr_snapshot = [copy.deepcopy(c) for c in HRR_COMBO_CACHE["data"]]
        hr_snapshot = [copy.deepcopy(h) for h in HR_THREATS_CACHE["data"]]
        plays_snapshot = [copy.deepcopy(p) for p in PLAYS_CACHE["data"]]
    # Tag categorical labels so _pick_key stays unique across boards and
    # the AI prompt clearly states what each pick is for.
    for h in hr_snapshot:
        h.setdefault("stat_label", "Home Run")
        h.setdefault("pick", "OVER")
    for c in hrr_snapshot:
        c.setdefault("stat_label", "1+ H/R/RBI")
        c.setdefault("pick", "OVER")
    bundle = locks_snapshot + hrr_snapshot + hr_snapshot + plays_snapshot
    if not bundle:
        return
    reviews = ai_review.review_picks(bundle, kind="mlb-slate-all")
    if not reviews:
        return
    ai_review.attach_reviews(locks_snapshot, reviews)
    ai_review.attach_reviews(hrr_snapshot, reviews)
    ai_review.attach_reviews(hr_snapshot, reviews)
    ai_review.attach_reviews(plays_snapshot, reviews)
    # Atomic publish — replace the list refs entirely so a reader holding
    # the old list never sees a dict get an `ai_review` key bolted on after
    # the fact.
    with _PLAYS_LOCK:
        LOCKS_CACHE["data"] = locks_snapshot
        HRR_COMBO_CACHE["data"] = hrr_snapshot
        HR_THREATS_CACHE["data"] = hr_snapshot
        PLAYS_CACHE["data"] = plays_snapshot
    print(f"[ai-review] attached {len(reviews)} verdicts across {len(bundle)} picks")


def _refresh_plays_blocking():
    import gc
    try:
        try:
            games = get_todays_games()
        except Exception as e:
            print(f"[plays-refresh] failed to load schedule: {e}")
            return

        all_plays = []
        all_hr_threats = []
        all_nrfi = []
        all_hrr = []

        def _publish_partial():
            """Push current results into the cache so the UI can render them
            progressively as each game finishes — instead of waiting for the
            entire slate scan to complete."""
            sorted_plays = sorted(all_plays, key=lambda p: p.get("probability", 0), reverse=True)
            sorted_hr = sorted(all_hr_threats, key=lambda x: x.get("probability", 0), reverse=True)
            sorted_nrfi = sorted(all_nrfi, key=lambda x: x.get("probability", 0), reverse=True)
            sorted_hrr = sorted(all_hrr, key=lambda x: x.get("probability", 0), reverse=True)
            # Diversify the displayed plays. Two-pass strategy guarantees
            # that EVERY game on the slate gets at least one entry on the
            # board, then fills the remaining slots by raw probability with
            # a per-game cap. Previously only top-prob games made the cut
            # and 4-5 games could be entirely missing from the dashboard.
            diversified = []
            seen_ids: set[int] = set()
            game_counts: dict = {}
            # Pass 1: top play from each unique game (sorted by prob desc).
            for p in sorted_plays:
                gpk = p.get("game_pk")
                if gpk is None or gpk in game_counts:
                    continue
                diversified.append(p)
                seen_ids.add(id(p))
                game_counts[gpk] = 1
            # Total slots scale with slate size so a 15-game card isn't
            # crammed into a fixed 20-row list.
            target = max(PLAYS_LIMIT, len(game_counts) * 2)
            # Pass 2: fill with the next-best plays, respecting per-game cap.
            for p in sorted_plays:
                if id(p) in seen_ids:
                    continue
                gpk = p.get("game_pk")
                if gpk is not None and game_counts.get(gpk, 0) >= PLAYS_PER_GAME_CAP:
                    continue
                diversified.append(p)
                seen_ids.add(id(p))
                game_counts[gpk] = game_counts.get(gpk, 0) + 1
                if len(diversified) >= target:
                    break

            # Same first-pass-per-game treatment for HRR combos so the
            # 1+ H/R/RBI board doesn't get monopolized by 1-2 games. Also
            # apply the per-game cap on the fill pass so a single hot lineup
            # can't take 6+ of the 12 slots.
            hrr_diversified = []
            hrr_seen_ids: set[int] = set()
            hrr_game_counts: dict = {}
            for c in sorted_hrr:
                gpk = c.get("game_pk")
                if gpk is None or gpk in hrr_game_counts:
                    continue
                hrr_diversified.append(c)
                hrr_seen_ids.add(id(c))
                hrr_game_counts[gpk] = 1
            for c in sorted_hrr:
                if id(c) in hrr_seen_ids:
                    continue
                gpk = c.get("game_pk")
                if gpk is not None and hrr_game_counts.get(gpk, 0) >= PLAYS_PER_GAME_CAP:
                    continue
                hrr_diversified.append(c)
                hrr_seen_ids.add(id(c))
                hrr_game_counts[gpk] = hrr_game_counts.get(gpk, 0) + 1
                if len(hrr_diversified) >= HRR_COMBO_LIMIT:
                    break
            hrr_diversified = hrr_diversified[:HRR_COMBO_LIMIT]

            specials = _build_specials(sorted_plays, sorted_hr)
            with _PLAYS_LOCK:
                PLAYS_CACHE["ts"] = time.time()
                PLAYS_CACHE["data"] = diversified
                HR_THREATS_CACHE["ts"] = time.time()
                HR_THREATS_CACHE["data"] = sorted_hr[:HR_THREATS_LIMIT]
                NRFI_CACHE["ts"] = time.time()
                NRFI_CACHE["data"] = sorted_nrfi
                HRR_COMBO_CACHE["ts"] = time.time()
                HRR_COMBO_CACHE["data"] = hrr_diversified
                # Locks = top plays before per-game diversification, deduped
                # by (player, stat) so doubleheaders don't repeat in the hero.
                seen_locks = set()
                deduped_locks = []
                for p in sorted_plays:
                    key = (p.get("headline"), p.get("stat_label"))
                    if key in seen_locks:
                        continue
                    seen_locks.add(key)
                    deduped_locks.append(p)
                    if len(deduped_locks) >= LOCKS_LIMIT:
                        break
                LOCKS_CACHE["ts"] = time.time()
                LOCKS_CACHE["data"] = deduped_locks
                SPECIALS_CACHE["ts"] = time.time()
                SPECIALS_CACHE["data"] = specials

        def _absorb_game(g):
            """Run one game's board build and merge its plays + HR threats + NRFI + HRR combo."""
            game_pk = g.get("gamePk")
            try:
                all_plays.extend(_build_plays_for_game(g))
                cached = BOARD_CACHE.get(game_pk)
                if cached and cached.get("data"):
                    all_hr_threats.extend(cached["data"].get("hr_threats", []) or [])
                    nrfi_entry = cached["data"].get("nrfi")
                    if nrfi_entry:
                        all_nrfi.append(nrfi_entry)
                    all_hrr.extend(cached["data"].get("hrr_combo", []) or [])
            except Exception as e:
                print(f"[plays-refresh] game {game_pk} failed: {e}")
            gc.collect()
            _publish_partial()

        if PLAYS_GAME_CONCURRENCY <= 1:
            # Sequential mode for memory-constrained hosts.
            for g in games:
                _absorb_game(g)
        else:
            with ThreadPoolExecutor(max_workers=PLAYS_GAME_CONCURRENCY) as pool:
                futures = {pool.submit(_build_plays_for_game, g): g for g in games}
                for fut in as_completed(futures):
                    g = futures[fut]
                    game_pk = g.get("gamePk")
                    try:
                        all_plays.extend(fut.result())
                        cached = BOARD_CACHE.get(game_pk)
                        if cached and cached.get("data"):
                            all_hr_threats.extend(cached["data"].get("hr_threats", []) or [])
                            nrfi_entry = cached["data"].get("nrfi")
                            if nrfi_entry:
                                all_nrfi.append(nrfi_entry)
                            all_hrr.extend(cached["data"].get("hrr_combo", []) or [])
                    except Exception as e:
                        print(f"[plays-refresh] game {game_pk} failed: {e}")
                    gc.collect()
                    _publish_partial()

        print(f"[plays-refresh] done: {len(all_plays)} plays, {len(all_hr_threats)} HR threats")
        # OpenAI second-opinion pass over the published top picks. Runs once
        # per refresh cycle and uses gpt-4o-mini, so cost stays small. If the
        # API call fails or no key is set, this is a no-op.
        try:
            _run_ai_review_pass()
        except Exception as e:
            print(f"[ai-review] pass failed: {e}")
    except Exception as e:
        print(f"[plays-refresh] unexpected error: {e}")
    finally:
        # ALWAYS release the computing flag, even on crash. Prevents "stuck
        # on computing" state forever if the worker hits an unhandled error.
        with _PLAYS_LOCK:
            PLAYS_CACHE["computing"] = False


def _ensure_plays_refresh():
    """Kick off a background plays refresh if the cache is stale and no
    refresh is already running. Caller holds nothing; returns whether the
    cache is currently fresh (True = no refresh needed)."""
    now = time.time()
    with _PLAYS_LOCK:
        fresh = PLAYS_CACHE["data"] and now - PLAYS_CACHE["ts"] < PLAYS_CACHE_TTL_SECONDS
        if not fresh and not PLAYS_CACHE["computing"]:
            PLAYS_CACHE["computing"] = True
            t = threading.Thread(target=_refresh_plays_blocking, daemon=True)
            t.start()
    return fresh


def get_plays_of_day_snapshot():
    """Return what we have right now, kicking off a background refresh if stale."""
    fresh = _ensure_plays_refresh()
    with _PLAYS_LOCK:
        return {
            "plays": list(PLAYS_CACHE["data"]),
            "computing": PLAYS_CACHE["computing"] and not fresh,
            "ts": PLAYS_CACHE["ts"],
        }


@app.route("/", methods=["GET"])
@login_required
def home():
    games = get_todays_games()
    return render_template("index.html", games=games, best_plays=[])


@app.route("/api/hr-threats", methods=["GET"])
@login_required
def api_hr_threats():
    with _PLAYS_LOCK:
        data = list(HR_THREATS_CACHE["data"])
        ts = HR_THREATS_CACHE["ts"]
        computing = PLAYS_CACHE["computing"]
    return jsonify({
        "threats": data,
        "ts": ts,
        "computing": computing,
    })


@app.route("/api/plays-of-day", methods=["GET"])
@login_required
def api_plays_of_day():
    return jsonify(get_plays_of_day_snapshot())


@app.route("/api/nrfi", methods=["GET"])
@login_required
def api_nrfi():
    """No-Runs-First-Inning watch board for every game tonight."""
    with _PLAYS_LOCK:
        data = list(NRFI_CACHE["data"])
        ts = NRFI_CACHE["ts"]
        computing = PLAYS_CACHE["computing"]
    return jsonify({"nrfi": data, "ts": ts, "computing": computing})


@app.route("/api/locks", methods=["GET"])
@login_required
def api_locks():
    """Top N single plays of the night, uncapped by per-game diversification.
    Used by the hero strip on the dashboard so it shows the truly highest-
    probability plays — not just the top of the diversified board."""
    _ensure_plays_refresh()
    with _PLAYS_LOCK:
        data = list(LOCKS_CACHE["data"])
        ts = LOCKS_CACHE["ts"]
        computing = PLAYS_CACHE["computing"]
    return jsonify({"locks": data, "ts": ts, "computing": computing})


@app.route("/api/hrr-combo", methods=["GET"])
@login_required
def api_hrr_combo():
    """Best Hits + Runs + RBIs combo prop board for tonight."""
    _ensure_plays_refresh()
    with _PLAYS_LOCK:
        data = list(HRR_COMBO_CACHE["data"])
        ts = HRR_COMBO_CACHE["ts"]
        computing = PLAYS_CACHE["computing"]
    return jsonify({"combos": data, "ts": ts, "computing": computing})


@app.route("/api/specials", methods=["GET"])
@login_required
def api_specials():
    """Daily specials: best run line, 3-leg SGP, HR pair."""
    with _PLAYS_LOCK:
        data = dict(SPECIALS_CACHE["data"])
        ts = SPECIALS_CACHE["ts"]
        computing = PLAYS_CACHE["computing"]
    return jsonify({"specials": data, "ts": ts, "computing": computing})


@app.route("/admin/warm-cache", methods=["GET", "POST"])
@login_required
def admin_warm_cache():
    """Kick off a background slate scan to pre-warm the profile + plays cache.
    Useful right after a Render deploy. Returns immediately; UI shows
    'computing...' until done. Safe to call repeatedly — no-op if already
    running or already fresh."""
    started = False
    with _PLAYS_LOCK:
        if not PLAYS_CACHE["computing"]:
            PLAYS_CACHE["computing"] = True
            t = threading.Thread(target=_refresh_plays_blocking, daemon=True)
            t.start()
            started = True
    return jsonify({
        "started": started,
        "message": (
            "Cache warm-up started in background. Check /api/plays-of-day "
            "in 1-2 minutes for results."
            if started
            else "Cache warm-up already in progress."
        ),
    })


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
            "hrr_combo": [],
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

    sorted_hits = _sort(boards["top_hits"])
    sorted_hr = _sort(boards["top_home_runs"])
    sorted_tb = _sort(boards["top_total_bases"])
    sorted_rbi = _sort(boards["top_rbis"])
    sorted_k = _sort(boards["top_strikeouts"])
    sorted_hrr = _sort(boards.get("hrr_combo", []))

    # On-demand AI review for THIS game's top picks. Single batched call,
    # cached 6h via ai_review._CACHE so repeat opens are free. Top 2 from
    # each category keeps the prompt small (~10-12 picks total).
    try:
        bundle = []
        for items, label in (
            (sorted_hits, "Hits"),
            (sorted_hr, "Home Runs"),
            (sorted_tb, "Total Bases"),
            (sorted_rbi, "RBIs"),
            (sorted_k, "Strikeouts"),
            (sorted_hrr, "1+ H/R/RBI"),
        ):
            for p in items:
                if p.get("pick") == "PASS":
                    continue
                # Tag stat_label so AI prompt + pick_key reflect category.
                if not p.get("stat_label"):
                    p["stat_label"] = label
                bundle.append(p)
        if bundle:
            reviews = ai_review.review_picks(bundle, kind=f"game-{game_pk}")
            if reviews:
                ai_review.attach_reviews(bundle, reviews)
    except Exception as e:
        print(f"[ai-review] game {game_pk} failed: {e}")

    return render_template(
        "game_detail.html",
        game=game,
        error=None,
        top_hits=sorted_hits,
        top_home_runs=sorted_hr,
        top_total_bases=sorted_tb,
        top_rbis=sorted_rbi,
        top_strikeouts=sorted_k,
        hrr_combo=sorted_hrr,
        spread_lean=boards["spread_lean"],
        weather=boards["weather"],
        lineup_confirmed=boards["lineup_confirmed"],
        projected_mode=boards["projected_mode"],
        sort_mode=sort_mode,
    )


def _auto_warm_cache_on_boot():
    """Fire a background slate warm-up once per process at startup.

    Skipped when AUTO_WARM_CACHE=0 or when running under Flask's debug
    reloader parent process (only the child should warm)."""
    if os.getenv("AUTO_WARM_CACHE", "1") != "1":
        return
    if os.getenv("WERKZEUG_RUN_MAIN") == "false":
        return
    try:
        with _PLAYS_LOCK:
            if PLAYS_CACHE["computing"]:
                return
            PLAYS_CACHE["computing"] = True
        t = threading.Thread(target=_refresh_plays_blocking, daemon=True)
        t.start()
        print("[startup] auto-warming slate cache in background")
    except Exception as e:
        print(f"[startup] auto-warm failed: {e}")


_auto_warm_cache_on_boot()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)