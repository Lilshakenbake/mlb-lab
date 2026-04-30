"""Auto-grader: pulls final box scores from the MLB Stats API and settles
pending tracked picks (hitter/pitcher props, moneyline, run line)."""

import threading
import time
from datetime import datetime
from typing import Optional

import requests

from src import tracker

LIVE_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
HTTP_TIMEOUT = 15

_GRADE_LOCK = threading.Lock()
_GRADE_STATE = {
    "running": False,
    "last_run_ts": 0,
    "last_result": None,  # {graded, skipped, errors}
}
GRADE_THROTTLE_SECONDS = 300  # 5 minutes between auto-grades

BAT_FIELD_MAP = {
    "Hits": "hits",
    "Total Bases": "totalBases",
    "Home Runs": "homeRuns",
    "RBIs": "rbi",
}

# Combo prop: sum of multiple boxscore fields. The "1+ H/R/RBI" prop
# resolves to OVER 0.5 if the sum >= 1 (any of the three contributed).
# Legacy "H+R+RBI" label is kept so previously-tracked plays continue
# to grade after the relabel.
COMBO_FIELD_MAP = {
    "1+ H/R/RBI": ("hits", "runs", "rbi"),
    "H+R+RBI": ("hits", "runs", "rbi"),
}


def _fetch_feed(game_pk: int) -> Optional[dict]:
    try:
        r = requests.get(LIVE_FEED_URL.format(game_pk=game_pk), timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def _is_final(feed: dict) -> bool:
    status = feed.get("gameData", {}).get("status", {}) or {}
    abstract = status.get("abstractGameState") or ""
    coded = status.get("codedGameState") or ""
    detailed = status.get("detailedState") or ""
    if abstract == "Final" or coded == "F":
        return True
    if "Final" in detailed or "Game Over" in detailed or "Completed Early" in detailed:
        return True
    return False


def _team_names(feed: dict) -> tuple[str, str]:
    teams = feed.get("gameData", {}).get("teams", {}) or {}
    home = teams.get("home", {})
    away = teams.get("away", {})
    return (
        (home.get("name") or "").strip(),
        (away.get("name") or "").strip(),
    )


def _team_short_names(feed: dict) -> tuple[str, str]:
    teams = feed.get("gameData", {}).get("teams", {}) or {}
    home = teams.get("home", {})
    away = teams.get("away", {})
    return (
        (home.get("teamName") or home.get("clubName") or "").strip(),
        (away.get("teamName") or away.get("clubName") or "").strip(),
    )


def _runs(feed: dict) -> tuple[Optional[int], Optional[int]]:
    line = feed.get("liveData", {}).get("linescore", {}).get("teams", {}) or {}
    return (
        line.get("home", {}).get("runs"),
        line.get("away", {}).get("runs"),
    )


def _find_player_actual(feed: dict, headline: str, stat_label: str) -> Optional[float]:
    """Search both rosters for the named player and return the requested stat."""
    boxscore = feed.get("liveData", {}).get("boxscore", {}) or {}
    target = (headline or "").lower().strip()
    if not target:
        return None
    target_last = target.split()[-1]

    for side in ("home", "away"):
        players = boxscore.get("teams", {}).get(side, {}).get("players", {}) or {}
        for _pid, pdata in players.items():
            person = pdata.get("person", {}) or {}
            full = (person.get("fullName") or "").lower().strip()
            if not full:
                continue
            match = (
                full == target
                or full.replace(".", "") == target.replace(".", "")
                or full.endswith(" " + target_last) and full.split()[0][0] == target.split()[0][0]
            )
            if not match:
                continue

            stats = pdata.get("stats", {}) or {}
            if stat_label in BAT_FIELD_MAP:
                bat = stats.get("batting", {}) or {}
                field = BAT_FIELD_MAP[stat_label]
                if field in bat:
                    return float(bat[field])
            elif stat_label in COMBO_FIELD_MAP:
                bat = stats.get("batting", {}) or {}
                fields = COMBO_FIELD_MAP[stat_label]
                if all(f in bat for f in fields):
                    return float(sum(bat[f] for f in fields))
            elif stat_label == "Strikeouts":
                pit = stats.get("pitching", {}) or {}
                if "strikeOuts" in pit:
                    return float(pit["strikeOuts"])
    return None


def _grade_over_under(actual: Optional[float], line: Optional[float], pick: str):
    if actual is None or line is None or pick not in ("OVER", "UNDER"):
        return None
    if pick == "OVER":
        if actual > line:
            return ("WIN", actual)
        if actual < line:
            return ("LOSS", actual)
        return ("PUSH", actual)
    if actual < line:
        return ("WIN", actual)
    if actual > line:
        return ("LOSS", actual)
    return ("PUSH", actual)


def _team_side_in_feed(feed: dict, headline: str) -> Optional[str]:
    """Return 'home' or 'away' if headline matches one of the teams."""
    home_full, away_full = _team_names(feed)
    home_short, away_short = _team_short_names(feed)
    h = (headline or "").lower().strip()
    if not h:
        return None
    if home_full and (h in home_full.lower() or home_full.lower() in h):
        return "home"
    if away_full and (h in away_full.lower() or away_full.lower() in h):
        return "away"
    if home_short and h.startswith(home_short.lower()):
        return "home"
    if away_short and h.startswith(away_short.lower()):
        return "away"
    return None


def _grade_moneyline(feed: dict, headline: str):
    side = _team_side_in_feed(feed, headline)
    home_r, away_r = _runs(feed)
    if side is None or home_r is None or away_r is None:
        return None
    if home_r == away_r:
        return ("PUSH", abs(home_r - away_r))
    if side == "home":
        return ("WIN" if home_r > away_r else "LOSS", abs(home_r - away_r))
    return ("WIN" if away_r > home_r else "LOSS", abs(home_r - away_r))


def _parse_runline_spread(headline: str) -> tuple[str, float]:
    """Pull a +1.5/-1.5 (etc) from the headline. Default to -1.5 if absent."""
    if not headline:
        return ("", -1.5)
    h = headline.strip()
    for token in ("-1.5", "+1.5", "-2.5", "+2.5"):
        if token in h:
            try:
                spread = float(token)
            except ValueError:
                spread = -1.5
            return (h.replace(token, "").strip(), spread)
    return (h, -1.5)


def _grade_runline(feed: dict, headline: str):
    team, spread = _parse_runline_spread(headline)
    side = _team_side_in_feed(feed, team)
    home_r, away_r = _runs(feed)
    if side is None or home_r is None or away_r is None:
        return None
    margin = (home_r - away_r) if side == "home" else (away_r - home_r)
    adjusted = margin + spread
    if adjusted > 0:
        return ("WIN", margin)
    if adjusted < 0:
        return ("LOSS", margin)
    return ("PUSH", margin)


def _grade_one(feed: dict, play: dict):
    kind = play.get("kind")
    if kind in ("hitter", "pitcher", "hitter_combo"):
        actual = _find_player_actual(feed, play.get("headline", ""), play.get("stat_label", ""))
        return _grade_over_under(actual, play.get("line"), play.get("pick"))
    if kind == "moneyline":
        return _grade_moneyline(feed, play.get("headline", ""))
    if kind == "runline":
        return _grade_runline(feed, play.get("headline", ""))
    return None


def grade_pending_plays() -> dict:
    """Walk pending picks, fetch the feed once per game_pk, and settle anything we can."""
    pending = tracker.list_plays(only_pending=True)

    by_game: dict[int, list[dict]] = {}
    no_game = []
    for p in pending:
        gp = p.get("game_pk")
        if gp is None:
            no_game.append(p)
        else:
            by_game.setdefault(int(gp), []).append(p)

    graded = 0
    skipped_not_final = 0
    skipped_no_data = 0
    errors = 0

    for game_pk, plays in by_game.items():
        feed = _fetch_feed(game_pk)
        if not feed:
            errors += len(plays)
            continue
        if not _is_final(feed):
            skipped_not_final += len(plays)
            continue
        for p in plays:
            try:
                outcome = _grade_one(feed, p)
            except Exception:
                errors += 1
                continue
            if outcome is None:
                skipped_no_data += 1
                continue
            result, actual = outcome
            ok = tracker.settle_play(
                p["id"], result, actual_value=actual, notes="auto-graded"
            )
            if ok:
                graded += 1
            else:
                errors += 1

    skipped_no_data += len(no_game)

    return {
        "graded": graded,
        "skipped_not_final": skipped_not_final,
        "skipped_no_data": skipped_no_data,
        "errors": errors,
        "pending_seen": len(pending),
    }


def _run_in_background():
    try:
        result = grade_pending_plays()
    except Exception as e:
        result = {"error": str(e)}
    with _GRADE_LOCK:
        _GRADE_STATE["running"] = False
        _GRADE_STATE["last_run_ts"] = time.time()
        _GRADE_STATE["last_result"] = result


def trigger_background_grade(force: bool = False) -> dict:
    """Kick off a background grade if it's stale and not already running.
    Returns a snapshot of the current state."""
    now = time.time()
    with _GRADE_LOCK:
        running = _GRADE_STATE["running"]
        last = _GRADE_STATE["last_run_ts"]
        stale = (now - last) > GRADE_THROTTLE_SECONDS
        will_run = not running and (force or stale)
        if will_run:
            _GRADE_STATE["running"] = True
            t = threading.Thread(target=_run_in_background, daemon=True)
            t.start()
        snapshot = {
            "started": will_run,
            "running": _GRADE_STATE["running"],
            "last_run_ts": _GRADE_STATE["last_run_ts"],
            "last_result": _GRADE_STATE["last_result"],
        }
    return snapshot


def get_state() -> dict:
    with _GRADE_LOCK:
        return {
            "running": _GRADE_STATE["running"],
            "last_run_ts": _GRADE_STATE["last_run_ts"],
            "last_result": _GRADE_STATE["last_result"],
        }


def humanize_age(ts: float) -> str:
    if not ts:
        return "never"
    delta = time.time() - ts
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
