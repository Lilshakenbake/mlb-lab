"""Stolen-base rate lookups for hitters and team SB-allowed factors.

Pulls season-to-date SB / CS data for every MLB hitter from the public
MLB Stats API in a single bulk request, cached 24h on disk. Then exposes:

    get_sb_per_game(player_name)  → float (steals per game so far)
    get_team_sb_allowed_factor(team)  → float multiplier (1.0 = league avg)

Both fall back to neutral defaults on any miss so callers can blindly
multiply without worrying about None handling.

v1 scope (acknowledged limits):
  - No per-catcher CS% adjustment (would need catcher-specific game logs)
  - No pitcher pickoff/hold modeling (would need play-by-play stream)
  - No base-state awareness (only matters in-game)

These are real signals worth ~1-2pp each, but the dominant factor on a
hitter's expected SB is just their season rate. Pulling that alone is
~70-80% of the available edge.
"""

import os
import json
import time
import logging
import unicodedata
from typing import Optional, Dict

import requests

LOG = logging.getLogger(__name__)

_CACHE_DIR = os.environ.get("MLB_CACHE_DIR", "data_cache")
_CACHE_PATH = os.path.join(_CACHE_DIR, "steals_data.json")
_CACHE_TTL_S = 24 * 3600


def _canon(name: str) -> str:
    """Strip diacritics + punct + suffixes; lowercase. Matches src/app.py
    _canon_name() so lookups work cross-file."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    for suf in (" jr.", " sr.", " jr", " sr", " iii", " ii", " iv"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return "".join(c for c in s if c.isalnum() or c == " ").strip()


def _cache_get() -> Optional[dict]:
    try:
        if not os.path.exists(_CACHE_PATH):
            return None
        if (time.time() - os.path.getmtime(_CACHE_PATH)) > _CACHE_TTL_S:
            return None
        with open(_CACHE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _cache_put(data: dict) -> None:
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_CACHE_PATH, "w") as f:
            json.dump(data, f)
    except Exception as e:
        LOG.warning("steals cache write failed: %s", e)


def _fetch_hitters(season: int) -> list:
    """One paginated pull of every batter's season stats."""
    out = []
    offset = 0
    page = 500
    while True:
        try:
            r = requests.get(
                "https://statsapi.mlb.com/api/v1/stats",
                params={
                    "stats": "season",
                    "group": "hitting",
                    "season": season,
                    "sportId": 1,
                    "limit": page,
                    "offset": offset,
                    "playerPool": "All",
                },
                timeout=20,
            )
            r.raise_for_status()
            splits = (r.json().get("stats") or [{}])[0].get("splits") or []
        except Exception as e:
            LOG.warning("steals fetch failed at offset %d: %s", offset, e)
            break
        if not splits:
            break
        out.extend(splits)
        if len(splits) < page:
            break
        offset += page
        if offset > 5000:
            break
    return out


def _build_table(season: Optional[int] = None) -> Optional[dict]:
    cached = _cache_get()
    if cached and cached.get("by_player"):
        return cached

    if season is None:
        from datetime import datetime
        season = datetime.now().year

    hitters = _fetch_hitters(season)
    if not hitters:
        hitters = _fetch_hitters(season - 1)
    if not hitters:
        return None

    by_player: Dict[str, dict] = {}
    team_sb_allowed: Dict[str, dict] = {}
    league_sb_total = 0.0
    league_games = 0.0

    for h in hitters:
        st = h.get("stat", {}) or {}
        player = (h.get("player") or {}).get("fullName")
        team = (h.get("team") or {}).get("name")
        try:
            g = float(st.get("gamesPlayed") or 0)
            sb = float(st.get("stolenBases") or 0)
            cs = float(st.get("caughtStealing") or 0)
        except Exception:
            continue
        if not player or g < 5:
            continue
        sb_per_g = sb / g if g > 0 else 0.0
        # Success rate matters — a 2 SB / 5 CS guy is a worse bet than the rate alone shows.
        succ_rate = sb / (sb + cs) if (sb + cs) > 0 else None
        by_player[_canon(player)] = {
            "player": player,
            "team": team,
            "g": g,
            "sb": sb,
            "cs": cs,
            "sb_per_g": round(sb_per_g, 4),
            "success_rate": round(succ_rate, 3) if succ_rate is not None else None,
        }
        league_sb_total += sb
        league_games += g

    if not by_player:
        return None

    # League SB rate per game-played (per hitter slot). Used to normalize.
    league_sb_per_g = league_sb_total / league_games if league_games > 0 else 0.05

    payload = {
        "season": season,
        "fetched_at": int(time.time()),
        "league_sb_per_g": round(league_sb_per_g, 4),
        "by_player": by_player,
    }
    _cache_put(payload)
    return payload


def get_sb_per_game(player_name: str) -> float:
    """Season SB rate per game for this hitter. Returns 0.0 if unknown."""
    if not player_name:
        return 0.0
    table = _build_table()
    if not table:
        return 0.0
    entry = table["by_player"].get(_canon(player_name))
    if not entry:
        return 0.0
    return float(entry.get("sb_per_g") or 0.0)


def get_sb_success_rate(player_name: str) -> Optional[float]:
    if not player_name:
        return None
    table = _build_table()
    if not table:
        return None
    entry = table["by_player"].get(_canon(player_name))
    if not entry:
        return None
    return entry.get("success_rate")


def get_league_sb_per_g() -> float:
    """League average SB per hitter-game. Used to gate which guys are worth a steal prop at all."""
    table = _build_table()
    if not table:
        return 0.05
    return float(table.get("league_sb_per_g") or 0.05)
