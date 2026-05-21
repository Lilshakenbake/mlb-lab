"""Live team-bullpen multipliers from the MLB Stats API.

Replaces the hardcoded ranks in `bullpen_factors.BULLPEN_FACTORS` with a
24-hour-cached, season-to-date pull. Pitchers are classified as relievers
when GS/G < 0.3 (the standard cutoff — anyone with a meaningful starter
share doesn't belong in the pen calculation).

For each team we compute:
  - bullpen ERA, K/9, HR/9
And normalize them against the LEAGUE bullpen mean to derive multipliers
in the same shape as the hardcoded `BULLPEN_FACTORS` dict:

    {"hits": float, "tb": float, "hr": float, "k": float}

The multipliers cap at ±15% so a single bad bullpen month can't blow up
the whole projection. Falls back silently to None on any error — the
caller in `bullpen_factors.get_bullpen_factor` then uses the hardcoded
ranks.
"""

import os
import json
import time
import logging
from typing import Optional, Dict

import requests

LOG = logging.getLogger(__name__)

# 24h on-disk cache. Lives alongside the other caches.
_CACHE_DIR = os.environ.get("MLB_CACHE_DIR", "data_cache")
_CACHE_PATH = os.path.join(_CACHE_DIR, "bullpen_live.json")
_CACHE_TTL_S = 24 * 3600

# Multiplier caps — bullpen is only part of the picture, so don't let any
# single team swing more than ±15% off league average.
_CAP = 0.15

# Per-pitcher classifier. A pitcher with GS/G ≥ 0.3 is treated as a starter
# (and skipped from the bullpen aggregation).
_RELIEVER_GS_RATIO = 0.30


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
        LOG.warning("bullpen_live cache write failed: %s", e)


def _fetch_all_pitchers(season: int) -> list:
    """One paginated pull of every pitcher's season stats."""
    out = []
    offset = 0
    page = 500
    while True:
        try:
            r = requests.get(
                "https://statsapi.mlb.com/api/v1/stats",
                params={
                    "stats": "season",
                    "group": "pitching",
                    "season": season,
                    "sportId": 1,
                    "limit": page,
                    "offset": offset,
                    # CRITICAL: without playerPool=All the API returns
                    # only ~50 qualified pitchers (no relievers qualify).
                    "playerPool": "All",
                },
                timeout=20,
            )
            r.raise_for_status()
            splits = (r.json().get("stats") or [{}])[0].get("splits") or []
        except Exception as e:
            LOG.warning("bullpen_live pitcher fetch failed at offset %d: %s", offset, e)
            break
        if not splits:
            break
        out.extend(splits)
        if len(splits) < page:
            break
        offset += page
        if offset > 5000:  # hard stop, no chance of legit need
            break
    return out


def _aggregate_by_team(pitchers: list) -> Dict[str, dict]:
    """Aggregate reliever stats per team. Returns
    { team_name: { 'era': float, 'k9': float, 'hr9': float, 'ip': float } }
    """
    teams: Dict[str, dict] = {}
    for p in pitchers:
        st = p.get("stat", {}) or {}
        team = (p.get("team") or {}).get("name")
        if not team:
            continue
        try:
            g = int(st.get("gamesPlayed") or 0)
            gs = int(st.get("gamesStarted") or 0)
            if g <= 0:
                continue
            if (gs / g) >= _RELIEVER_GS_RATIO:
                continue  # treat as starter
            # MLB IP uses baseball notation: ".1" = 1/3 inning, ".2" = 2/3.
            # Convert "12.1" → 12 + 1/3 = 12.333, NOT 12.1.
            ip_raw = str(st.get("inningsPitched") or "0")
            try:
                whole, _, frac = ip_raw.partition(".")
                whole_i = int(whole) if whole else 0
                frac_i = int(frac[:1]) if frac else 0  # ".1" or ".2"
                ip = whole_i + (frac_i / 3.0 if frac_i in (1, 2) else 0.0)
            except Exception:
                ip = 0.0
            if ip < 5.0:
                continue  # too thin, skip (call-ups / cup-of-coffee)
            er = float(st.get("earnedRuns") or 0)
            so = float(st.get("strikeOuts") or st.get("strikeouts") or 0)
            hr = float(st.get("homeRuns") or 0)
        except Exception:
            continue

        bucket = teams.setdefault(team, {"ip": 0.0, "er": 0.0, "so": 0.0, "hr": 0.0})
        bucket["ip"] += ip
        bucket["er"] += er
        bucket["so"] += so
        bucket["hr"] += hr

    out: Dict[str, dict] = {}
    for team, b in teams.items():
        ip = b["ip"]
        if ip < 30.0:  # team's pen barely throws — skip and let caller fall back
            continue
        out[team] = {
            "era": round(b["er"] * 9.0 / ip, 3),
            "k9": round(b["so"] * 9.0 / ip, 3),
            "hr9": round(b["hr"] * 9.0 / ip, 3),
            "ip": round(ip, 1),
        }
    return out


def _league_means(per_team: Dict[str, dict]) -> dict:
    if not per_team:
        return {}
    eras = [v["era"] for v in per_team.values()]
    k9s = [v["k9"] for v in per_team.values()]
    hr9s = [v["hr9"] for v in per_team.values()]
    return {
        "era": sum(eras) / len(eras),
        "k9": sum(k9s) / len(k9s),
        "hr9": sum(hr9s) / len(hr9s),
    }


def _to_multipliers(team_stats: dict, league: dict) -> dict:
    """Convert raw bullpen rates into the same multiplier shape as the
    hardcoded BULLPEN_FACTORS dict.

    Sign convention (same as the hardcoded table):
      hits/tb/hr > 1.0 → bad pen, inflates hitter projections
      hits/tb/hr < 1.0 → good pen, suppresses
      k > 1.0          → bullpen strikes guys out → boost pitcher K props
      k < 1.0          → contact pen → fade pitcher K props
    """
    le = league.get("era") or 4.00
    lk = league.get("k9") or 9.00
    lh = league.get("hr9") or 1.10

    era_ratio = team_stats["era"] / le if le else 1.0  # >1 = worse pen
    k_ratio = team_stats["k9"] / lk if lk else 1.0     # >1 = whiff pen
    hr_ratio = team_stats["hr9"] / lh if lh else 1.0   # >1 = homer-prone

    def _cap(x):
        return max(1.0 - _CAP, min(1.0 + _CAP, x))

    # Scale: 25% of ERA-ratio deviation lands in the hit multiplier (since
    # ERA captures more than just hits). TB tracks the same direction
    # slightly stronger. HR multiplier comes directly from HR/9 ratio.
    hits = _cap(1.0 + (era_ratio - 1.0) * 0.50)
    tb = _cap(1.0 + (era_ratio - 1.0) * 0.65)
    hr = _cap(1.0 + (hr_ratio - 1.0) * 0.80)
    # K multiplier is INVERTED relative to ERA: a bad pen still strikes
    # guys out if their K/9 is high. Use K/9 directly.
    k = _cap(1.0 + (k_ratio - 1.0) * 0.60)

    return {
        "hits": round(hits, 3),
        "tb": round(tb, 3),
        "hr": round(hr, 3),
        "k": round(k, 3),
    }


def get_live_bullpen_table(season: Optional[int] = None) -> Optional[dict]:
    """Return { team_name: multipliers_dict } from the live MLB API.

    Cached 24h. Returns None on any failure so the caller can fall back
    to the hardcoded ranks. Multipliers match BULLPEN_FACTORS shape.
    """
    cached = _cache_get()
    if cached and cached.get("multipliers"):
        return cached["multipliers"]

    if season is None:
        from datetime import datetime
        season = datetime.now().year

    pitchers = _fetch_all_pitchers(season)
    if not pitchers:
        # Last fallback: try previous season (early March before data exists)
        pitchers = _fetch_all_pitchers(season - 1)
    if not pitchers:
        return None

    per_team = _aggregate_by_team(pitchers)
    if len(per_team) < 20:  # incomplete season pull — bail
        return None
    league = _league_means(per_team)
    if not league:
        return None

    multipliers = {team: _to_multipliers(stats, league) for team, stats in per_team.items()}
    _cache_put({
        "season": season,
        "fetched_at": int(time.time()),
        "league": {k: round(v, 3) for k, v in league.items()},
        "raw_per_team": per_team,
        "multipliers": multipliers,
    })
    return multipliers
