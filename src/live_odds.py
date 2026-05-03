"""Live sportsbook odds via The Odds API (the-odds-api.com).

Compares our model's fair odds vs the actual prices at DraftKings, FanDuel,
MGM, Caesars, etc. Surfaces edge % so we only fire when a book is stale
relative to our number — the only thing that beats sportsbooks long-term.

Quota strategy (free tier = 500 req/month):
- Game odds (h2h/spreads/totals): 1 request returns ALL games for the slate.
  Refresh every 2 hours during the day → ~12/day → ~360/month. Fits.
- Player props: skipped in MVP (each prop market = 1 quota PER game; would
  burn the budget in a day). Add as on-demand v2 once usage is proven.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import requests

from src import cache as _cache

API_KEY = os.getenv("ODDS_API_KEY", "").strip()
BASE = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"
REGIONS = "us"
GAME_MARKETS = "h2h,spreads,totals"
# Player prop markets (each one is +1 quota unit per event request).
PROP_MARKETS = "batter_hits,batter_home_runs,batter_total_bases,batter_rbis,pitcher_strikeouts"
GAME_TTL = 2 * 3600       # 2 hours
PROP_TTL = 12 * 3600      # 12 hours per game (controls quota burn)
TIMEOUT = 10

# Track last quota usage from response headers so we can show it in UI.
_USAGE = {"used": None, "remaining": None, "ts": None}


def is_enabled() -> bool:
    return bool(API_KEY)


def get_usage() -> dict:
    return dict(_USAGE)


def _american_to_prob(american: float) -> float:
    """Convert American odds to implied probability (with vig)."""
    a = float(american)
    if a >= 0:
        return 100.0 / (a + 100.0)
    return -a / (-a + 100.0)


def _devig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Strip the book's vig from a 2-way market by proportional scaling."""
    s = prob_a + prob_b
    if s <= 0:
        return prob_a, prob_b
    return prob_a / s, prob_b / s


def fetch_game_odds() -> Optional[list]:
    """Pull moneyline + spread + total for every MLB game today.
    Returns the raw odds list or None on failure. Cached 2hr to disk."""
    if not is_enabled():
        return None

    cached = _cache.get("live_odds_games", "today", GAME_TTL)
    if cached is not None:
        return cached

    url = f"{BASE}/sports/{SPORT}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": REGIONS,
        "markets": GAME_MARKETS,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        # Capture quota headers for the UI.
        try:
            _USAGE["used"] = int(resp.headers.get("x-requests-used", 0))
            _USAGE["remaining"] = int(resp.headers.get("x-requests-remaining", 0))
            _USAGE["ts"] = time.time()
        except (TypeError, ValueError):
            pass
        if resp.status_code != 200:
            print(f"[live-odds] HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        _cache.put("live_odds_games", "today", data)
        return data
    except Exception as e:
        print(f"[live-odds] fetch failed: {e}")
        return None


def _team_match(name: str, target: str) -> bool:
    """Loose team-name match. The Odds API uses full names ('New York Yankees')
    while our internal names may be partial ('Yankees')."""
    if not name or not target:
        return False
    n = name.lower()
    t = target.lower()
    return t in n or n in t


def find_game(odds_list: list, home_team: str, away_team: str) -> Optional[dict]:
    """Locate a single game in the odds list by team names."""
    if not odds_list:
        return None
    for g in odds_list:
        if _team_match(g.get("home_team", ""), home_team) and \
           _team_match(g.get("away_team", ""), away_team):
            return g
    return None


def best_moneyline(game_odds: dict, team_name: str) -> Optional[dict]:
    """Return {book, american, prob} for the best (highest payout) ML on `team_name`."""
    if not game_odds or not team_name:
        return None
    best = None
    for bm in game_odds.get("bookmakers", []):
        for mk in bm.get("markets", []):
            if mk.get("key") != "h2h":
                continue
            for o in mk.get("outcomes", []):
                if not _team_match(o.get("name", ""), team_name):
                    continue
                price = o.get("price")
                if price is None:
                    continue
                if best is None or price > best["american"]:
                    best = {
                        "book": bm.get("title", "?"),
                        "american": int(price),
                        "prob": _american_to_prob(price),
                    }
    return best


def best_runline(game_odds: dict, team_name: str, side: str) -> Optional[dict]:
    """Best run-line price. side='-1.5' for favorite, '+1.5' for dog."""
    if not game_odds or not team_name:
        return None
    target_point = -1.5 if "-" in side else 1.5
    best = None
    for bm in game_odds.get("bookmakers", []):
        for mk in bm.get("markets", []):
            if mk.get("key") != "spreads":
                continue
            for o in mk.get("outcomes", []):
                if not _team_match(o.get("name", ""), team_name):
                    continue
                if o.get("point") != target_point:
                    continue
                price = o.get("price")
                if price is None:
                    continue
                if best is None or price > best["american"]:
                    best = {
                        "book": bm.get("title", "?"),
                        "american": int(price),
                        "point": target_point,
                        "prob": _american_to_prob(price),
                    }
    return best


def best_total(game_odds: dict, side: str) -> Optional[dict]:
    """Best price + line for over/under. side='Over' or 'Under'.
    Returns the most common line across books with the best price."""
    if not game_odds:
        return None
    target = side.lower()
    best = None
    for bm in game_odds.get("bookmakers", []):
        for mk in bm.get("markets", []):
            if mk.get("key") != "totals":
                continue
            for o in mk.get("outcomes", []):
                if str(o.get("name", "")).lower() != target:
                    continue
                price = o.get("price")
                point = o.get("point")
                if price is None or point is None:
                    continue
                if best is None or price > best["american"]:
                    best = {
                        "book": bm.get("title", "?"),
                        "american": int(price),
                        "point": point,
                        "prob": _american_to_prob(price),
                    }
    return best


def edge_pct(model_prob: float, book_american: float) -> float:
    """Probability edge in percentage points: model_prob - implied_prob.

    A +5% edge means our model gives the bet 5pp more chance than the book's
    line implies. Sharp bettors look for 3pp+; 5pp is great, 8pp+ is huge.
    Anything past ~12pp usually means our model is overconfident or the line
    is stale, not a real edge."""
    try:
        implied = _american_to_prob(book_american)
        return round((float(model_prob) - implied) * 100, 1)
    except Exception:
        return 0.0


def ev_pct(model_prob: float, book_american: float) -> float:
    """Expected value per $1 staked, as a percentage. Useful for Kelly sizing."""
    try:
        if book_american >= 0:
            payout = book_american / 100.0
        else:
            payout = 100.0 / abs(book_american)
        ev = float(model_prob) * payout - (1 - float(model_prob))
        return round(ev * 100, 1)
    except Exception:
        return 0.0


def attach_game_edges(spread_lean: dict, game: dict, odds_list: list) -> dict:
    """Mutates `spread_lean`, adding live-book pricing + edge for ML and run line.

    Adds keys:
      ml_book, ml_book_odds, ml_edge_pct
      rl_book, rl_book_odds, rl_edge_pct
    Falls back silently when no matching market/team is found.
    """
    if not spread_lean or not odds_list:
        return spread_lean
    g = find_game(odds_list, game.get("home_team"), game.get("away_team"))
    if not g:
        return spread_lean

    # ── Moneyline edge ──
    ml_pick = spread_lean.get("ml_pick", "")
    ml_team = ml_pick.replace(" ML", "").strip()
    ml_best = best_moneyline(g, ml_team)
    if ml_best:
        model_p = float(spread_lean.get("ml_probability", 0)) / 100.0
        spread_lean["ml_book"] = ml_best["book"]
        spread_lean["ml_book_odds"] = ml_best["american"]
        spread_lean["ml_edge_pct"] = edge_pct(model_p, ml_best["american"])
        spread_lean["ml_ev_pct"] = ev_pct(model_p, ml_best["american"])

    # ── Run line edge ──
    rl_pick = spread_lean.get("run_line_pick", "")
    if " -1.5" in rl_pick:
        team = rl_pick.split(" -1.5")[0].strip()
        rl_best = best_runline(g, team, "-1.5")
    elif " +1.5" in rl_pick:
        team = rl_pick.split(" +1.5")[0].strip()
        rl_best = best_runline(g, team, "+1.5")
    else:
        rl_best = None
    if rl_best:
        model_p = float(spread_lean.get("run_line_probability", 0)) / 100.0
        spread_lean["rl_book"] = rl_best["book"]
        spread_lean["rl_book_odds"] = rl_best["american"]
        spread_lean["rl_edge_pct"] = edge_pct(model_p, rl_best["american"])
        spread_lean["rl_ev_pct"] = ev_pct(model_p, rl_best["american"])

    return spread_lean


def fetch_player_props(event_id: str) -> Optional[dict]:
    """Pull player-prop markets for a single game. 12hr disk cache per event.
    Returns the raw event-odds payload (with bookmakers/markets) or None.

    Quota cost: ~5 units per call (one per market). With 12hr TTL and a few
    games viewed per day, this stays well under the 500/mo free tier."""
    if not is_enabled() or not event_id:
        return None
    cached = _cache.get("live_odds_props", event_id, PROP_TTL)
    if cached is not None:
        return cached
    url = f"{BASE}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": REGIONS,
        "markets": PROP_MARKETS,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        try:
            _USAGE["used"] = int(resp.headers.get("x-requests-used", 0))
            _USAGE["remaining"] = int(resp.headers.get("x-requests-remaining", 0))
            _USAGE["ts"] = time.time()
        except (TypeError, ValueError):
            pass
        if resp.status_code != 200:
            print(f"[live-odds] props HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        _cache.put("live_odds_props", event_id, data)
        return data
    except Exception as e:
        print(f"[live-odds] props fetch failed: {e}")
        return None


# Map our internal stat_type → Odds API market key.
_PROP_MARKET_MAP = {
    "hits": "batter_hits",
    "total_bases": "batter_total_bases",
    "home_runs": "batter_home_runs",
    "rbis": "batter_rbis",
    "strikeouts": "pitcher_strikeouts",
}


def _player_match(book_name: str, our_name: str) -> bool:
    """Loose player-name match — handles 'J. Smith' vs 'John Smith' etc."""
    if not book_name or not our_name:
        return False
    a = book_name.lower().strip()
    b = our_name.lower().strip()
    if a == b:
        return True
    # Last-name match if first names initial-only.
    a_last = a.split()[-1]
    b_last = b.split()[-1]
    return a_last == b_last and a.split()[0][0] == b.split()[0][0]


def best_player_prop(event_odds: dict, market_key: str, player_name: str,
                     side: str) -> Optional[dict]:
    """Best price for a player's Over/Under on a market. side='Over'|'Under'."""
    if not event_odds:
        return None
    target = side.lower()
    best = None
    for bm in event_odds.get("bookmakers", []):
        for mk in bm.get("markets", []):
            if mk.get("key") != market_key:
                continue
            for o in mk.get("outcomes", []):
                if not _player_match(o.get("description", ""), player_name):
                    continue
                if str(o.get("name", "")).lower() != target:
                    continue
                price = o.get("price")
                point = o.get("point")
                if price is None or point is None:
                    continue
                if best is None or price > best["american"]:
                    best = {
                        "book": bm.get("title", "?"),
                        "american": int(price),
                        "point": point,
                        "prob": _american_to_prob(price),
                    }
    return best


def attach_prop_edges(prop_list: list, event_odds: dict, stat_type: str,
                      player_field: str = "player") -> int:
    """Mutate each prop dict with book_line/book_odds/book/edge_pct/ev_pct
    based on the side our model picks (OVER/UNDER). Returns count attached."""
    if not prop_list or not event_odds:
        return 0
    market_key = _PROP_MARKET_MAP.get(stat_type)
    if not market_key:
        return 0
    n = 0
    for prop in prop_list:
        side = (prop.get("pick") or "").upper()
        if side not in ("OVER", "UNDER"):
            continue
        name = prop.get(player_field) or prop.get("pitcher")
        if not name:
            continue
        side_norm = "Over" if side == "OVER" else "Under"
        best = best_player_prop(event_odds, market_key, name, side_norm)
        if not best:
            continue
        # Model probability is the prop's win % (already 0-100).
        model_p = float(prop.get("probability", 50.0)) / 100.0
        prop["book_line"] = best["point"]
        prop["book_odds"] = best["american"]
        prop["book"] = best["book"]
        prop["prop_edge_pct"] = edge_pct(model_p, best["american"])
        prop["prop_ev_pct"] = ev_pct(model_p, best["american"])
        n += 1
    return n


def attach_total_edge(total_lean: dict, game: dict, odds_list: list) -> dict:
    """Add live book over/under price + edge to total_lean."""
    if not total_lean or not odds_list:
        return total_lean
    g = find_game(odds_list, game.get("home_team"), game.get("away_team"))
    if not g:
        return total_lean

    over_best = best_total(g, "Over")
    under_best = best_total(g, "Under")
    if not over_best or not under_best:
        return total_lean

    book_line = over_best["point"]
    projected = float(total_lean.get("projected_runs", 0) or 0)

    import math
    sigma = 0.45 * math.sqrt(max(1.0, projected))
    z = (projected - book_line) / sigma if sigma > 0 else 0
    p_over = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    p_over = max(0.05, min(0.95, p_over))
    p_under = 1.0 - p_over

    total_lean["book_line"] = book_line
    total_lean["over_book"] = over_best["book"]
    total_lean["over_odds"] = over_best["american"]
    total_lean["under_book"] = under_best["book"]
    total_lean["under_odds"] = under_best["american"]
    total_lean["model_p_over"] = round(p_over * 100, 1)
    total_lean["over_edge_pct"] = edge_pct(p_over, over_best["american"])
    total_lean["under_edge_pct"] = edge_pct(p_under, under_best["american"])
    total_lean["over_ev_pct"] = ev_pct(p_over, over_best["american"])
    total_lean["under_ev_pct"] = ev_pct(p_under, under_best["american"])

    if p_over > p_under:
        total_lean["pick"] = f"OVER {book_line}"
        total_lean["pick_side"] = "OVER"
        total_lean["pick_book"] = over_best["book"]
        total_lean["pick_odds"] = over_best["american"]
        total_lean["pick_edge_pct"] = total_lean["over_edge_pct"]
        total_lean["pick_ev_pct"] = total_lean["over_ev_pct"]
        total_lean["probability"] = round(p_over * 100, 1)
    else:
        total_lean["pick"] = f"UNDER {book_line}"
        total_lean["pick_side"] = "UNDER"
        total_lean["pick_book"] = under_best["book"]
        total_lean["pick_odds"] = under_best["american"]
        total_lean["pick_edge_pct"] = total_lean["under_edge_pct"]
        total_lean["pick_ev_pct"] = total_lean["under_ev_pct"]
        total_lean["probability"] = round(p_under * 100, 1)

    return total_lean
