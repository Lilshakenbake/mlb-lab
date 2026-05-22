import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pybaseball import statcast_batter, statcast_pitcher, playerid_lookup

try:
    import pybaseball
    _pyb_cache_dir = os.getenv("PYBASEBALL_CACHE_DIR")
    if _pyb_cache_dir:
        try:
            os.makedirs(_pyb_cache_dir, exist_ok=True)
            pybaseball.cache.config.cache_directory = _pyb_cache_dir
        except Exception:
            pass
    pybaseball.cache.enable()
except Exception:
    pass

from src import cache as _cache

BASE_URL = "https://statsapi.mlb.com/api/v1"

# In-process memo (avoids re-reading disk inside one request)
BatterProfileCache = {}
PitcherProfileCache = {}
GameHittersCache = {}
WeatherCache = {}
PlayerIdCache = {}

# TTLs (seconds). Player stats only change once a day after games end, so
# 12h is plenty. Schedule/lineups/weather rotate faster.
SCHEDULE_TTL = 5 * 60
LINEUP_TTL = 15 * 60
WEATHER_TTL = 60 * 60
PROFILE_TTL = 12 * 60 * 60
ROSTER_TTL = 6 * 60 * 60

STADIUMS = {
    "Arizona Diamondbacks": {"lat": 33.4453, "lon": -112.0667, "indoor": True, "park": "Chase Field"},
    "Atlanta Braves": {"lat": 33.8908, "lon": -84.4677, "indoor": False, "park": "Truist Park"},
    "Baltimore Orioles": {"lat": 39.2839, "lon": -76.6217, "indoor": False, "park": "Camden Yards"},
    "Boston Red Sox": {"lat": 42.3467, "lon": -71.0972, "indoor": False, "park": "Fenway Park"},
    "Chicago Cubs": {"lat": 41.9484, "lon": -87.6553, "indoor": False, "park": "Wrigley Field"},
    "Chicago White Sox": {"lat": 41.8300, "lon": -87.6338, "indoor": False, "park": "Guaranteed Rate Field"},
    "Cincinnati Reds": {"lat": 39.0979, "lon": -84.5082, "indoor": False, "park": "Great American Ball Park"},
    "Cleveland Guardians": {"lat": 41.4962, "lon": -81.6852, "indoor": False, "park": "Progressive Field"},
    "Colorado Rockies": {"lat": 39.7559, "lon": -104.9942, "indoor": False, "park": "Coors Field"},
    "Detroit Tigers": {"lat": 42.3390, "lon": -83.0485, "indoor": False, "park": "Comerica Park"},
    "Houston Astros": {"lat": 29.7573, "lon": -95.3555, "indoor": True, "park": "Minute Maid Park"},
    "Kansas City Royals": {"lat": 39.0517, "lon": -94.4803, "indoor": False, "park": "Kauffman Stadium"},
    "Los Angeles Angels": {"lat": 33.8003, "lon": -117.8827, "indoor": False, "park": "Angel Stadium"},
    "Los Angeles Dodgers": {"lat": 34.0739, "lon": -118.2400, "indoor": False, "park": "Dodger Stadium"},
    "Miami Marlins": {"lat": 25.7781, "lon": -80.2197, "indoor": True, "park": "loanDepot park"},
    "Milwaukee Brewers": {"lat": 43.0280, "lon": -87.9712, "indoor": True, "park": "American Family Field"},
    "Minnesota Twins": {"lat": 44.9817, "lon": -93.2776, "indoor": False, "park": "Target Field"},
    "New York Mets": {"lat": 40.7571, "lon": -73.8458, "indoor": False, "park": "Citi Field"},
    "New York Yankees": {"lat": 40.8296, "lon": -73.9262, "indoor": False, "park": "Yankee Stadium"},
    "Athletics": {"lat": 36.0908, "lon": -115.1830, "indoor": False, "park": "Athletics Park"},
    "Oakland Athletics": {"lat": 37.7516, "lon": -122.2005, "indoor": False, "park": "Oakland Coliseum"},
    "Philadelphia Phillies": {"lat": 39.9061, "lon": -75.1665, "indoor": False, "park": "Citizens Bank Park"},
    "Pittsburgh Pirates": {"lat": 40.4469, "lon": -80.0057, "indoor": False, "park": "PNC Park"},
    "San Diego Padres": {"lat": 32.7073, "lon": -117.1566, "indoor": False, "park": "Petco Park"},
    "San Francisco Giants": {"lat": 37.7786, "lon": -122.3893, "indoor": False, "park": "Oracle Park"},
    "Seattle Mariners": {"lat": 47.5914, "lon": -122.3325, "indoor": True, "park": "T-Mobile Park"},
    "St. Louis Cardinals": {"lat": 38.6226, "lon": -90.1928, "indoor": False, "park": "Busch Stadium"},
    "Tampa Bay Rays": {"lat": 27.7683, "lon": -82.6534, "indoor": True, "park": "Tropicana Field"},
    "Texas Rangers": {"lat": 32.7473, "lon": -97.0827, "indoor": True, "park": "Globe Life Field"},
    "Toronto Blue Jays": {"lat": 43.6414, "lon": -79.3894, "indoor": True, "park": "Rogers Centre"},
    "Washington Nationals": {"lat": 38.8730, "lon": -77.0074, "indoor": False, "park": "Nationals Park"},
}


def _split_name(full_name):
    parts = full_name.strip().split()
    if len(parts) < 2:
        return None, None

    suffixes = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}
    last_part = parts[-1].lower()

    if last_part in suffixes and len(parts) >= 3:
        return parts[0], parts[-2]
    return parts[0], parts[-1]


def _lookup_player_id(full_name):
    if full_name in PlayerIdCache:
        return PlayerIdCache[full_name]

    # Positive hits cache for 30d (mlbam ids don't change). Negative ("not
    # found") results cache for only 24h so a mid-season signing or
    # mid-offseason trade can't be locked out of the model for a month.
    cached = _cache.get("player_id", full_name, ttl_seconds=30 * 24 * 3600)
    if cached is not None:
        if cached.get("id") is None:
            # Negative result — only honor it if it's recent.
            cached_recent = _cache.get("player_id", full_name, ttl_seconds=24 * 3600)
            if cached_recent is None:
                cached = None  # fall through and re-lookup
        if cached is not None:
            PlayerIdCache[full_name] = (cached.get("id"), cached.get("error"))
            return PlayerIdCache[full_name]

    first, last = _split_name(full_name)
    if not first or not last:
        result = (None, "Enter full name (first and last)")
        PlayerIdCache[full_name] = result
        return result

    try:
        players = playerid_lookup(last, first)
    except Exception as e:
        return None, f"Player lookup failed: {e}"

    if players.empty:
        result = (None, "Player not found")
        PlayerIdCache[full_name] = result
        _cache.put("player_id", full_name, {"id": None, "error": "Player not found"})
        return result

    pid = int(players.iloc[0]["key_mlbam"])
    PlayerIdCache[full_name] = (pid, None)
    _cache.put("player_id", full_name, {"id": pid, "error": None})
    return pid, None


from zoneinfo import ZoneInfo  # add this at the top of the file if not already there

def get_todays_games():
    eastern_now = datetime.now(ZoneInfo("America/New_York"))
    today = eastern_now.strftime("%Y-%m-%d")

    cached = _cache.get("schedule", today, SCHEDULE_TTL)
    if cached is not None:
        return cached

    url = f"{BASE_URL}/schedule"
    params = {
        "sportId": 1,
        "date": today,
        "hydrate": "probablePitcher,team,linescore"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    games = []

    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            away = game["teams"]["away"]
            home = game["teams"]["home"]
            ls = game.get("linescore") or {}

            games.append({
                "gamePk": game.get("gamePk"),
                "away_team": away["team"]["name"],
                "home_team": home["team"]["name"],
                "away_team_id": away["team"]["id"],
                "home_team_id": home["team"]["id"],
                "away_pitcher": away.get("probablePitcher", {}).get("fullName", "TBD"),
                "home_pitcher": home.get("probablePitcher", {}).get("fullName", "TBD"),
                "game_time": game.get("gameDate", ""),
                "status": game["status"]["detailedState"],
                "current_inning": ls.get("currentInning"),
                "inning_state": ls.get("inningState"),
            })

    games.sort(key=lambda g: g.get("game_time", ""))
    _cache.put("schedule", today, games)
    return games

def get_confirmed_starting_hitters(game_pk, side):
    cache_key = f"{game_pk}_{side}"
    if cache_key in GameHittersCache:
        return GameHittersCache[cache_key]

    disk = _cache.get("lineup", cache_key, LINEUP_TTL)
    if disk is not None:
        GameHittersCache[cache_key] = disk
        return disk

    try:
        url = f"{BASE_URL}/game/{game_pk}/feed/live"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        team_players = data["liveData"]["boxscore"]["teams"][side]["players"]

        hitters = []
        for player_data in team_players.values():
            batting_order = player_data.get("battingOrder")
            position_abbr = (
                player_data.get("position", {}).get("abbreviation")
                or player_data.get("person", {}).get("primaryPosition", {}).get("abbreviation")
                or ""
            )

            if batting_order and position_abbr != "P":
                hitters.append((int(batting_order), player_data["person"]["fullName"]))

        hitters.sort(key=lambda x: x[0])
        names = [name for _, name in hitters]

        GameHittersCache[cache_key] = names
        _cache.put("lineup", cache_key, names)
        return names

    except Exception:
        # Don't memoize transient fetch failures — if the MLB API blips
        # we'd otherwise lock in "no lineup" for the process lifetime,
        # silently disabling the lineup gate for that game. Returning []
        # without caching lets the next caller retry.
        return []


def get_team_active_hitters(team_id):
    cache_key = str(team_id)
    disk = _cache.get("roster", cache_key, ROSTER_TTL)
    if disk is not None:
        return disk

    try:
        url = f"{BASE_URL}/teams/{team_id}/roster"
        params = {"rosterType": "active"}

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        hitters = []
        for row in data.get("roster", []):
            position = row.get("position", {}).get("abbreviation", "")
            if position != "P":
                hitters.append(row["person"]["fullName"])

        _cache.put("roster", cache_key, hitters)
        return hitters

    except Exception:
        return []


def _get_batter_statcast(player_name, days_back=90):
    player_id, error = _lookup_player_id(player_name)
    if error:
        return None, error

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    df = statcast_batter(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        player_id
    )

    if df.empty:
        return None, "No Statcast data found"

    return df, None


def _get_pitcher_statcast(pitcher_name, days_back=60):
    pitcher_id, error = _lookup_player_id(pitcher_name)
    if error:
        return None, error

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    df = statcast_pitcher(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        pitcher_id
    )

    if df.empty:
        return None, "No pitcher Statcast data found"

    return df, None


def _safe_hand(df, col):
    """Return the most common L/R/S code in column `col` of df, or None."""
    if col not in df.columns:
        return None
    try:
        s = df[col].dropna()
        if s.empty:
            return None
        v = s.mode().iloc[0]
        v = str(v).upper().strip()
        return v if v in ("L", "R", "S") else None
    except Exception:
        return None


def get_last5_hitter_profile(player_name):
    if player_name in BatterProfileCache:
        return BatterProfileCache[player_name], None

    disk = _cache.get("hitter_profile", player_name, PROFILE_TTL)
    if disk is not None:
        BatterProfileCache[player_name] = disk
        return disk, None

    try:
        df, error = _get_batter_statcast(player_name)
        if error:
            return None, error

        grouped = df.groupby("game_date")

        # ── Pull season + L15 windows from the same DF (90d pull) ─────────
        # We compute every per-game series across the FULL season window,
        # then build the projection as a SEASON-WEIGHTED blend with L15 as
        # a smaller hot/cold tilt. 35% L15 / 65% season — a star with a
        # 74-TB season pedigree should NOT get dragged below the line by
        # a 2-week cold spell, and a journeyman shouldn't get inflated by
        # 15 lucky games. True-talent dominates; recency only nudges.
        hits_full = grouped["events"].apply(
            lambda x: x.isin(["single", "double", "triple", "home_run"]).sum()
        )
        hr_full = grouped["events"].apply(
            lambda x: x.isin(["home_run"]).sum()
        )
        tb_full = grouped["events"].apply(
            lambda x: (
                (x == "single").sum() * 1
                + (x == "double").sum() * 2
                + (x == "triple").sum() * 3
                + (x == "home_run").sum() * 4
            )
        )
        if "rbi" in df.columns:
            rbi_col = pd.to_numeric(df["rbi"], errors="coerce").fillna(0)
            temp = df.copy()
            temp["rbi_num"] = rbi_col
            rbi_full = temp.groupby("game_date")["rbi_num"].sum()
        else:
            rbi_full = hits_full * 0.35 + hr_full * 0.65 + tb_full * 0.10

        if len(hits_full) == 0:
            return None, "No recent games found"

        # L15 = last 15 games; "season" = everything in the 90d window
        # (typically 45-70 games depending on rest days / IL stints).
        def _season_l15_blend(full_series, l15_weight=0.35):
            """Blend last-15 form with full-window baseline.

            Falls back gracefully when sample is thin: if we have <15
            games we just use what we have for L15; if we have <5 we
            skip the blend and use the full series only.
            """
            if len(full_series) == 0:
                return 0.0
            season_mean = float(full_series.mean())
            if len(full_series) < 5:
                return season_mean
            l15_mean = float(full_series.tail(15).mean())
            return l15_weight * l15_mean + (1.0 - l15_weight) * season_mean

        # Per-stat blended means (these REPLACE the old weighted_mean output
        # but keep the same field names so predict.py needs no changes).
        hits_blend = _season_l15_blend(hits_full)
        hr_blend = _season_l15_blend(hr_full)
        tb_blend = _season_l15_blend(tb_full)
        rbi_blend = _season_l15_blend(rbi_full)

        # Hot-ratio rebuilt: L15 vs season baseline (was L3 vs L8 — too
        # noisy, often false-positive on a 2-game heater). L15 vs season
        # is a real form signal.
        season_hits = float(hits_full.mean()) or 0.0
        l15_hits = float(hits_full.tail(15).mean()) if len(hits_full) >= 5 else season_hits
        hot_ratio = (l15_hits / season_hits) if season_hits > 0 else 1.0

        # Diagnostic fields so the dashboard / future tuning can see the
        # raw L15 and season numbers behind the blend.
        diag_hits_l15 = l15_hits
        diag_hits_season = season_hits

        # Keep an alias so downstream code referencing these names stays sane.
        hits_by_game = hits_full
        hr_by_game = hr_full
        tb_by_game = tb_full

        # ── Statcast expected stats (xStats): strip out luck. ─────────────
        # `type` is "X" for batted balls, "S"/"B" for swings/balls. xBA and
        # xwOBA only exist for batted balls; PA-level wOBA exists everywhere.
        bbe = df[df["type"] == "X"] if "type" in df.columns else df
        xba_per_bbe = pd.to_numeric(
            bbe.get("estimated_ba_using_speedangle"), errors="coerce"
        ).dropna() if "estimated_ba_using_speedangle" in bbe.columns else None
        xwoba_per_pa = pd.to_numeric(
            df.get("estimated_woba_using_speedangle"), errors="coerce"
        ).dropna() if "estimated_woba_using_speedangle" in df.columns else None

        # bbe/pa per game: blend L15 with full window like the counting stats
        # so a guy whose lineup spot changed (3→8) doesn't get a stale PA est.
        if "type" in df.columns:
            bbe_series = (df["type"] == "X").groupby(df["game_date"]).sum()
            bbe_per_game = _season_l15_blend(bbe_series)
        else:
            bbe_per_game = None
        pa_series = df.groupby("game_date").size()
        pa_per_game = _season_l15_blend(pa_series)

        xhits_avg = (
            float(xba_per_bbe.mean()) * float(bbe_per_game)
            if xba_per_bbe is not None and len(xba_per_bbe) and bbe_per_game else None
        )
        # xwOBA scales roughly to slugging; *1.7 rough TB-per-PA conversion.
        xtb_avg = (
            float(xwoba_per_pa.mean()) * pa_per_game * 1.7
            if xwoba_per_pa is not None and len(xwoba_per_pa) else None
        )

        # Barrel rate (Statcast quality bucket = 6) and hard-hit rate (≥95mph EV).
        barrel_rate = None
        hard_hit_rate = None
        if "launch_speed_angle" in bbe.columns and len(bbe) > 0:
            lsa = pd.to_numeric(bbe["launch_speed_angle"], errors="coerce").dropna()
            if len(lsa) > 0:
                barrel_rate = float((lsa == 6).sum()) / float(len(lsa))
        # Fly ball rate (launch angle ≥ 25°) — direct HR signal beyond barrels.
        # Even non-barrel fly balls leave the yard in HR-friendly parks.
        fly_ball_rate = None
        if "launch_angle" in bbe.columns and len(bbe) > 0:
            la = pd.to_numeric(bbe["launch_angle"], errors="coerce").dropna()
            if len(la) > 0:
                fly_ball_rate = float((la >= 25).sum()) / float(len(la))
        if "launch_speed" in bbe.columns and len(bbe) > 0:
            ls = pd.to_numeric(bbe["launch_speed"], errors="coerce").dropna()
            if len(ls) > 0:
                hard_hit_rate = float((ls >= 95).sum()) / float(len(ls))

        # ── Handedness splits — biggest single accuracy boost. ────────────
        # A LHH facing a LHP loses ~25-30% of expected production. Built from
        # the same DF, no new HTTP. Falls back to None when sample is too thin
        # to be meaningful (so the predictor knows to skip the adjustment).
        splits = {"vs_L": {}, "vs_R": {}}
        if "p_throws" in df.columns:
            for hand_key, hand_val in [("vs_L", "L"), ("vs_R", "R")]:
                hand_df = df[df["p_throws"] == hand_val]
                if len(hand_df) < 25:  # too thin; skip
                    continue
                hand_grouped = hand_df.groupby("game_date")
                h_hits = hand_grouped["events"].apply(
                    lambda x: x.isin(["single", "double", "triple", "home_run"]).sum()
                )
                h_hr = hand_grouped["events"].apply(
                    lambda x: x.isin(["home_run"]).sum()
                )
                h_tb = hand_grouped["events"].apply(
                    lambda x: (
                        (x == "single").sum() * 1
                        + (x == "double").sum() * 2
                        + (x == "triple").sum() * 3
                        + (x == "home_run").sum() * 4
                    )
                )
                splits[hand_key] = {
                    "hits_avg": round(float(h_hits.mean()), 2) if len(h_hits) else None,
                    "hr_avg": round(float(h_hr.mean()), 3) if len(h_hr) else None,
                    "tb_avg": round(float(h_tb.mean()), 2) if len(h_tb) else None,
                    "samples": int(len(hand_df)),
                }

        # Pure extra-base power — separates singles hitters from sluggers.
        # iso = (TB - hits) / hits = avg extra bases per hit.
        hits_mean = float(hits_by_game.mean()) or 0.0
        tb_mean = float(tb_by_game.mean()) or 0.0
        iso_power = (tb_mean - hits_mean) / hits_mean if hits_mean > 0 else 0.0

        # Contact rate = balls in play / plate appearances. High = contact
        # hitter (great for Hits), low = K-prone (great for HR/TB only).
        contact_rate = float(bbe_per_game) / pa_per_game if (
            bbe_per_game is not None and pa_per_game and pa_per_game > 0
        ) else None

        profile = {
            # Headline averages = L15 (60%) + season (40%) blend.
            "hits_avg": round(hits_blend, 2),
            "hr_avg": round(hr_blend, 2),
            "tb_avg": round(tb_blend, 2),
            "rbi_avg": round(rbi_blend, 2),
            # Stds still computed over full window for stability.
            "hits_std": round(float(hits_by_game.std() or 0.0), 2),
            "tb_std": round(float(tb_by_game.std() or 0.0), 2),
            "iso_power": round(iso_power, 3),
            "contact_rate": round(contact_rate, 3) if contact_rate is not None else None,
            "hand": _safe_hand(df, "stand"),  # L/R/S
            "hot_ratio": round(hot_ratio, 2),  # now L15/season, not L3/L8
            "games_used": int(len(hits_by_game)),
            # Diagnostics — surface so we can audit blend behavior later.
            "hits_l15_avg": round(diag_hits_l15, 2),
            "hits_season_avg": round(diag_hits_season, 2),
            "games_l15": int(min(15, len(hits_by_game))),
            "games_season": int(len(hits_by_game)),
            # xStats — luck-stripped projections for the predictor.
            "xhits_avg": round(xhits_avg, 3) if xhits_avg is not None else None,
            "xtb_avg": round(xtb_avg, 3) if xtb_avg is not None else None,
            "barrel_rate": round(barrel_rate, 3) if barrel_rate is not None else None,
            "fly_ball_rate": round(fly_ball_rate, 3) if fly_ball_rate is not None else None,
            "hard_hit_rate": round(hard_hit_rate, 3) if hard_hit_rate is not None else None,
            "bbe_per_game": round(float(bbe_per_game), 2) if bbe_per_game is not None else None,
            "splits": splits,  # vs_L / vs_R handedness performance
        }

        BatterProfileCache[player_name] = profile
        _cache.put("hitter_profile", player_name, profile)
        return profile, None

    except Exception as e:
        return None, f"Stats lookup failed: {e}"


def get_last5_pitcher_profile(pitcher_name):
    if pitcher_name in PitcherProfileCache:
        return PitcherProfileCache[pitcher_name], None

    disk = _cache.get("pitcher_profile", pitcher_name, PROFILE_TTL)
    if disk is not None:
        PitcherProfileCache[pitcher_name] = disk
        return disk, None

    try:
        df, error = _get_pitcher_statcast(pitcher_name)
        if error:
            return None, error

        grouped = df.groupby("game_date")

        # Full window (60d ≈ 11-12 starts) = "season" baseline.
        hits_allowed_by_game = grouped["events"].apply(
            lambda x: x.isin(["single", "double", "triple", "home_run"]).sum()
        )
        strikeouts_by_game = grouped["events"].apply(
            lambda x: x.isin(["strikeout", "strikeout_double_play"]).sum()
        )

        if len(strikeouts_by_game) == 0:
            return None, "No recent pitcher games found"

        # ── L5-starts vs season blend for pitchers ───────────────────────
        # Pitchers go every 5th day, so L15 doesn't make sense — L5 starts
        # ≈ 25 days of starts. Same logic as hitters but smaller recent
        # window. Still 60/40 weighting on recent vs full.
        def _season_l5_blend(full_series, recent_weight=0.60):
            if len(full_series) == 0:
                return 0.0
            season_mean = float(full_series.mean())
            if len(full_series) < 4:
                return season_mean
            l5_mean = float(full_series.tail(5).mean())
            return recent_weight * l5_mean + (1.0 - recent_weight) * season_mean

        # ── Power-allowed metrics — separate aces from contact-friendly arms ──
        hr_allowed_by_game = grouped["events"].apply(
            lambda x: x.isin(["home_run"]).sum()
        )
        tb_allowed_by_game = grouped["events"].apply(
            lambda x: (
                (x == "single").sum() * 1
                + (x == "double").sum() * 2
                + (x == "triple").sum() * 3
                + (x == "home_run").sum() * 4
            )
        )

        bbe_p = df[df["type"] == "X"] if "type" in df.columns else df
        xwoba_allowed = pd.to_numeric(
            df.get("estimated_woba_using_speedangle"), errors="coerce"
        ).dropna() if "estimated_woba_using_speedangle" in df.columns else None
        hard_hit_allowed = None
        barrel_allowed = None
        if "launch_speed" in bbe_p.columns and len(bbe_p) > 0:
            ls = pd.to_numeric(bbe_p["launch_speed"], errors="coerce").dropna()
            if len(ls) > 0:
                hard_hit_allowed = float((ls >= 95).sum()) / float(len(ls))
        if "launch_speed_angle" in bbe_p.columns and len(bbe_p) > 0:
            lsa = pd.to_numeric(bbe_p["launch_speed_angle"], errors="coerce").dropna()
            if len(lsa) > 0:
                barrel_allowed = float((lsa == 6).sum()) / float(len(lsa))

        # ── First-inning metrics for NRFI predictions ─────────────────────
        # Compute runs allowed per first inning across recent starts. We use
        # bat_score progression since Statcast doesn't have a direct "runs"
        # field — last pitch's post_bat_score minus first pitch's bat_score
        # in that (game, inning=1) slice = runs scored that half-inning.
        first_inning_runs_avg = None
        nrfi_solo_rate = None  # % of starts where pitcher held the 1st scoreless
        if "inning" in df.columns and "bat_score" in df.columns and "post_bat_score" in df.columns:
            inn1 = df[df["inning"] == 1]
            if len(inn1) > 0:
                runs_per_start = []
                sort_cols = [c for c in ("at_bat_number", "pitch_number") if c in inn1.columns]
                for _, gdf in inn1.groupby("game_date"):
                    gdf2 = gdf.sort_values(sort_cols) if sort_cols else gdf
                    try:
                        first_score = float(pd.to_numeric(gdf2["bat_score"], errors="coerce").iloc[0])
                        last_score = float(pd.to_numeric(gdf2["post_bat_score"], errors="coerce").iloc[-1])
                        runs = max(0.0, last_score - first_score)
                        runs_per_start.append(runs)
                    except Exception:
                        continue
                if runs_per_start:
                    # Use full window for NRFI rate — bigger sample = more
                    # stable scoreless% (was tail-8). Recency form gets
                    # captured by the L5 blend on the counting stats above.
                    runs_per_start = runs_per_start[-12:]
                    first_inning_runs_avg = round(sum(runs_per_start) / len(runs_per_start), 2)
                    nrfi_solo_rate = round(
                        sum(1 for r in runs_per_start if r == 0) / len(runs_per_start), 3
                    )

        # ── Fastball usage + velocity (HR-prone arms throw a lot of slow heat) ─
        fb_pct = None
        fb_velo = None
        if "pitch_type" in df.columns:
            pt = df["pitch_type"].dropna().astype(str)
            if len(pt) >= 50:
                # FF=4-seam, SI=sinker, FC=cutter, FT=2-seam — all "fastballs".
                fb_mask_full = df["pitch_type"].astype(str).isin(["FF", "SI", "FC", "FT"])
                fb_pct = round(float(fb_mask_full.mean()), 3)
                if "release_speed" in df.columns:
                    rs = pd.to_numeric(
                        df.loc[fb_mask_full, "release_speed"],
                        errors="coerce",
                    ).dropna()
                    if len(rs) >= 30:
                        fb_velo = round(float(rs.mean()), 1)

        # ── Handedness splits — pitcher's performance vs L vs R hitters ───
        pitcher_splits = {"vs_L": {}, "vs_R": {}}
        if "stand" in df.columns:
            for hand_key, hand_val in [("vs_L", "L"), ("vs_R", "R")]:
                hand_df = df[df["stand"] == hand_val]
                if len(hand_df) < 25:
                    continue
                hg = hand_df.groupby("game_date")
                h_hits = hg["events"].apply(
                    lambda x: x.isin(["single", "double", "triple", "home_run"]).sum()
                )
                h_hr = hg["events"].apply(
                    lambda x: x.isin(["home_run"]).sum()
                )
                pitcher_splits[hand_key] = {
                    "hits_allowed_avg": round(float(h_hits.mean()), 2) if len(h_hits) else None,
                    "hr_allowed_avg": round(float(h_hr.mean()), 3) if len(h_hr) else None,
                    "samples": int(len(hand_df)),
                }

        profile = {
            # L5-starts (60%) + season (40%) blend.
            "hits_allowed_avg": round(_season_l5_blend(hits_allowed_by_game), 2),
            "tb_allowed_avg": round(_season_l5_blend(tb_allowed_by_game), 2),
            "hr_allowed_avg": round(_season_l5_blend(hr_allowed_by_game), 3),
            "strikeouts_avg": round(_season_l5_blend(strikeouts_by_game), 2),
            # Diagnostics for tuning audit.
            "strikeouts_l5_avg": round(
                float(strikeouts_by_game.tail(5).mean()) if len(strikeouts_by_game) >= 4
                else float(strikeouts_by_game.mean()), 2
            ),
            "strikeouts_season_avg": round(float(strikeouts_by_game.mean()), 2),
            "starts_used_l5": int(min(5, len(strikeouts_by_game))),
            "starts_used_season": int(len(strikeouts_by_game)),
            "k_std": round(float(strikeouts_by_game.std() or 0.0), 2),
            "xwoba_allowed": round(float(xwoba_allowed.mean()), 3) if xwoba_allowed is not None and len(xwoba_allowed) else None,
            "hard_hit_allowed": round(hard_hit_allowed, 3) if hard_hit_allowed is not None else None,
            "barrel_allowed": round(barrel_allowed, 3) if barrel_allowed is not None else None,
            "hand": _safe_hand(df, "p_throws"),  # L/R
            "games_used": int(len(strikeouts_by_game)),
            "first_inning_runs_avg": first_inning_runs_avg,
            "nrfi_solo_rate": nrfi_solo_rate,
            "splits": pitcher_splits,
            "fb_pct": fb_pct,
            "fb_velo": fb_velo,
        }

        PitcherProfileCache[pitcher_name] = profile
        _cache.put("pitcher_profile", pitcher_name, profile)
        return profile, None

    except Exception as e:
        return None, f"Pitcher stats lookup failed: {e}"


def get_game_weather(game):
    home_team = game.get("home_team")
    game_time = game.get("game_time", "")
    cache_key = f"{home_team}_{game_time}"

    if cache_key in WeatherCache:
        return WeatherCache[cache_key]

    disk = _cache.get("weather", cache_key, WEATHER_TTL)
    if disk is not None:
        WeatherCache[cache_key] = disk
        return disk

    stadium = STADIUMS.get(home_team)
    if not stadium:
        result = {
            "park": "Unknown Park",
            "is_indoor": False,
            "temperature": 70,
            "wind_speed": 0,
            "wind_direction": 0,
            "weather_note": "No stadium mapping available",
        }
        WeatherCache[cache_key] = result
        return result

    if stadium["indoor"]:
        result = {
            "park": stadium["park"],
            "is_indoor": True,
            "temperature": 72,
            "wind_speed": 0,
            "wind_direction": 0,
            "weather_note": "Indoor / roof-closed environment",
        }
        WeatherCache[cache_key] = result
        _cache.put("weather", cache_key, result)
        return result

    try:
        first_pitch = datetime.fromisoformat(game_time.replace("Z", "+00:00"))
        target_hour = first_pitch.strftime("%Y-%m-%dT%H:00")

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": stadium["lat"],
            "longitude": stadium["lon"],
            "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
            "timezone": "auto",
            "forecast_days": 2,
            "wind_speed_unit": "mph",
            "temperature_unit": "fahrenheit",
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        times = data["hourly"]["time"]
        idx = times.index(target_hour) if target_hour in times else 0

        temp = data["hourly"]["temperature_2m"][idx]
        wind_speed = data["hourly"]["wind_speed_10m"][idx]
        wind_dir = data["hourly"]["wind_direction_10m"][idx]

        result = {
            "park": stadium["park"],
            "is_indoor": False,
            "temperature": temp,
            "wind_speed": wind_speed,
            "wind_direction": wind_dir,
            "weather_note": f"Outdoor weather loaded for {stadium['park']}",
        }
        WeatherCache[cache_key] = result
        _cache.put("weather", cache_key, result)
        return result

    except Exception:
        result = {
            "park": stadium["park"],
            "is_indoor": False,
            "temperature": 70,
            "wind_speed": 0,
            "wind_direction": 0,
            "weather_note": "Weather lookup fallback used",
        }
        WeatherCache[cache_key] = result
        return result