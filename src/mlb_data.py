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

    cached = _cache.get("player_id", full_name, ttl_seconds=30 * 24 * 3600)
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
        "hydrate": "probablePitcher,team"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    games = []

    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            away = game["teams"]["away"]
            home = game["teams"]["home"]

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
        GameHittersCache[cache_key] = []
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


def _get_batter_statcast(player_name, days_back=45):
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

        # Bigger window (8 games) cuts single-game noise without any new HTTP.
        hits_by_game = grouped["events"].apply(
            lambda x: x.isin(["single", "double", "triple", "home_run"]).sum()
        ).tail(8)

        hr_by_game = grouped["events"].apply(
            lambda x: x.isin(["home_run"]).sum()
        ).tail(8)

        tb_by_game = grouped["events"].apply(
            lambda x: (
                (x == "single").sum() * 1
                + (x == "double").sum() * 2
                + (x == "triple").sum() * 3
                + (x == "home_run").sum() * 4
            )
        ).tail(8)

        if "rbi" in df.columns:
            rbi_col = pd.to_numeric(df["rbi"], errors="coerce").fillna(0)
            temp = df.copy()
            temp["rbi_num"] = rbi_col
            rbi_by_game = temp.groupby("game_date")["rbi_num"].sum().tail(8)
        else:
            rbi_by_game = (hits_by_game * 0.35 + hr_by_game * 0.65 + tb_by_game * 0.10).tail(8)

        if len(hits_by_game) == 0:
            return None, "No recent games found"

        # ── Recency weighting: last 3 games count 2x ──────────────────────
        # A guy on a 7-for-9 tear should NOT be averaged with a cold week.
        # Weights: last 3 games × 2.0, prior 5 × 1.0  → catches streaks early.
        def _weighted_mean(series):
            if len(series) == 0:
                return 0.0
            n = len(series)
            weights = [2.0 if i >= max(0, n - 3) else 1.0 for i in range(n)]
            total_weight = sum(weights)
            return sum(float(v) * w for v, w in zip(series, weights)) / total_weight

        # Hot-streak: last 3 vs overall window. >1 = hot, <1 = cold.
        last3 = hits_by_game.tail(3)
        hits_overall = float(hits_by_game.mean()) or 0.0
        last3_mean = float(last3.mean()) if len(last3) else hits_overall
        hot_ratio = (last3_mean / hits_overall) if hits_overall > 0 else 1.0

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

        bbe_per_game = (
            (df["type"] == "X").groupby(df["game_date"]).sum().tail(8).mean()
            if "type" in df.columns else None
        )
        pa_per_game = float(df.groupby("game_date").size().tail(8).mean())

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
            "hits_avg": round(_weighted_mean(hits_by_game), 2),
            "hr_avg": round(_weighted_mean(hr_by_game), 2),
            "tb_avg": round(_weighted_mean(tb_by_game), 2),
            "rbi_avg": round(_weighted_mean(rbi_by_game), 2),
            "hits_std": round(float(hits_by_game.std() or 0.0), 2),
            "tb_std": round(float(tb_by_game.std() or 0.0), 2),
            "iso_power": round(iso_power, 3),
            "contact_rate": round(contact_rate, 3) if contact_rate is not None else None,
            "hand": _safe_hand(df, "stand"),  # L/R/S
            "hot_ratio": round(hot_ratio, 2),
            "games_used": int(len(hits_by_game)),
            # xStats — luck-stripped projections for the predictor.
            "xhits_avg": round(xhits_avg, 3) if xhits_avg is not None else None,
            "xtb_avg": round(xtb_avg, 3) if xtb_avg is not None else None,
            "barrel_rate": round(barrel_rate, 3) if barrel_rate is not None else None,
            "fly_ball_rate": round(fly_ball_rate, 3) if fly_ball_rate is not None else None,
            "hard_hit_rate": round(hard_hit_rate, 3) if hard_hit_rate is not None else None,
            "bbe_per_game": round(float(bbe_per_game), 2) if bbe_per_game is not None else None,
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

        hits_allowed_by_game = grouped["events"].apply(
            lambda x: x.isin(["single", "double", "triple", "home_run"]).sum()
        ).tail(8)

        strikeouts_by_game = grouped["events"].apply(
            lambda x: x.isin(["strikeout", "strikeout_double_play"]).sum()
        ).tail(8)

        if len(strikeouts_by_game) == 0:
            return None, "No recent pitcher games found"

        # ── Recency weighting for pitchers: last 3 starts count 2x ────────
        # A pitcher coming off two gems vs his last 8 = different threat.
        def _weighted_mean(series):
            if len(series) == 0:
                return 0.0
            n = len(series)
            weights = [2.0 if i >= max(0, n - 3) else 1.0 for i in range(n)]
            total_weight = sum(weights)
            return sum(float(v) * w for v, w in zip(series, weights)) / total_weight

        # ── Power-allowed metrics — separate aces from contact-friendly arms ──
        # Pulled from the SAME Statcast DF, no new HTTP.
        hr_allowed_by_game = grouped["events"].apply(
            lambda x: x.isin(["home_run"]).sum()
        ).tail(8)
        tb_allowed_by_game = grouped["events"].apply(
            lambda x: (
                (x == "single").sum() * 1
                + (x == "double").sum() * 2
                + (x == "triple").sum() * 3
                + (x == "home_run").sum() * 4
            )
        ).tail(8)

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

        profile = {
            "hits_allowed_avg": round(_weighted_mean(hits_allowed_by_game), 2),
            "tb_allowed_avg": round(_weighted_mean(tb_allowed_by_game), 2),
            "hr_allowed_avg": round(_weighted_mean(hr_allowed_by_game), 3),
            "strikeouts_avg": round(_weighted_mean(strikeouts_by_game), 2),
            "k_std": round(float(strikeouts_by_game.std() or 0.0), 2),
            "xwoba_allowed": round(float(xwoba_allowed.mean()), 3) if xwoba_allowed is not None and len(xwoba_allowed) else None,
            "hard_hit_allowed": round(hard_hit_allowed, 3) if hard_hit_allowed is not None else None,
            "barrel_allowed": round(barrel_allowed, 3) if barrel_allowed is not None else None,
            "hand": _safe_hand(df, "p_throws"),  # L/R
            "games_used": int(len(strikeouts_by_game)),
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