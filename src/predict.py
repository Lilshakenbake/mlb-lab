import math

from src import model as _ml
from src.park_factors import get_factor as _get_park_factor


# Compass bearing (degrees, 0=N) from home plate to dead center field.
# Used to figure out whether tonight's wind is blowing OUT to CF (HR boost)
# or IN from CF (HR damper). Indoor parks are omitted.
PARK_ORIENTATIONS = {
    "Coors Field": 0,
    "Yankee Stadium": 75,
    "Fenway Park": 45,
    "Wrigley Field": 30,
    "Citi Field": 30,
    "Camden Yards": 30,
    "Citizens Bank Park": 15,
    "Petco Park": 0,
    "Oracle Park": 90,
    "Busch Stadium": 60,
    "Truist Park": 60,
    "Great American Ball Park": 120,
    "Comerica Park": 150,
    "Dodger Stadium": 30,
    "Angel Stadium": 60,
    "Kauffman Stadium": 45,
    "Target Field": 90,
    "Nationals Park": 30,
    "PNC Park": 60,
    "Progressive Field": 0,
    "Oakland Coliseum": 60,
    "Athletics Park": 0,
    "Guaranteed Rate Field": 60,
}


def _wind_out_component(park_name, wind_dir_deg, wind_speed_mph):
    """Returns mph component of wind blowing OUT to CF (positive) or IN from
    CF (negative). Wind direction follows meteorological convention: it's the
    direction the wind is coming FROM. Returns None if we can't compute."""
    if not park_name or wind_speed_mph is None or wind_dir_deg is None:
        return None
    bearing = PARK_ORIENTATIONS.get(park_name)
    if bearing is None:
        return None
    # Direction the wind is moving toward = (from + 180) % 360.
    blowing_to = (float(wind_dir_deg) + 180.0) % 360.0
    # Angle between "blowing to" and the CF bearing. 0 = exactly out, 180 = in.
    diff = abs(((blowing_to - bearing) + 540.0) % 360.0 - 180.0)
    return math.cos(math.radians(diff)) * float(wind_speed_mph)


def _xstat_blend(stat_type, base_projection, hitter_profile):
    """Blend Statcast expected stats into the projection. xStats catch lucky
    hitters about to regress and unlucky ones about to break out."""
    if not hitter_profile:
        return base_projection

    if stat_type == "hits":
        x = hitter_profile.get("xhits_avg")
        if x is not None:
            return 0.6 * base_projection + 0.4 * float(x)

    elif stat_type == "total_bases":
        x = hitter_profile.get("xtb_avg")
        if x is not None:
            return 0.6 * base_projection + 0.4 * float(x)

    elif stat_type == "home_runs":
        # Barrel rate is the single best HR predictor — about 25% of barrels
        # actually leave the yard on average.
        bbl = hitter_profile.get("barrel_rate")
        bbe = hitter_profile.get("bbe_per_game")
        if bbl is not None and bbe is not None:
            x_hr = float(bbl) * float(bbe) * 0.25
            return 0.55 * base_projection + 0.45 * x_hr

    elif stat_type == "rbis":
        x = hitter_profile.get("xtb_avg")
        if x is not None:
            # Half of expected TB roughly maps to RBI rate at lineup-spot baseline.
            return 0.7 * base_projection + 0.3 * (float(x) * 0.5)

    return base_projection


def _platoon_factor(stat_type, batter_hand, pitcher_hand):
    """Multiplicative platoon adjustment. Same-hand matchup hurts hitter,
    opposite-hand helps. Switch hitters always face opposite hand → small boost."""
    if not batter_hand or not pitcher_hand:
        return 1.0
    bh = batter_hand.upper()
    ph = pitcher_hand.upper()
    if bh == "S":
        same = False  # switch hitter takes the favorable side
    else:
        same = (bh == ph)

    if stat_type == "hits":
        return 0.93 if same else 1.05
    if stat_type == "total_bases":
        return 0.91 if same else 1.06
    if stat_type == "home_runs":
        return 0.85 if same else 1.10
    if stat_type == "rbis":
        return 0.93 if same else 1.05
    return 1.0


def _pitcher_k_damper(stat_type, opp_pitcher_k_avg):
    """High-K starter suppresses contact-based stats; soft-tossing starter inflates them."""
    if opp_pitcher_k_avg is None:
        return 1.0
    k = float(opp_pitcher_k_avg)
    if stat_type == "hits":
        if k >= 8.0: return 0.88
        if k >= 6.5: return 0.95
        if k <= 4.0: return 1.08
        return 1.0
    if stat_type == "total_bases":
        if k >= 8.0: return 0.85
        if k >= 6.5: return 0.94
        if k <= 4.0: return 1.10
        return 1.0
    if stat_type == "home_runs":
        if k >= 8.0: return 0.93
        if k <= 4.0: return 1.05
        return 1.0
    if stat_type == "rbis":
        if k >= 8.0: return 0.88
        if k <= 4.0: return 1.08
        return 1.0
    return 1.0


def _hot_factor(hot_ratio):
    """Tiny nudge for players riding a hot/cold streak (last 3 vs window)."""
    if hot_ratio is None:
        return 1.0
    r = float(hot_ratio)
    if r >= 1.40: return 1.05
    if r >= 1.20: return 1.03
    if r <= 0.60: return 0.95
    if r <= 0.80: return 0.97
    return 1.0


def _clamp_factor(f, lo=0.55, hi=1.55):
    return max(lo, min(hi, f))


def edge_to_probability(stat_type, edge):
    abs_edge = abs(edge)

    if stat_type == "hits":
        if abs_edge >= 0.55:
            return 74
        elif abs_edge >= 0.40:
            return 68
        elif abs_edge >= 0.25:
            return 61
        elif abs_edge >= 0.15:
            return 56
        return 51

    if stat_type == "home_runs":
        if abs_edge >= 0.22:
            return 68
        elif abs_edge >= 0.15:
            return 62
        elif abs_edge >= 0.08:
            return 56
        return 51

    if stat_type == "total_bases":
        if abs_edge >= 0.90:
            return 73
        elif abs_edge >= 0.65:
            return 66
        elif abs_edge >= 0.40:
            return 60
        elif abs_edge >= 0.20:
            return 55
        return 51

    if stat_type == "rbis":
        if abs_edge >= 0.35:
            return 69
        elif abs_edge >= 0.22:
            return 62
        elif abs_edge >= 0.12:
            return 56
        return 51

    if stat_type == "pitcher_strikeouts":
        if abs_edge >= 1.75:
            return 74
        elif abs_edge >= 1.20:
            return 67
        elif abs_edge >= 0.70:
            return 60
        elif abs_edge >= 0.35:
            return 55
        return 51

    return 51


def _lineup_boost(stat_type, lineup_index):
    spot = lineup_index + 1

    if stat_type == "hits":
        if spot == 1:
            return 0.22
        elif spot == 2:
            return 0.20
        elif spot == 3:
            return 0.18
        elif spot == 4:
            return 0.14
        elif spot == 5:
            return 0.10
        elif spot == 6:
            return 0.04
        return -0.05

    if stat_type == "total_bases":
        if spot == 1:
            return 0.24
        elif spot == 2:
            return 0.22
        elif spot == 3:
            return 0.25
        elif spot == 4:
            return 0.24
        elif spot == 5:
            return 0.16
        elif spot == 6:
            return 0.06
        return -0.08

    if stat_type == "home_runs":
        if spot in [3, 4]:
            return 0.08
        elif spot in [2, 5]:
            return 0.05
        elif spot == 1:
            return 0.02
        return -0.03

    if stat_type == "rbis":
        if spot == 3:
            return 0.22
        elif spot == 4:
            return 0.28
        elif spot == 5:
            return 0.18
        elif spot == 2:
            return 0.08
        elif spot == 1:
            return 0.03
        return -0.04

    return 0.0


def _weather_adjustment(stat_type, weather):
    adj = 0.0
    note = "Weather neutral"

    if not weather:
        return adj, note

    if weather.get("is_indoor"):
        return 0.0, "Indoor / roof closed"

    temp = float(weather.get("temperature", 70) or 70)
    wind = float(weather.get("wind_speed", 0) or 0)

    if stat_type in ["home_runs", "total_bases", "rbis"]:
        if temp >= 82 and wind >= 10:
            adj += 0.08
            note = "Weather favors offense"
        elif temp <= 52:
            adj -= 0.05
            note = "Cool weather penalty"

    elif stat_type == "hits":
        if temp >= 82 and wind >= 10:
            adj += 0.03
            note = "Small weather boost"
        elif temp <= 52:
            adj -= 0.02
            note = "Small weather penalty"

    # Wind direction relative to park orientation. Strongest signal for HR.
    wind_dir = weather.get("wind_direction")
    park = weather.get("park")
    out_comp = _wind_out_component(park, wind_dir, wind)
    if out_comp is not None and abs(out_comp) >= 4:
        if stat_type == "home_runs":
            adj += out_comp * 0.012  # +0.18 per game with 15mph straight out
        elif stat_type in ("total_bases", "rbis"):
            adj += out_comp * 0.008
        elif stat_type == "hits":
            adj += out_comp * 0.003
        if abs(out_comp) >= 8:
            direction = "out to CF" if out_comp > 0 else "in from CF"
            note = f"Wind {abs(out_comp):.0f}mph {direction}"

    return adj, note


def _pitcher_adjustment(stat_type, pitcher_hits_allowed):
    if pitcher_hits_allowed is None:
        return 0.0, "No pitcher adjustment"

    if stat_type == "hits":
        if pitcher_hits_allowed >= 7.0:
            return 0.15, "Weak opposing pitcher"
        elif pitcher_hits_allowed <= 4.0:
            return -0.12, "Strong opposing pitcher"
        return 0.0, "Neutral matchup"

    if stat_type == "total_bases":
        if pitcher_hits_allowed >= 7.0:
            return 0.22, "Weak opposing pitcher"
        elif pitcher_hits_allowed <= 4.0:
            return -0.18, "Strong opposing pitcher"
        return 0.0, "Neutral matchup"

    if stat_type == "home_runs":
        if pitcher_hits_allowed >= 7.0:
            return 0.06, "Weak opposing pitcher"
        elif pitcher_hits_allowed <= 4.0:
            return -0.05, "Strong opposing pitcher"
        return 0.0, "Neutral matchup"

    if stat_type == "rbis":
        if pitcher_hits_allowed >= 7.0:
            return 0.10, "Weak opposing pitcher"
        elif pitcher_hits_allowed <= 4.0:
            return -0.08, "Strong opposing pitcher"
        return 0.0, "Neutral matchup"

    return 0.0, "Neutral matchup"


def build_hitter_prop(stat_type, player_name, pitcher_name, line, base_projection,
                      pitcher_hits_allowed, lineup_index, weather, hitter_profile=None,
                      opp_pitcher_profile=None, park_name=None):
    lineup_boost = _lineup_boost(stat_type, lineup_index)
    weather_boost, weather_note = _weather_adjustment(stat_type, weather)
    pitcher_adjustment, matchup_note = _pitcher_adjustment(stat_type, pitcher_hits_allowed)

    model_used = False
    model_std = None
    indoor = 1 if (weather and weather.get("is_indoor")) else 0
    if hitter_profile and stat_type in ("hits", "total_bases", "home_runs", "rbis"):
        fn_map = {
            "hits": _ml.hitter_hits,
            "total_bases": _ml.hitter_total_bases,
            "home_runs": _ml.hitter_home_runs,
            "rbis": _ml.hitter_rbis,
        }
        fn = fn_map[stat_type]
        ml = fn(
            hits_avg_5=hitter_profile.get("hits_avg", base_projection),
            tb_avg_5=hitter_profile.get("tb_avg", base_projection),
            hr_avg_5=hitter_profile.get("hr_avg", 0.0),
            rbi_avg_5=hitter_profile.get("rbi_avg", 0.0),
            indoor=indoor,
        )
        if ml is not None:
            ml_proj, model_std = ml
            base_projection = 0.5 * base_projection + 0.5 * ml_proj
            model_used = True

    # Statcast xStat blend — applies whether or not a trained model exists.
    # Catches lucky/unlucky hitters before raw counting stats catch up.
    pre_x = base_projection
    base_projection = _xstat_blend(stat_type, base_projection, hitter_profile)
    xstat_used = abs(base_projection - pre_x) >= 0.01

    # Multiplicative context factors (zero new HTTP — derived from data we already pull).
    park_key = {"hits": "hits", "total_bases": "tb", "home_runs": "hr", "rbis": "tb"}.get(stat_type, "hits")
    park_f = _get_park_factor(park_name).get(park_key, 1.0)
    k_avg = (opp_pitcher_profile or {}).get("strikeouts_avg")
    pitcher_k_f = _pitcher_k_damper(stat_type, k_avg)
    platoon_f = _platoon_factor(
        stat_type,
        (hitter_profile or {}).get("hand"),
        (opp_pitcher_profile or {}).get("hand"),
    )
    hot_f = _hot_factor((hitter_profile or {}).get("hot_ratio"))
    context_factor = _clamp_factor(park_f * pitcher_k_f * platoon_f * hot_f)

    # Anti double-count: the K damper and the additive pitcher_adjustment both
    # encode pitcher quality. When the K damper has already pulled the projection
    # the same direction as the hits-allowed nudge, shrink the additive piece.
    if pitcher_adjustment != 0 and pitcher_k_f != 1.0:
        same_dir = (pitcher_k_f < 1.0 and pitcher_adjustment < 0) or (pitcher_k_f > 1.0 and pitcher_adjustment > 0)
        if same_dir:
            pitcher_adjustment *= 0.5

    projection = (base_projection * context_factor) + lineup_boost + weather_boost + pitcher_adjustment

    # Build a short why-string so the user can see what moved this pick.
    factor_bits = []
    if abs(park_f - 1.0) >= 0.04:
        factor_bits.append(f"park {park_f - 1:+.0%}")
    if abs(pitcher_k_f - 1.0) >= 0.04:
        factor_bits.append(f"opp K {pitcher_k_f - 1:+.0%}")
    if abs(platoon_f - 1.0) >= 0.04:
        factor_bits.append(f"platoon {platoon_f - 1:+.0%}")
    if abs(hot_f - 1.0) >= 0.04:
        factor_bits.append(f"trend {hot_f - 1:+.0%}")
    if xstat_used:
        factor_bits.append("xStats")

    # stricter penalties for weak starter profiles
    if stat_type == "hits" and base_projection < 0.70:
        projection -= 0.16
    elif stat_type == "total_bases" and base_projection < 1.05:
        projection -= 0.22
    elif stat_type == "home_runs" and base_projection < 0.12:
        projection -= 0.05
    elif stat_type == "rbis" and base_projection < 0.22:
        projection -= 0.08

    projection = round(projection, 2)
    edge = round(projection - line, 2)

    # stronger PASS rules so weak starters stop surfacing
    if stat_type == "hits" and abs(edge) < 0.12:
        pick = "PASS"
    elif stat_type == "total_bases" and abs(edge) < 0.18:
        pick = "PASS"
    elif stat_type == "home_runs" and abs(edge) < 0.06:
        pick = "PASS"
    elif stat_type == "rbis" and abs(edge) < 0.08:
        pick = "PASS"
    else:
        pick = "OVER" if edge > 0 else "UNDER"

    if model_used and model_std:
        probability = _ml.over_probability(projection, line, model_std)
    else:
        probability = edge_to_probability(stat_type, edge)

    note_parts = [matchup_note, weather_note]
    if factor_bits:
        note_parts.append(", ".join(factor_bits))

    return {
        "stat_type": stat_type,
        "player": player_name,
        "pitcher": pitcher_name,
        "line": line,
        "projection": projection,
        "edge": edge,
        "pick": pick,
        "probability": probability,
        "matchup_note": " | ".join(p for p in note_parts if p),
        "model_used": model_used,
    }


def build_pitcher_k_prop(pitcher_name, line, projection, weather, pitcher_profile=None):
    weather_note = "Weather neutral"
    weather_adjustment = 0.0

    if weather:
        if weather.get("is_indoor"):
            weather_note = "Indoor / roof closed"
        else:
            temp = float(weather.get("temperature", 70) or 70)
            if temp <= 52:
                weather_adjustment += 0.15
                weather_note = "Cool weather slight K boost"

    model_used = False
    model_std = None
    indoor = 1 if (weather and weather.get("is_indoor")) else 0
    if pitcher_profile:
        ml = _ml.pitcher_strikeouts(
            k_avg_5=pitcher_profile.get("strikeouts_avg", projection),
            hits_allowed_avg_5=pitcher_profile.get("hits_allowed_avg", 6.0),
            indoor=indoor,
        )
        if ml is not None:
            ml_proj, model_std = ml
            projection = 0.5 * projection + 0.5 * ml_proj
            model_used = True

    adjusted_projection = round(projection + weather_adjustment, 2)
    edge = round(adjusted_projection - line, 2)

    if abs(edge) < 0.25:
        pick = "PASS"
    else:
        pick = "OVER" if edge > 0 else "UNDER"

    if model_used and model_std:
        probability = _ml.over_probability(adjusted_projection, line, model_std)
    else:
        probability = edge_to_probability("pitcher_strikeouts", edge)

    return {
        "pitcher": pitcher_name,
        "line": line,
        "projection": adjusted_projection,
        "edge": edge,
        "pick": pick,
        "probability": probability,
        "matchup_note": weather_note,
        "model_used": model_used,
    }


def build_spread_lean(game, home_team_score, away_team_score, home_pitcher_profile, away_pitcher_profile, weather):
    home_pitching_score = 0.0
    away_pitching_score = 0.0

    if home_pitcher_profile:
        home_pitching_score += home_pitcher_profile.get("strikeouts_avg", 0) * 0.22
        home_pitching_score -= home_pitcher_profile.get("hits_allowed_avg", 0) * 0.18

    if away_pitcher_profile:
        away_pitching_score += away_pitcher_profile.get("strikeouts_avg", 0) * 0.22
        away_pitching_score -= away_pitcher_profile.get("hits_allowed_avg", 0) * 0.18

    home_field_boost = 0.65

    weather_adjustment = 0.0
    weather_note = "Weather neutral"

    if weather:
        if weather.get("is_indoor"):
            weather_note = "Indoor / roof closed"
        else:
            temp = float(weather.get("temperature", 70) or 70)
            wind = float(weather.get("wind_speed", 0) or 0)

            if temp >= 82 and wind >= 10:
                weather_adjustment += 0.20
                weather_note = "Weather favors offense"
            elif temp <= 52:
                weather_adjustment -= 0.15
                weather_note = "Cool weather may suppress scoring"

    home_total = home_team_score + home_pitching_score + home_field_boost + weather_adjustment
    away_total = away_team_score + away_pitching_score

    margin = round(home_total - away_total, 2)
    abs_margin = abs(margin)

    if abs_margin >= 4.0:
        ml_probability = 72
    elif abs_margin >= 3.0:
        ml_probability = 68
    elif abs_margin >= 2.0:
        ml_probability = 63
    elif abs_margin >= 1.25:
        ml_probability = 58
    elif abs_margin >= 0.6:
        ml_probability = 54
    else:
        ml_probability = 51

    if abs_margin >= 4.0:
        run_line_probability = 64
    elif abs_margin >= 3.0:
        run_line_probability = 60
    elif abs_margin >= 2.0:
        run_line_probability = 56
    elif abs_margin >= 1.25:
        run_line_probability = 53
    else:
        run_line_probability = 50

    if margin > 0:
        ml_pick = f"{game['home_team']} ML"
        run_line_pick = f"{game['home_team']} -1.5"
    else:
        ml_pick = f"{game['away_team']} ML"
        run_line_pick = f"{game['away_team']} -1.5"

    confidence = "LOW"
    if ml_probability >= 66:
        confidence = "HIGH"
    elif ml_probability >= 57:
        confidence = "MED"

    return {
        "ml_pick": ml_pick,
        "ml_probability": ml_probability,
        "run_line_pick": run_line_pick,
        "run_line_probability": run_line_probability,
        "margin": margin,
        "confidence": confidence,
        "note": weather_note,
    }