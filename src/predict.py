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


def build_hitter_prop(stat_type, player_name, pitcher_name, line, base_projection, pitcher_hits_allowed, lineup_index, weather):
    lineup_boost = _lineup_boost(stat_type, lineup_index)
    weather_boost, weather_note = _weather_adjustment(stat_type, weather)
    pitcher_adjustment, matchup_note = _pitcher_adjustment(stat_type, pitcher_hits_allowed)

    projection = base_projection + lineup_boost + weather_boost + pitcher_adjustment

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

    probability = edge_to_probability(stat_type, edge)

    return {
        "stat_type": stat_type,
        "player": player_name,
        "pitcher": pitcher_name,
        "line": line,
        "projection": projection,
        "edge": edge,
        "pick": pick,
        "probability": probability,
        "matchup_note": f"{matchup_note} | {weather_note}",
    }


def build_pitcher_k_prop(pitcher_name, line, projection, weather):
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

    adjusted_projection = round(projection + weather_adjustment, 2)
    edge = round(adjusted_projection - line, 2)

    if abs(edge) < 0.25:
        pick = "PASS"
    else:
        pick = "OVER" if edge > 0 else "UNDER"

    probability = edge_to_probability("pitcher_strikeouts", edge)

    return {
        "pitcher": pitcher_name,
        "line": line,
        "projection": adjusted_projection,
        "edge": edge,
        "pick": pick,
        "probability": probability,
        "matchup_note": weather_note,
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