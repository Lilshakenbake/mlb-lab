"""Team defense factors — multiplier applied to hitter projections based on
the OPPONENT'S defensive quality (OAA / Defensive Runs Saved).

Defense suppresses HITS first (the same liner is a single off Albies, a hit
off Schwarber). It also nudges TB (extra-base hits get cut down by good
outfielders). HR is unaffected (over the wall doesn't care about D).

Multipliers are mild (~±8%) — even an elite defense only saves ~30-40 hits
per season vs a brutal one, spread across ~1,400 balls in play.

Built from 2026 season-to-date team OAA + DRS rankings.
Refresh quarterly to stay current with mid-season trades + alignment shifts.
"""

NEUTRAL = {"hits": 1.00, "tb": 1.00}

# Format: team name → multipliers for hitter projections vs this defense.
# (rank 1 = best D = suppresses opponent hits; rank 30 = worst = inflates)
DEFENSE_FACTORS = {
    # ── Elite defense (top 5) ────────────────────────────────────────────
    "Brewers":            {"hits": 0.93, "tb": 0.94},
    "Diamondbacks":       {"hits": 0.94, "tb": 0.95},
    "Padres":             {"hits": 0.94, "tb": 0.95},
    "Astros":             {"hits": 0.95, "tb": 0.96},
    "Mariners":           {"hits": 0.95, "tb": 0.96},
    # ── Above avg (6-12) ─────────────────────────────────────────────────
    "Guardians":          {"hits": 0.96, "tb": 0.97},
    "Dodgers":            {"hits": 0.97, "tb": 0.97},
    "Rays":               {"hits": 0.97, "tb": 0.97},
    "Tigers":             {"hits": 0.97, "tb": 0.98},
    "Yankees":            {"hits": 0.98, "tb": 0.98},
    "Phillies":           {"hits": 0.98, "tb": 0.98},
    "Giants":             {"hits": 0.98, "tb": 0.99},
    # ── Average (13-18) ──────────────────────────────────────────────────
    "Cubs":               {"hits": 0.99, "tb": 0.99},
    "Mets":               {"hits": 0.99, "tb": 1.00},
    "Braves":             {"hits": 1.00, "tb": 1.00},
    "Red Sox":            {"hits": 1.00, "tb": 1.00},
    "Twins":              {"hits": 1.00, "tb": 1.01},
    "Orioles":            {"hits": 1.01, "tb": 1.01},
    # ── Below avg (19-24) ────────────────────────────────────────────────
    "Cardinals":          {"hits": 1.02, "tb": 1.02},
    "Reds":               {"hits": 1.02, "tb": 1.02},
    "Pirates":            {"hits": 1.03, "tb": 1.03},
    "Blue Jays":          {"hits": 1.03, "tb": 1.03},
    "Royals":             {"hits": 1.04, "tb": 1.03},
    "Rangers":            {"hits": 1.04, "tb": 1.04},
    # ── Bad defense (25-30) — INFLATE opp hits ───────────────────────────
    "Marlins":            {"hits": 1.05, "tb": 1.04},
    "Athletics":          {"hits": 1.05, "tb": 1.05},
    "Angels":             {"hits": 1.06, "tb": 1.05},
    "Nationals":          {"hits": 1.06, "tb": 1.06},
    "White Sox":          {"hits": 1.07, "tb": 1.06},
    "Rockies":            {"hits": 1.08, "tb": 1.07},
}


def get_defense_factor(team_name):
    """Return defense multipliers dict for the given team, or NEUTRAL if unknown."""
    if not team_name:
        return NEUTRAL
    name_lower = team_name.lower()
    for key, factors in DEFENSE_FACTORS.items():
        if key.lower() in name_lower:
            return factors
    return NEUTRAL
