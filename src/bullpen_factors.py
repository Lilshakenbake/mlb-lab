"""Team bullpen quality factors — multiplier applied to hitter projections
when modeling late-inning ABs against the opposing pen (innings 6-9).

Based on 2026 season-to-date bullpen ERA / FIP rankings. Lower ERA = tougher
pen = SUPPRESSES hitter projections; higher ERA = inflates them.

Multipliers are mild (~±8%) because the starter still throws ~60% of game IP.
The factor is BLENDED with starter quality rather than replacing it.

Refresh quarterly to stay current with bullpen turnover."""

NEUTRAL = {"hits": 1.00, "tb": 1.00, "hr": 1.00, "k": 1.00}

# Format: team name → multipliers for late-inning hitter ABs
# (rank 1 = best pen, rank 30 = worst)
BULLPEN_FACTORS = {
    # ── Elite pens (top 5) — suppress hitter projections ─────────────────
    "Padres":             {"hits": 0.93, "tb": 0.92, "hr": 0.88, "k": 1.10},
    "Yankees":            {"hits": 0.94, "tb": 0.93, "hr": 0.90, "k": 1.08},
    "Astros":             {"hits": 0.94, "tb": 0.93, "hr": 0.91, "k": 1.07},
    "Phillies":           {"hits": 0.95, "tb": 0.94, "hr": 0.92, "k": 1.06},
    "Brewers":            {"hits": 0.95, "tb": 0.94, "hr": 0.92, "k": 1.06},
    # ── Above avg (6-12) ─────────────────────────────────────────────────
    "Guardians":          {"hits": 0.96, "tb": 0.96, "hr": 0.94, "k": 1.05},
    "Dodgers":            {"hits": 0.96, "tb": 0.96, "hr": 0.95, "k": 1.04},
    "Mariners":           {"hits": 0.97, "tb": 0.96, "hr": 0.95, "k": 1.04},
    "Red Sox":            {"hits": 0.97, "tb": 0.97, "hr": 0.96, "k": 1.03},
    "Tigers":             {"hits": 0.97, "tb": 0.97, "hr": 0.96, "k": 1.03},
    "Mets":               {"hits": 0.98, "tb": 0.98, "hr": 0.97, "k": 1.02},
    "Cubs":               {"hits": 0.98, "tb": 0.98, "hr": 0.97, "k": 1.02},
    # ── Average (13-18) ──────────────────────────────────────────────────
    "Rays":               {"hits": 0.99, "tb": 0.99, "hr": 0.98, "k": 1.01},
    "Giants":             {"hits": 0.99, "tb": 0.99, "hr": 0.99, "k": 1.01},
    "Braves":             {"hits": 1.00, "tb": 1.00, "hr": 1.00, "k": 1.00},
    "Orioles":            {"hits": 1.00, "tb": 1.00, "hr": 1.00, "k": 1.00},
    "Twins":              {"hits": 1.00, "tb": 1.00, "hr": 1.01, "k": 0.99},
    "Reds":               {"hits": 1.01, "tb": 1.01, "hr": 1.01, "k": 0.99},
    # ── Below avg (19-24) ────────────────────────────────────────────────
    "Diamondbacks":       {"hits": 1.02, "tb": 1.02, "hr": 1.02, "k": 0.98},
    "Royals":             {"hits": 1.02, "tb": 1.02, "hr": 1.03, "k": 0.98},
    "Blue Jays":          {"hits": 1.03, "tb": 1.03, "hr": 1.04, "k": 0.97},
    "Cardinals":          {"hits": 1.03, "tb": 1.04, "hr": 1.04, "k": 0.97},
    "Rangers":            {"hits": 1.04, "tb": 1.04, "hr": 1.05, "k": 0.96},
    "Pirates":            {"hits": 1.04, "tb": 1.05, "hr": 1.06, "k": 0.95},
    # ── Bad pens (25-30) — INFLATE hitter projections ────────────────────
    "Athletics":          {"hits": 1.05, "tb": 1.06, "hr": 1.08, "k": 0.94},
    "Angels":             {"hits": 1.05, "tb": 1.06, "hr": 1.08, "k": 0.93},
    "Nationals":          {"hits": 1.06, "tb": 1.07, "hr": 1.09, "k": 0.93},
    "White Sox":          {"hits": 1.06, "tb": 1.07, "hr": 1.10, "k": 0.92},
    "Marlins":            {"hits": 1.07, "tb": 1.08, "hr": 1.10, "k": 0.92},
    "Rockies":            {"hits": 1.08, "tb": 1.10, "hr": 1.12, "k": 0.90},
}


def get_bullpen_factor(team_name):
    """Return bullpen multipliers dict for the given team, or NEUTRAL if unknown."""
    if not team_name:
        return NEUTRAL
    # Match on substring so "Los Angeles Dodgers" / "LAD" / "Dodgers" all hit.
    name_lower = team_name.lower()
    for key, factors in BULLPEN_FACTORS.items():
        if key.lower() in name_lower:
            return factors
    return NEUTRAL
