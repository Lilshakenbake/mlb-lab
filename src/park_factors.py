"""Multiplicative park factors (1.0 = neutral). Sourced from public Statcast
park-factor tables, normalized to 1.00 for the league average ballpark.
Higher = better for that stat. Keep in this module so it's a pure in-memory
lookup (no HTTP, no slowdown)."""

NEUTRAL = {"hits": 1.00, "tb": 1.00, "hr": 1.00, "k": 1.00}

PARK_FACTORS = {
    "Coors Field":              {"hits": 1.10, "tb": 1.20, "hr": 1.28, "k": 0.92},
    "Fenway Park":              {"hits": 1.05, "tb": 1.10, "hr": 0.95, "k": 0.97},
    "Yankee Stadium":           {"hits": 1.00, "tb": 1.05, "hr": 1.20, "k": 0.99},
    "Great American Ball Park": {"hits": 1.02, "tb": 1.08, "hr": 1.20, "k": 0.98},
    "Citizens Bank Park":       {"hits": 1.01, "tb": 1.06, "hr": 1.15, "k": 0.99},
    "Wrigley Field":            {"hits": 1.02, "tb": 1.05, "hr": 1.05, "k": 0.99},
    "Globe Life Field":         {"hits": 1.00, "tb": 1.02, "hr": 1.05, "k": 1.00},
    "Camden Yards":             {"hits": 0.99, "tb": 1.00, "hr": 0.92, "k": 1.00},
    "Chase Field":              {"hits": 1.01, "tb": 1.03, "hr": 1.05, "k": 1.00},
    "Truist Park":              {"hits": 1.00, "tb": 1.01, "hr": 1.02, "k": 1.00},
    "Dodger Stadium":           {"hits": 0.97, "tb": 0.98, "hr": 1.00, "k": 1.02},
    "Petco Park":               {"hits": 0.97, "tb": 0.95, "hr": 0.92, "k": 1.04},
    "Oracle Park":              {"hits": 0.96, "tb": 0.92, "hr": 0.85, "k": 1.04},
    "Tropicana Field":          {"hits": 0.97, "tb": 0.95, "hr": 0.92, "k": 1.03},
    "loanDepot park":           {"hits": 0.97, "tb": 0.95, "hr": 0.93, "k": 1.03},
    "Comerica Park":            {"hits": 0.98, "tb": 0.97, "hr": 0.95, "k": 1.02},
    "T-Mobile Park":            {"hits": 0.97, "tb": 0.95, "hr": 0.93, "k": 1.03},
    "Kauffman Stadium":         {"hits": 1.00, "tb": 1.00, "hr": 0.90, "k": 1.00},
    "Oakland Coliseum":         {"hits": 0.96, "tb": 0.94, "hr": 0.92, "k": 1.03},
    "Athletics Park":           NEUTRAL,
    "Citi Field":               {"hits": 0.98, "tb": 0.97, "hr": 0.95, "k": 1.02},
    "Busch Stadium":            {"hits": 0.99, "tb": 0.98, "hr": 0.95, "k": 1.01},
    "Target Field":             NEUTRAL,
    "Guaranteed Rate Field":    {"hits": 1.01, "tb": 1.04, "hr": 1.10, "k": 0.99},
    "American Family Field":    {"hits": 1.00, "tb": 1.02, "hr": 1.05, "k": 1.00},
    "PNC Park":                 {"hits": 0.99, "tb": 0.98, "hr": 0.95, "k": 1.01},
    "Progressive Field":        {"hits": 0.99, "tb": 0.99, "hr": 0.97, "k": 1.01},
    "Angel Stadium":            NEUTRAL,
    "Nationals Park":           {"hits": 1.01, "tb": 1.02, "hr": 1.05, "k": 1.00},
    "Minute Maid Park":         {"hits": 1.01, "tb": 1.05, "hr": 1.10, "k": 0.99},
    "Rogers Centre":            {"hits": 1.01, "tb": 1.03, "hr": 1.05, "k": 1.00},
}


def get_factor(park_name: str) -> dict:
    if not park_name:
        return dict(NEUTRAL)
    return dict(PARK_FACTORS.get(park_name, NEUTRAL))
