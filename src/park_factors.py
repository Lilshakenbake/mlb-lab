"""Multiplicative park factors (1.0 = neutral). Sourced from public Statcast
park-factor tables, normalized to 1.00 for the league average ballpark.
Higher = better for that stat. Keep in this module so it's a pure in-memory
lookup (no HTTP, no slowdown).

HR factors come in three flavors:
  - "hr"      : overall (used when batter handedness unknown)
  - "hr_lhb"  : HR factor for left-handed batters
  - "hr_rhb"  : HR factor for right-handed batters
Switch hitters use the favorable side at game time (handled by callers).
"""

NEUTRAL = {"hits": 1.00, "tb": 1.00, "hr": 1.00, "k": 1.00}

PARK_FACTORS = {
    "Coors Field":              {"hits": 1.10, "tb": 1.20, "hr": 1.28, "hr_lhb": 1.28, "hr_rhb": 1.28, "k": 0.92},
    "Fenway Park":              {"hits": 1.05, "tb": 1.10, "hr": 0.95, "hr_lhb": 0.85, "hr_rhb": 1.05, "k": 0.97},
    "Yankee Stadium":           {"hits": 1.00, "tb": 1.05, "hr": 1.20, "hr_lhb": 1.40, "hr_rhb": 1.05, "k": 0.99},
    "Great American Ball Park": {"hits": 1.02, "tb": 1.08, "hr": 1.20, "hr_lhb": 1.22, "hr_rhb": 1.18, "k": 0.98},
    "Citizens Bank Park":       {"hits": 1.01, "tb": 1.06, "hr": 1.15, "hr_lhb": 1.18, "hr_rhb": 1.12, "k": 0.99},
    "Wrigley Field":            {"hits": 1.02, "tb": 1.05, "hr": 1.05, "hr_lhb": 1.08, "hr_rhb": 1.02, "k": 0.99},
    "Globe Life Field":         {"hits": 1.00, "tb": 1.02, "hr": 1.05, "hr_lhb": 1.08, "hr_rhb": 1.02, "k": 1.00},
    "Camden Yards":             {"hits": 0.99, "tb": 1.00, "hr": 0.85, "hr_lhb": 0.95, "hr_rhb": 0.78, "k": 1.00},
    "Chase Field":              {"hits": 1.01, "tb": 1.03, "hr": 1.05, "hr_lhb": 1.06, "hr_rhb": 1.04, "k": 1.00},
    "Truist Park":              {"hits": 1.00, "tb": 1.01, "hr": 1.02, "hr_lhb": 1.04, "hr_rhb": 1.00, "k": 1.00},
    "Dodger Stadium":           {"hits": 0.97, "tb": 0.98, "hr": 1.00, "hr_lhb": 1.05, "hr_rhb": 0.95, "k": 1.02},
    "Petco Park":               {"hits": 0.97, "tb": 0.95, "hr": 0.92, "hr_lhb": 0.90, "hr_rhb": 0.94, "k": 1.04},
    "Oracle Park":              {"hits": 0.96, "tb": 0.92, "hr": 0.85, "hr_lhb": 0.78, "hr_rhb": 0.92, "k": 1.04},
    "Tropicana Field":          {"hits": 0.97, "tb": 0.95, "hr": 0.92, "hr_lhb": 0.92, "hr_rhb": 0.92, "k": 1.03},
    "loanDepot park":           {"hits": 0.97, "tb": 0.95, "hr": 0.93, "hr_lhb": 0.93, "hr_rhb": 0.93, "k": 1.03},
    "Comerica Park":            {"hits": 0.98, "tb": 0.97, "hr": 0.95, "hr_lhb": 0.93, "hr_rhb": 0.97, "k": 1.02},
    "T-Mobile Park":            {"hits": 0.97, "tb": 0.95, "hr": 0.93, "hr_lhb": 0.88, "hr_rhb": 0.96, "k": 1.03},
    "Kauffman Stadium":         {"hits": 1.00, "tb": 1.00, "hr": 0.90, "hr_lhb": 0.88, "hr_rhb": 0.92, "k": 1.00},
    "Oakland Coliseum":         {"hits": 0.96, "tb": 0.94, "hr": 0.92, "hr_lhb": 0.92, "hr_rhb": 0.92, "k": 1.03},
    "Athletics Park":           NEUTRAL,
    "Citi Field":               {"hits": 0.98, "tb": 0.97, "hr": 0.95, "hr_lhb": 0.88, "hr_rhb": 1.00, "k": 1.02},
    "Busch Stadium":            {"hits": 0.99, "tb": 0.98, "hr": 0.95, "hr_lhb": 0.93, "hr_rhb": 0.97, "k": 1.01},
    "Target Field":             NEUTRAL,
    "Guaranteed Rate Field":    {"hits": 1.01, "tb": 1.04, "hr": 1.10, "hr_lhb": 1.12, "hr_rhb": 1.08, "k": 0.99},
    "American Family Field":    {"hits": 1.00, "tb": 1.02, "hr": 1.05, "hr_lhb": 1.06, "hr_rhb": 1.04, "k": 1.00},
    "PNC Park":                 {"hits": 0.99, "tb": 0.98, "hr": 0.95, "hr_lhb": 0.88, "hr_rhb": 0.98, "k": 1.01},
    "Progressive Field":        {"hits": 0.99, "tb": 0.99, "hr": 0.97, "hr_lhb": 0.97, "hr_rhb": 0.97, "k": 1.01},
    "Angel Stadium":            NEUTRAL,
    "Nationals Park":           {"hits": 1.01, "tb": 1.02, "hr": 1.05, "hr_lhb": 1.06, "hr_rhb": 1.04, "k": 1.00},
    "Minute Maid Park":         {"hits": 1.01, "tb": 1.05, "hr": 1.10, "hr_lhb": 1.05, "hr_rhb": 1.18, "k": 0.99},
    "Rogers Centre":            {"hits": 1.01, "tb": 1.03, "hr": 1.05, "hr_lhb": 1.06, "hr_rhb": 1.04, "k": 1.00},
}


def get_factor(park_name: str) -> dict:
    if not park_name:
        return dict(NEUTRAL)
    return dict(PARK_FACTORS.get(park_name, NEUTRAL))


def get_hr_factor(park_name: str, batter_hand, pitcher_hand=None) -> float:
    """HR park factor for a specific batter handedness. Falls back to the
    overall HR factor when the park has no split or the hand is unknown.

    Switch hitters bat opposite the pitcher's hand (vs RHP they bat L, vs
    LHP they bat R) — so resolve via `pitcher_hand` when provided. If the
    pitcher hand is unknown, fall back to the overall park HR factor (no
    asymmetric bias toward whichever side happens to be better).
    """
    pf = PARK_FACTORS.get(park_name, NEUTRAL) if park_name else NEUTRAL
    overall = float(pf.get("hr", 1.0))
    if not batter_hand:
        return overall
    h = str(batter_hand).upper()
    lhb = float(pf.get("hr_lhb", overall))
    rhb = float(pf.get("hr_rhb", overall))
    if h == "L":
        return lhb
    if h == "R":
        return rhb
    if h == "S":
        if pitcher_hand:
            ph = str(pitcher_hand).upper()
            if ph == "R":
                return lhb  # bats L vs RHP
            if ph == "L":
                return rhb  # bats R vs LHP
        return overall
    return overall
