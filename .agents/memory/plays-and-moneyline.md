---
name: Plays of the Day filter & moneyline factoring
description: Product decisions for what surfaces in Plays of the Day, and what actually drives the moneyline pick/probability
---

# Plays of the Day — OVER-only for hitters, pitcher unders kept

`_build_plays_for_game` (app.py) feeds both the displayed Plays of the Day
(PLAYS_CACHE) and the parlay solver pool (RAW_PLAYS_CACHE). The hitter
count-stat loop (hits/total_bases/home_runs/rbis/steals) keeps **OVER only**
(`pick != "OVER"` skipped).

**Why:** Plays of the Day is a "hunt production" board — an UNDER on a hitter's
hits/bases isn't its philosophy, and the user saw a stray UNDER hits/bases pick
there and wanted them gone.

**Deliberately kept:** pitcher strikeout UNDERs (fading a weak-K starter) and
ML/RL/Totals leans. The user explicitly said books push a lot of hits/bases
unders but pitcher unders are fine — do NOT strip pitcher unders.

**How to apply:** filter only in `_build_plays_for_game`. Do NOT strip UNDER
from the `top_*` buckets globally — those also feed per-game board pages.
bases-board and hits-combos already filter OVER-only separately.

# Moneyline factoring (build_spread_lean in src/predict.py)

The ML pick + win% is a **heuristic**, not a real win-prob model:
- team offensive index (lineup projected hits×0.8 + TB×1.0)
- starter score (strikeouts_avg×0.22 − hits_allowed_avg×0.18)
- flat home-field boost +0.65
- weather (hot+windy +0.20, cold ≤52° −0.15)
- margin = home_total − away_total; pick = higher side
- win% = a **hardcoded bucket table** off |margin| (51% pickem → 72% at 4+)

The live betting market does NOT pick the side. `live_odds.attach_game_edges`
only prices the chosen pick: pulls best book ML, de-vigs, computes
edge%/EV%/Kelly vs the heuristic probability.

**Weakness (if asked to improve ROI):** margin mixes incommensurate units, and
win% is bucketed not distributional; de-vigged market prob is usually the single
best ML predictor and isn't blended in.
