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

# Moneyline factoring (build_spread_lean + attach_game_edges)

The model margin is a heuristic (lineup offense index + starter K/hits score +
flat home-field boost + weather). margin = home_total − away_total.

Two-stage probability:
1. **Model win%** (`build_spread_lean`, src/predict.py): `_winprob_from_margin`
   is a logistic curve (k=0.27) on the margin, clamped to [0.20, 0.80]. Returns
   `model_home_win_prob` (0..1, home frame) + `model_ml_probability`.
   Replaced the old hardcoded bucket table.
2. **Market blend** (`attach_game_edges`, src/live_odds.py): when odds exist,
   `market_home_winprob` de-vigs each book's two-way h2h and averages →
   consensus P(home). Blend = `ML_MARKET_WEIGHT`(0.65)·market + 0.35·model.
   The **blend picks the side** (can flip a thin model lean) and sets
   `ml_pick`/`ml_probability`/`confidence`, plus `ml_blended`/`ml_market_prob`/
   `ml_model_prob`. Then edge%/EV%/Kelly price the chosen side vs best book.

**Why market-weighted:** the de-vigged market line is the single best ML
predictor; the model only nudges it and finds disagreement edges.

**Cross-market guardrail:** the run line is still model-margin-based. If the
blend flips ML opposite the model's `-1.5` favorite (strong model/market
disagreement → model usually wrong), the RL is neutralized
(`run_line_probability=0`, `run_line_suppressed=True`) so the board isn't
contradictory.

**Fallback:** no odds (free API exhausted) → no blend, model-only ML stands.
