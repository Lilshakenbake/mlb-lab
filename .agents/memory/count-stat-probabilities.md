---
name: Count-stat prop probabilities
description: How hitter count-stat props convert a projection to P(over) and why the model choice matters
---

Hitter count props (hits 0.5, total_bases 1.5, home_runs 0.5, rbis 0.5) are
discrete `P(count >= k)` events, not continuous. `src/model.py` exposes
`count_prob_over(projection, line, stat_type)`:
- **Poisson tail** for hits / home_runs / rbis.
- **Over-dispersed negative-binomial tail** for total_bases (a HR adds 4
  bases at once → fatter tail than Poisson; dispersion ~0.6).

Wired in `build_hitter_prop` (`src/predict.py`): `p_over = count_prob_over(...)`,
and UNDER side is `1 - p_over`.

**Why this matters / durable lesson:** the old path used a normal-CDF
(`over_probability`) or an edge-bucket table that treated low integer counts
as continuous and ran ~7pp overconfident. Those **also capped** the output
(e.g. TB capped ~73%), which silently *masked* grossly inflated projections.
When you replace the probability conversion with a faithful discrete tail, it
will surface any latent projection bug as an absurdly high probability — so
after changing the conversion, sanity-check the projections feeding it, not
just the probabilities.

`app.py _calibrated_probability` applies a small residual haircut (×0.97 on
hitter props in the 65-80 band, ×0.95 on HR/HRR). It is a safety margin, not
the primary calibration — the tails do the real work.
