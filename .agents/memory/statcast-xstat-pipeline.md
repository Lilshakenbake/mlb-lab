---
name: Statcast xStat pipeline
description: How per-game expected stats are derived in src/mlb_data.py and the pitch-vs-PA gotcha that silently inflates them
---

The pybaseball statcast DataFrame in `src/mlb_data.py` is **pitch-level** —
one row per pitch. The `events` column is non-null only on the terminal
pitch of each plate appearance; `type` is per-pitch (S/B/X).

**The bug class:** any "per game" rate that does `df.groupby("game_date").size()`
is counting *pitches per game* (~17-19 for a regular), not plate appearances
(~4). That denominator/multiplier error cascades:
- `xtb_avg = xwOBA_mean * pa_per_game * 1.7` ballooned to ~13 TB/game (should ≈ tb_avg ~1.5-2.5).
- `contact_rate = bbe_per_game / pa_per_game` collapsed to ~0.11 (league norm ~0.55-0.62).

**Fix:** count PAs as `df[df["events"].notna()].groupby("game_date").size()`.
Counting hits/HR/TB via `grouped["events"].apply(lambda x: x.isin([...]).sum())`
is already correct because `.isin` treats the NaN non-terminal pitches as False.

**Why it matters:** these xStats feed `_xstat_blend` in `src/predict.py`. A
corrupted `xtb_avg` multiplied straight into the total_bases projection.

**Other gotchas in the same area:**
- `iso_power = (tb_mean - hits_mean) / hits_mean` is **extra bases per hit**
  (~0.4 singles hitter → ~1.5 slugger), NOT traditional ISO (SLG−AVG, ~0.2-0.4).
  Any consumer must center on ~0.55 and use a gentle, capped multiplier — do
  not treat it as trad ISO or it over-boosts TB ~50%.
- xwOBA mean is taken over batted balls only (null on K/BB), so it slightly
  overstates per-PA value. `xtb_avg` is clamped to `<= 1.8 * tb_blend` at source.

**Defense-in-depth:** cached profiles (`data_cache/hitter_profile/*.json`,
12h TTL) keep stale values after a source fix, so `_xstat_blend` also rejects
`xtb_avg > 1.8x` the base projection and applies a final TB sanity band
`[0.65x, 1.6x]` of base. To propagate a source fix immediately, delete the
cached profile JSONs instead of waiting for TTL rollover.
