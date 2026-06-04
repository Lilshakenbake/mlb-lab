---
name: F5 calibration
description: How build_f5_lean is calibrated and why the full_game_proj path exists
---

## Rule
Always pass `full_game_proj` (from `total_lean["projected_runs"]`) when calling `build_f5_lean`. Do NOT rely on the fallback path (raw `home_team_score / 3.4 × 0.55`) for production use.

## Why
The team offensive index (`home_team_score`) in the sim/live data runs 16–30 per team (not the 12–22 the code comment claimed). Using `score/3.4 × 0.55` pushed both teams to the per-team cap (3.2 R) on almost every game, resulting in all-OVER projections of 5.5–7.7.

The calibrated path:
- `base_f5 = full_game_proj × 0.53`
- `full_game_proj` is already pitcher-adjusted, park-adjusted, weather-adjusted by `build_total_lean`
- 0.53 = empirical ratio: MLB F5 avg ~4.6 / full-game avg ~8.7

On top of the base, `_pitcher_quality_index()` adds a Statcast delta (±0.10 R per team) for xwOBA/barrel/hard-hit signals that hits/K don't capture.

## How to apply
- `build_game_boards` calls `build_total_lean` first, then passes `total_lean["projected_runs"]` into `build_f5_lean` as `full_game_proj`.
- Unit test with realistic fg values: fg=8.6 → LEAN ~4.6, fg=7.0 ace duel → UNDER ~3.5, fg=10.5 → OVER ~5.6.
