# Workspace

## Overview

pnpm workspace monorepo using TypeScript. Each package manages its own dependencies.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.

## MLB Lab (imported)

Python Flask app under `mlb-lab/` imported from https://github.com/Lilshakenbake/mlb-lab.git.
Provides MLB game predictions (hitter props, pitcher strikeouts, spread leans) using `pybaseball`.

- Run via the `MLB Lab` workflow (`python app.py` on port 5000).
- Login password set via `APP_PASSWORD` env var (defaults to `mlb123`).
- Session secret via `SECRET_KEY` env var (uses `SESSION_SECRET` not required).
- Trained models live in `models/` (gitignored). Train them with:
  `python train_models.py --days 120`.
  Without trained models the app falls back to the original heuristic projections in `src/predict.py`.
- Inference layer: `src/model.py` (loads `.joblib` bundles, exposes
  `hitter_hits`, `hitter_total_bases`, `pitcher_strikeouts`, `over_probability`).
- Performance: `src/cache.py` writes a disk-backed JSON cache to
  `data_cache/` (gitignored). TTLs: schedule 5m, lineup 15m, weather 1h,
  player profile 12h, roster 6h, player-id 30d. `pybaseball.cache.enable()`
  also caches statcast HTTP responses on disk.
- Per-game work runs in a thread pool: weather + both team rosters/profiles
  + both pitcher profiles in parallel. Override the cache directory with
  `MLB_CACHE_DIR` env var if Render gives you a persistent disk mount.
- **Live odds + edges (`src/live_odds.py`)**: pulls FanDuel/DK/etc via The
  Odds API. Computes `edge_pct`, `ev_pct`, and `kelly_units` (half-Kelly,
  capped 5u) for ML/RL/Totals/Player props. Player props gated by
  `PROP_ODDS_ENABLED` env var (default 1). 2hr disk cache for game odds,
  12hr per-event cache for props.
- **HR Threats board** (`app.py`): top 20 hitters by 1+ HR probability.
  Two-pass diversification (top-of-each-game first, then fill by raw
  prob with `HR_THREATS_PER_GAME_CAP=2` per game) prevents
  Coors/Yankee-Rangers eating the whole board. Tunable via env vars
  `HR_THREATS_LIMIT` and `HR_THREATS_PER_GAME_CAP`.
- **Park HR factors (`src/park_factors.py`)**: handedness-split HR factors
  (`hr_lhb`/`hr_rhb`) for short-porch parks (Yankee LHB 1.40 vs RHB 1.05,
  Fenway LHB 0.85 vs RHB 1.05, Oracle LHB 0.78, etc.). Helper
  `get_hr_factor(park, hand)` returns the right value; switch hitters get
  the better side. Wired into `compute_hr_threat` and `build_hitter_prop`
  HR path in `src/predict.py`.
- **Bet tracker (`src/tracker.py`)**: SQLite at `tracked_data/tracked_plays.db`.
  Tracks per-pick odds, units (Kelly-suggested), and CLV (closing-line value).
  CLV columns: `opening_odds`, `opening_book`, `closing_odds`, `closing_book`,
  `clv_pp`. Watchlist auto-snaps closing lines on render and via the "Snap
  closing lines" button (`/api/clv/snap`). Prop CLV requires the closing line
  point to match the opening line — half-run line moves leave the pick
  pending instead of producing phantom CLV.
