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
