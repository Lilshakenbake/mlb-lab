"""SQLite-backed watchlist for tracking picks and grading them later.

Also stores external agent picks submitted via the API, and page-view counts.
"""

import json
import os
import sqlite3
import threading
from datetime import datetime, date
from contextlib import contextmanager

DATA_DIR = os.getenv("MLB_DATA_DIR", "tracked_data")
DB_PATH = os.path.join(DATA_DIR, "tracked_plays.db")

_INIT_LOCK = threading.Lock()
_INITIALIZED = False

VALID_RESULTS = {"WIN", "LOSS", "PUSH"}

DEFAULT_ODDS = -110


def american_to_implied(american_odds) -> float | None:
    try:
        o = float(american_odds)
    except (TypeError, ValueError):
        return None
    if o == 0:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    return abs(o) / (abs(o) + 100.0)


def compute_clv_pp(opening_odds, closing_odds) -> float | None:
    p_open = american_to_implied(opening_odds)
    p_close = american_to_implied(closing_odds)
    if p_open is None or p_close is None:
        return None
    return round((p_close - p_open) * 100.0, 2)


def odds_to_profit(american_odds) -> float:
    try:
        o = float(american_odds)
    except (TypeError, ValueError):
        o = float(DEFAULT_ODDS)
    if o == 0:
        return 0.0
    if o > 0:
        return o / 100.0
    return 100.0 / abs(o)


def kelly_fraction(prob: float, american_odds, fraction: float = 0.25) -> float:
    b = odds_to_profit(american_odds)
    q = 1.0 - prob
    if b <= 0:
        return 0.0
    k = (b * prob - q) / b
    return round(max(0.0, k) * fraction, 4)


def kelly_units(prob: float, american_odds,
                fraction: float = 0.25,
                lo: float = 0.25, hi: float = 4.0) -> float:
    raw = kelly_fraction(prob / 100.0, american_odds, fraction)
    return round(max(lo, min(hi, raw * 10)), 2)


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


@contextmanager
def _connect():
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    global _INITIALIZED
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        with _connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tracked_plays (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    game_pk INTEGER,
                    game_date TEXT,
                    matchup TEXT,
                    kind TEXT,
                    headline TEXT,
                    stat_label TEXT,
                    pick TEXT,
                    line REAL,
                    projection REAL,
                    edge REAL,
                    probability REAL,
                    model_used INTEGER DEFAULT 0,
                    result TEXT,
                    actual_value REAL,
                    notes TEXT,
                    settled_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_result ON tracked_plays(result);
                CREATE INDEX IF NOT EXISTS idx_game_date ON tracked_plays(game_date);

                CREATE TABLE IF NOT EXISTS agent_picks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submitted_at TEXT NOT NULL,
                    game_date TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    agent_source TEXT,
                    picks_json TEXT NOT NULL,
                    raw_payload TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_agent_game_date ON agent_picks(game_date);
                CREATE INDEX IF NOT EXISTS idx_agent_name ON agent_picks(agent_name);

                CREATE TABLE IF NOT EXISTS page_views (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts       TEXT NOT NULL,
                    date     TEXT NOT NULL,
                    session  TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_pv_date    ON page_views(date);
                CREATE INDEX IF NOT EXISTS idx_pv_session ON page_views(session);
                """
            )
            # Lightweight migrations
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(tracked_plays)")}
            if "odds" not in cols:
                conn.execute(f"ALTER TABLE tracked_plays ADD COLUMN odds REAL DEFAULT {DEFAULT_ODDS}")
                conn.execute("UPDATE tracked_plays SET odds=? WHERE odds IS NULL", (DEFAULT_ODDS,))
            if "units" not in cols:
                conn.execute("ALTER TABLE tracked_plays ADD COLUMN units REAL DEFAULT 1.0")
                conn.execute("UPDATE tracked_plays SET units=1.0 WHERE units IS NULL")
            if "opening_odds" not in cols:
                conn.execute("ALTER TABLE tracked_plays ADD COLUMN opening_odds REAL")
            if "closing_odds" not in cols:
                conn.execute("ALTER TABLE tracked_plays ADD COLUMN closing_odds REAL")
            if "clv_pp" not in cols:
                conn.execute("ALTER TABLE tracked_plays ADD COLUMN clv_pp REAL")
            if "market_edge_pct" not in cols:
                conn.execute("ALTER TABLE tracked_plays ADD COLUMN market_edge_pct REAL")
        _INITIALIZED = True


# ── Visitor counter ────────────────────────────────────────────────────────────

def record_visit(session_id: str | None = None) -> None:
    """Log one page view. Non-blocking — failures are silently swallowed."""
    try:
        init_db()
        now = datetime.utcnow()
        with _connect() as conn:
            conn.execute(
                "INSERT INTO page_views (ts, date, session) VALUES (?,?,?)",
                (now.isoformat(), now.strftime("%Y-%m-%d"), session_id),
            )
    except Exception:
        pass


def get_visit_stats() -> dict:
    """Return all-time total visits, today's visits, and unique sessions today."""
    try:
        init_db()
        today = date.today().isoformat()
        with _connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM page_views").fetchone()[0]
            today_total = conn.execute(
                "SELECT COUNT(*) FROM page_views WHERE date=?", (today,)
            ).fetchone()[0]
            today_unique = conn.execute(
                "SELECT COUNT(DISTINCT session) FROM page_views WHERE date=? AND session IS NOT NULL",
                (today,),
            ).fetchone()[0]
            # Last 7 days breakdown
            daily = conn.execute(
                """SELECT date, COUNT(*) as visits, COUNT(DISTINCT session) as unique_sessions
                   FROM page_views
                   WHERE date >= date('now','-6 days')
                   GROUP BY date ORDER BY date DESC""",
            ).fetchall()
        return {
            "total_all_time": total,
            "today_total": today_total,
            "today_unique": today_unique,
            "last_7_days": [
                {"date": r["date"], "visits": r["visits"], "unique": r["unique_sessions"]}
                for r in daily
            ],
        }
    except Exception as e:
        return {"error": str(e), "total_all_time": 0, "today_total": 0, "today_unique": 0, "last_7_days": []}


# ── Watchlist ──────────────────────────────────────────────────────────────────

def add_play(payload: dict) -> dict:
    init_db()
    headline = (payload.get("headline") or "").strip()
    if not headline:
        return {"ok": False, "error": "headline required"}

    stat_label = (payload.get("stat_label") or payload.get("stat") or "").strip()
    pick = (payload.get("pick") or "").strip().upper()
    if pick not in ("OVER", "UNDER", "ML", "RL", "YES", "NO", "PASS", ""):
        pick = ""

    try:
        line = float(payload["line"]) if payload.get("line") is not None else None
    except (TypeError, ValueError):
        line = None
    try:
        proj = float(payload["projection"]) if payload.get("projection") is not None else None
    except (TypeError, ValueError):
        proj = None
    try:
        edge = float(payload["edge"]) if payload.get("edge") is not None else None
    except (TypeError, ValueError):
        edge = None
    try:
        prob = float(payload["probability"]) if payload.get("probability") is not None else None
    except (TypeError, ValueError):
        prob = None
    try:
        odds = float(payload["odds"]) if payload.get("odds") is not None else float(DEFAULT_ODDS)
    except (TypeError, ValueError):
        odds = float(DEFAULT_ODDS)
    try:
        units = float(payload["units"]) if payload.get("units") is not None else 1.0
        units = max(0.25, min(10.0, units))
    except (TypeError, ValueError):
        units = 1.0
    try:
        market_edge_pct = float(payload["market_edge_pct"]) if payload.get("market_edge_pct") is not None else None
    except (TypeError, ValueError):
        market_edge_pct = None

    game_pk = payload.get("game_pk")
    matchup = (payload.get("matchup") or "").strip() or None
    kind = (payload.get("kind") or "hitter").strip()
    model_used = int(bool(payload.get("model_used", False)))
    now = datetime.utcnow().isoformat()
    game_date = date.today().isoformat()

    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO tracked_plays
               (created_at, game_pk, game_date, matchup, kind, headline,
                stat_label, pick, line, projection, edge, probability,
                model_used, odds, units, market_edge_pct)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (now, game_pk, game_date, matchup, kind, headline,
             stat_label, pick, line, proj, edge, prob,
             model_used, odds, units, market_edge_pct),
        )
        play_id = cur.lastrowid

    return {"ok": True, "id": play_id}


def get_plays(result_filter: str | None = None, limit: int = 200) -> list[dict]:
    init_db()
    with _connect() as conn:
        if result_filter and result_filter.upper() in VALID_RESULTS | {"OPEN"}:
            if result_filter.upper() == "OPEN":
                rows = conn.execute(
                    "SELECT * FROM tracked_plays WHERE result IS NULL ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tracked_plays WHERE result=? ORDER BY created_at DESC LIMIT ?",
                    (result_filter.upper(), limit),
                ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tracked_plays ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def settle_play(play_id: int, result: str, actual_value=None, notes: str | None = None) -> bool:
    init_db()
    result = (result or "").upper().strip()
    if result not in VALID_RESULTS:
        return False
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE tracked_plays SET result=?, actual_value=?, notes=?, settled_at=? WHERE id=?",
            (result, actual_value, notes, now, play_id),
        )
    return cur.rowcount > 0


def reopen_play(play_id: int) -> bool:
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE tracked_plays SET result=NULL, settled_at=NULL WHERE id=?",
            (play_id,),
        )
    return cur.rowcount > 0


def delete_play(play_id: int) -> bool:
    init_db()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM tracked_plays WHERE id=?", (play_id,))
    return cur.rowcount > 0


def get_stats() -> dict:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT result, COUNT(*) as n FROM tracked_plays WHERE result IS NOT NULL GROUP BY result"
        ).fetchall()
        open_count = conn.execute(
            "SELECT COUNT(*) FROM tracked_plays WHERE result IS NULL"
        ).fetchone()[0]
    counts = {r["result"]: r["n"] for r in rows}
    wins = counts.get("WIN", 0)
    losses = counts.get("LOSS", 0)
    pushes = counts.get("PUSH", 0)
    total = wins + losses + pushes
    return {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "total_settled": total,
        "open": open_count,
        "win_rate": round(wins / total * 100, 1) if total > 0 else None,
    }


# ── Agent picks ────────────────────────────────────────────────────────────────

def add_agent_picks(agent_name: str, picks: list, agent_source: str | None = None,
                    raw_payload: dict | None = None) -> dict:
    init_db()
    if not agent_name:
        return {"ok": False, "error": "agent_name required"}
    now = datetime.utcnow().isoformat()
    today = date.today().isoformat()
    picks_json = json.dumps(picks)
    raw_json = json.dumps(raw_payload) if raw_payload else None

    with _connect() as conn:
        conn.execute(
            "DELETE FROM agent_picks WHERE agent_name=? AND game_date=?",
            (agent_name, today),
        )
        conn.execute(
            """INSERT INTO agent_picks (submitted_at, game_date, agent_name, agent_source, picks_json, raw_payload)
               VALUES (?,?,?,?,?,?)""",
            (now, today, agent_name, agent_source, picks_json, raw_json),
        )
    return {"ok": True, "agent_name": agent_name, "picks_count": len(picks), "game_date": today}


def list_agent_picks(game_date: str | None = None, limit: int = 50) -> list[dict]:
    init_db()
    target_date = game_date or date.today().isoformat()
    with _connect() as conn:
        rows = conn.execute(
            """SELECT id, submitted_at, game_date, agent_name, agent_source, picks_json
               FROM agent_picks
               WHERE game_date = ?
               ORDER BY submitted_at DESC
               LIMIT ?""",
            (target_date, limit),
        ).fetchall()

    result = []
    for r in rows:
        try:
            picks = json.loads(r["picks_json"])
        except (json.JSONDecodeError, TypeError):
            picks = []
        result.append({
            "id": r["id"],
            "submitted_at": r["submitted_at"],
            "game_date": r["game_date"],
            "agent_name": r["agent_name"],
            "agent_source": r["agent_source"],
            "picks": picks,
        })
    return result
