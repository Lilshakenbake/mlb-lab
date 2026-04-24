"""SQLite-backed watchlist for tracking picks and grading them later."""

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
                """
            )
        _INITIALIZED = True


def _to_float(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (ValueError, TypeError):
        return None


def add_play(payload: dict) -> dict:
    """Insert a play. Dedupe by (game_pk, kind, headline, stat_label, pick, line) on same date."""
    init_db()
    today = (payload.get("game_date") or date.today().isoformat())[:10]
    headline = (payload.get("headline") or "").strip()
    if not headline:
        return {"ok": False, "error": "missing headline"}

    row = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        "game_pk": payload.get("game_pk"),
        "game_date": today,
        "matchup": payload.get("matchup"),
        "kind": payload.get("kind"),
        "headline": headline,
        "stat_label": payload.get("stat_label"),
        "pick": payload.get("pick"),
        "line": _to_float(payload.get("line")),
        "projection": _to_float(payload.get("projection")),
        "edge": _to_float(payload.get("edge")),
        "probability": _to_float(payload.get("probability")) or 0.0,
        "model_used": 1 if payload.get("model_used") else 0,
    }

    with _connect() as conn:
        existing = conn.execute(
            """
            SELECT id FROM tracked_plays
            WHERE game_date=? AND IFNULL(game_pk,-1)=IFNULL(?,-1)
              AND IFNULL(kind,'')=IFNULL(?,'')
              AND IFNULL(headline,'')=IFNULL(?,'')
              AND IFNULL(stat_label,'')=IFNULL(?,'')
              AND IFNULL(pick,'')=IFNULL(?,'')
              AND IFNULL(line,-999)=IFNULL(?,-999)
            """,
            (
                row["game_date"], row["game_pk"], row["kind"],
                row["headline"], row["stat_label"], row["pick"], row["line"],
            ),
        ).fetchone()
        if existing:
            return {"ok": True, "id": existing["id"], "duplicate": True}

        cur = conn.execute(
            """
            INSERT INTO tracked_plays
              (created_at, game_pk, game_date, matchup, kind, headline,
               stat_label, pick, line, projection, edge, probability, model_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["created_at"], row["game_pk"], row["game_date"], row["matchup"],
                row["kind"], row["headline"], row["stat_label"], row["pick"],
                row["line"], row["projection"], row["edge"], row["probability"],
                row["model_used"],
            ),
        )
        return {"ok": True, "id": cur.lastrowid, "duplicate": False}


def list_plays(only_pending: bool = False, limit: int | None = None) -> list[dict]:
    init_db()
    sql = "SELECT * FROM tracked_plays"
    params = []
    if only_pending:
        sql += " WHERE result IS NULL"
    sql += " ORDER BY (result IS NOT NULL) ASC, game_date DESC, created_at DESC"
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def settle_play(play_id: int, result: str, actual_value=None, notes: str | None = None) -> bool:
    init_db()
    result = (result or "").upper().strip()
    if result not in VALID_RESULTS:
        return False
    with _connect() as conn:
        cur = conn.execute(
            """
            UPDATE tracked_plays
            SET result=?, actual_value=?, notes=?, settled_at=?
            WHERE id=?
            """,
            (result, _to_float(actual_value), notes, datetime.utcnow().isoformat(timespec="seconds"), play_id),
        )
        return cur.rowcount > 0


def delete_play(play_id: int) -> bool:
    init_db()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM tracked_plays WHERE id=?", (play_id,))
        return cur.rowcount > 0


def reopen_play(play_id: int) -> bool:
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE tracked_plays SET result=NULL, actual_value=NULL, notes=NULL, settled_at=NULL WHERE id=?",
            (play_id,),
        )
        return cur.rowcount > 0


def summary_stats() -> dict:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT
              COUNT(*)                                 AS total,
              SUM(CASE WHEN result IS NULL THEN 1 ELSE 0 END) AS pending,
              SUM(CASE WHEN result='WIN'  THEN 1 ELSE 0 END) AS wins,
              SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END) AS losses,
              SUM(CASE WHEN result='PUSH' THEN 1 ELSE 0 END) AS pushes,
              AVG(CASE WHEN result IS NOT NULL THEN probability END) AS avg_pred_prob,
              AVG(CASE WHEN result IS NOT NULL AND model_used=1 THEN probability END) AS avg_pred_prob_model
            FROM tracked_plays
            """
        ).fetchone()

        model_row = conn.execute(
            """
            SELECT
              SUM(CASE WHEN result='WIN'  THEN 1 ELSE 0 END) AS wins,
              SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END) AS losses,
              SUM(CASE WHEN result='PUSH' THEN 1 ELSE 0 END) AS pushes
            FROM tracked_plays
            WHERE model_used=1 AND result IS NOT NULL
            """
        ).fetchone()

    total = row["total"] or 0
    pending = row["pending"] or 0
    wins = row["wins"] or 0
    losses = row["losses"] or 0
    pushes = row["pushes"] or 0
    decided = wins + losses
    win_rate = round((wins / decided) * 100, 1) if decided > 0 else None
    avg_pred = round(row["avg_pred_prob"], 1) if row["avg_pred_prob"] is not None else None

    model_wins = model_row["wins"] or 0
    model_losses = model_row["losses"] or 0
    model_decided = model_wins + model_losses
    model_win_rate = round((model_wins / model_decided) * 100, 1) if model_decided > 0 else None
    avg_pred_model = round(row["avg_pred_prob_model"], 1) if row["avg_pred_prob_model"] is not None else None

    return {
        "total": total,
        "pending": pending,
        "settled": total - pending,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": win_rate,
        "avg_pred_prob": avg_pred,
        "model_wins": model_wins,
        "model_losses": model_losses,
        "model_pushes": model_row["pushes"] or 0,
        "model_win_rate": model_win_rate,
        "avg_pred_prob_model": avg_pred_model,
    }
