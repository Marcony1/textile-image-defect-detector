"""SQLite log of every inspected towel (one row per photo) and operator reviews.

SQLite is enough for one inspection line (~1 towel every few seconds) and needs
no server. Every connection is short-lived, so the worker thread and the web
server can share the file safely (WAL mode).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,          -- local time, ISO 8601
    file         TEXT NOT NULL,
    decision     TEXT NOT NULL,          -- 'OK' | 'NO OK' | 'ERROR'
    score        REAL,
    threshold    REAL,
    operating_point TEXT,
    model_scores TEXT,                   -- JSON {model: raw score}
    defects      TEXT,                   -- JSON [{cls, conf, box}]
    top_defect   TEXT,
    elapsed_ms   REAL,
    image_path   TEXT,                   -- archived original
    preview_path TEXT,                   -- annotated JPEG for the dashboard
    error        TEXT,
    review       TEXT,                   -- 'defecto_real' | 'falsa_alarma' | 'ok_confirmado' | 'defecto_no_detectado'
    review_note  TEXT,
    review_ts    TEXT
);
CREATE INDEX IF NOT EXISTS idx_ts ON inspections(ts);
"""

REVIEW_LABELS = {"defecto_real", "falsa_alarma", "ok_confirmado", "defecto_no_detectado"}


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # ------------------------------------------------------------------ writes
    def add(self, **row) -> int:
        row.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
        for k in ("model_scores", "defects"):
            if k in row and not isinstance(row[k], str):
                row[k] = json.dumps(row[k])
        cols = ", ".join(row)
        with self._conn() as c:
            cur = c.execute(f"INSERT INTO inspections ({cols}) VALUES ({', '.join('?' * len(row))})",
                            list(row.values()))
            return int(cur.lastrowid)

    def set_preview(self, iid: int, preview_path: str):
        with self._conn() as c:
            c.execute("UPDATE inspections SET preview_path=? WHERE id=?", (preview_path, iid))

    def review(self, iid: int, label: str, note: str = "") -> bool:
        if label not in REVIEW_LABELS:
            raise ValueError(f"review must be one of {sorted(REVIEW_LABELS)}")
        with self._conn() as c:
            cur = c.execute("UPDATE inspections SET review=?, review_note=?, review_ts=? WHERE id=?",
                            (label, note, datetime.now().isoformat(timespec="seconds"), iid))
            return cur.rowcount == 1

    # ------------------------------------------------------------------- reads
    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        for k in ("model_scores", "defects"):
            d[k] = json.loads(d[k]) if d.get(k) else ({} if k == "model_scores" else [])
        return d

    def get(self, iid: int) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM inspections WHERE id=?", (iid,)).fetchone()
        return self._row(r) if r else None

    def recent(self, limit: int = 50, since: str | None = None, decision: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM inspections WHERE 1=1", []
        if since:
            q += " AND ts >= ?"; args.append(since)
        if decision:
            q += " AND decision = ?"; args.append(decision)
        q += " ORDER BY id DESC LIMIT ?"; args.append(limit)
        with self._conn() as c:
            return [self._row(r) for r in c.execute(q, args).fetchall()]

    def rows_since(self, since: str) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT id, ts, decision, top_defect, elapsed_ms, review, score, threshold "
                "FROM inspections WHERE ts >= ? ORDER BY id", (since,)).fetchall()]


def window_start(window: str) -> str:
    """'turno' = last 8 h, 'hoy' = since midnight, '24h', '7d'."""
    now = datetime.now()
    start = {"turno": now - timedelta(hours=8), "24h": now - timedelta(hours=24),
             "7d": now - timedelta(days=7)}.get(window)
    if start is None:  # 'hoy'
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.isoformat(timespec="seconds")
