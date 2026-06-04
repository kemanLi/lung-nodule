from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.config import DB_PATH


class CaseDatabase:
    """SQLite store for analysis cases (richer than the legacy GUI history)."""

    def __init__(self, path: Path | str = DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT UNIQUE NOT NULL,
                    image_name TEXT NOT NULL,
                    image_url TEXT NOT NULL,
                    input_url TEXT NOT NULL,
                    overlay_url TEXT NOT NULL,
                    nodule_count INTEGER NOT NULL,
                    max_confidence REAL NOT NULL,
                    mode TEXT NOT NULL,
                    det_weights TEXT,
                    seg_weights TEXT,
                    detections TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def add_case(self, record: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cases (
                    case_id, image_name, image_url, input_url, overlay_url, nodule_count,
                    max_confidence, mode, det_weights, seg_weights, detections, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["case_id"],
                    record["image_name"],
                    record["image_url"],
                    record.get("input_url", record["image_url"]),
                    record["overlay_url"],
                    record["nodule_count"],
                    record["max_confidence"],
                    record["mode"],
                    record.get("det_weights"),
                    record.get("seg_weights"),
                    json.dumps(record.get("detections", []), ensure_ascii=False),
                    record.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )

    def list_cases(self, limit: int = 50, search: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM cases"
        params: list[Any] = []
        if search:
            query += " WHERE image_name LIKE ?"
            params.append(f"%{search}%")
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def delete_case(self, case_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM cases WHERE case_id = ?", (case_id,))
            return cur.rowcount > 0

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["detections"] = json.loads(data.get("detections") or "[]")
        return data
