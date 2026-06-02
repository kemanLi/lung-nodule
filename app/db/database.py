import sqlite3
from datetime import datetime
from pathlib import Path


class HistoryDatabase:
    def __init__(self, path: str = "outputs/history.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS studies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT NOT NULL,
                    overlay_path TEXT NOT NULL,
                    nodule_count INTEGER NOT NULL,
                    mode TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def add_study(self, source_path: str, overlay_path: str, nodule_count: int, mode: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO studies(source_path, overlay_path, nodule_count, mode, created_at) VALUES (?, ?, ?, ?, ?)",
                (source_path, overlay_path, nodule_count, mode, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )

    def list_studies(self, limit: int = 20) -> list[tuple]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT created_at, nodule_count, mode, source_path, overlay_path FROM studies ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
