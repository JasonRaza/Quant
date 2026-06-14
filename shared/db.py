"""SQLite metadata tracking for data pipeline."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional


class MetadataStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS symbol_metadata (
                    symbol TEXT PRIMARY KEY,
                    last_date TEXT,
                    last_updated TEXT,
                    row_count INTEGER DEFAULT 0,
                    last_error TEXT
                )
                """
            )
            conn.commit()

    def get_last_date(self, symbol: str) -> Optional[date]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT last_date FROM symbol_metadata WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        if row and row["last_date"]:
            return date.fromisoformat(row["last_date"])
        return None

    def update_symbol(
        self,
        symbol: str,
        last_date: Optional[date],
        row_count: int,
        error: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO symbol_metadata (symbol, last_date, last_updated, row_count, last_error)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    last_date = excluded.last_date,
                    last_updated = excluded.last_updated,
                    row_count = excluded.row_count,
                    last_error = excluded.last_error
                """,
                (
                    symbol,
                    last_date.isoformat() if last_date else None,
                    datetime.now(timezone.utc).isoformat(),
                    row_count,
                    error,
                ),
            )
            conn.commit()

    def list_symbols(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT symbol, last_date, last_updated, row_count, last_error FROM symbol_metadata"
            ).fetchall()
        return [dict(r) for r in rows]
