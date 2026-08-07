"""Persistent on-disk cache for LLM extraction and pipeline results."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


class LLMCache:
    """SQLite-backed cache keyed by SQL + prompt version + model (+ pipeline options)."""

    def __init__(self, path: str | None = None):
        default = Path.home() / ".cache" / "llm4lineage" / "llm_cache.sqlite"
        self.path = Path(path) if path else default
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                  cache_key TEXT PRIMARY KEY,
                  payload TEXT NOT NULL,
                  quality_score REAL DEFAULT 0,
                  entry_type TEXT DEFAULT 'extraction',
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(llm_cache)")}
            if "quality_score" not in columns:
                conn.execute("ALTER TABLE llm_cache ADD COLUMN quality_score REAL DEFAULT 0")
            if "entry_type" not in columns:
                conn.execute("ALTER TABLE llm_cache ADD COLUMN entry_type TEXT DEFAULT 'extraction'")

    @staticmethod
    def make_key(sql: str, *, prompt_version: str, model: str) -> str:
        digest = hashlib.sha256(f"{prompt_version}|{model}|{sql}".encode()).hexdigest()
        return digest

    @staticmethod
    def make_pipeline_key(
        sql: str,
        *,
        prompt_version: str,
        model: str,
        dialect: str,
        use_llm_verify: bool,
        use_llm_enhance: bool,
    ) -> str:
        raw = (
            f"pipeline|{prompt_version}|{model}|{dialect}|"
            f"verify={use_llm_verify}|enhance={use_llm_enhance}|{sql}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        entry = self.get_entry(cache_key)
        if entry is None:
            return None
        return entry["payload"]

    def get_entry(self, cache_key: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT payload, quality_score, entry_type, created_at FROM llm_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if not row:
            return None
        return {
            "payload": json.loads(row[0]),
            "quality_score": float(row[1] or 0.0),
            "entry_type": row[2] or "extraction",
            "created_at": row[3],
        }

    def set(
        self,
        cache_key: str,
        payload: dict[str, Any],
        *,
        quality_score: float = 0.0,
        entry_type: str = "extraction",
    ) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache(cache_key, payload, quality_score, entry_type)
                VALUES (?, ?, ?, ?)
                """,
                (cache_key, json.dumps(payload), quality_score, entry_type),
            )

    def set_if_better(
        self,
        cache_key: str,
        payload: dict[str, Any],
        *,
        quality_score: float,
        entry_type: str = "extraction",
    ) -> dict[str, Any]:
        """Store payload only when quality is >= existing entry (or no entry yet)."""
        existing = self.get_entry(cache_key)
        if existing is not None and quality_score < existing["quality_score"]:
            return {
                "updated": False,
                "quality_score": quality_score,
                "previous_quality_score": existing["quality_score"],
            }
        self.set(cache_key, payload, quality_score=quality_score, entry_type=entry_type)
        return {
            "updated": True,
            "quality_score": quality_score,
            "previous_quality_score": existing["quality_score"] if existing else None,
        }
