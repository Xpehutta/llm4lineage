"""Persistent on-disk cache for LLM extraction calls."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional


class LLMCache:
  """SQLite-backed cache keyed by SQL + prompt version + model."""

  def __init__(self, path: Optional[str] = None):
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
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
      )

  @staticmethod
  def make_key(sql: str, *, prompt_version: str, model: str) -> str:
    digest = hashlib.sha256(f"{prompt_version}|{model}|{sql}".encode("utf-8")).hexdigest()
    return digest

  def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(self.path) as conn:
      row = conn.execute("SELECT payload FROM llm_cache WHERE cache_key = ?", (cache_key,)).fetchone()
    if not row:
      return None
    return json.loads(row[0])

  def set(self, cache_key: str, payload: Dict[str, Any]) -> None:
    with sqlite3.connect(self.path) as conn:
      conn.execute(
        "INSERT OR REPLACE INTO llm_cache(cache_key, payload) VALUES (?, ?)",
        (cache_key, json.dumps(payload)),
      )
