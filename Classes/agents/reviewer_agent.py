"""Reviewer agent: PASS only when a candidate edge is confirmed by source SQL.

Edges with ``verified=False`` must not be published. The Reviewer is the gate:
PASS requires string / table / column evidence for both ``src`` and ``dst`` in
the provided SQL. LLM suggestions may enrich the fragment but cannot override
missing evidence.
"""

from __future__ import annotations

import json
import re
from typing import Any

from Classes.agents._json_util import extract_json
from Classes.agents.models import CandidateEdge, ReviewResult
from Classes.pipeline.core.llm_interface import LLMInterface

__all__ = ["ReviewerAgent", "find_sql_evidence"]

_IDENT_SPLIT = re.compile(r"[.\s]+")


def find_sql_evidence(sql: str, ref: str) -> str | None:
    """Return a SQL substring that evidences ``ref``, or ``None`` if absent.

    Evidence is a table / column / qualified-name fragment appearing in ``sql``.
    Matching is case-insensitive; the returned slice preserves original casing.
    """
    if not sql or not ref:
        return None
    parts = [p for p in _IDENT_SPLIT.split(ref.strip()) if p]
    if not parts:
        return None

    sql_lower = sql.lower()
    # Prefer the longest contiguous qualified fragment that appears literally.
    for length in range(len(parts), 0, -1):
        for start in range(0, len(parts) - length + 1):
            fragment = ".".join(parts[start : start + length])
            idx = sql_lower.find(fragment.lower())
            if idx >= 0:
                return sql[idx : idx + len(fragment)]

    # Fall back to bare identifier word boundaries.
    for part in reversed(parts):  # prefer column / leaf names
        match = re.search(rf"(?i)\b{re.escape(part)}\b", sql)
        if match:
            return match.group(0)
    return None


class ReviewerAgent:
    """Confirm candidate edges against source SQL (read-only)."""

    def __init__(self, llm: LLMInterface | None = None, *, use_llm: bool = True):
        self.llm = llm
        self.use_llm = use_llm and llm is not None

    def review(self, edge: CandidateEdge | dict[str, Any], source_sql: str) -> ReviewResult:
        """Return PASS/FAIL for ``edge`` based on evidence in ``source_sql``."""
        candidate = (
            edge if isinstance(edge, CandidateEdge) else CandidateEdge.model_validate(edge)
        )
        src_hit = find_sql_evidence(source_sql, candidate.src)
        dst_hit = find_sql_evidence(source_sql, candidate.dst)

        if not src_hit or not dst_hit:
            missing = []
            if not src_hit:
                missing.append(f"src={candidate.src!r}")
            if not dst_hit:
                missing.append(f"dst={candidate.dst!r}")
            return ReviewResult.fail(
                reason=f"No code evidence for {', '.join(missing)}",
                sql_fragment=candidate.sql_fragment or "",
            )

        fragment = self._resolve_fragment(candidate, source_sql, src_hit, dst_hit)
        return ReviewResult.pass_(
            sql_fragment=fragment,
            reason=f"Confirmed by code evidence: {src_hit!r} → {dst_hit!r}",
        )

    def publishable(self, edge: CandidateEdge | dict[str, Any], source_sql: str) -> CandidateEdge | None:
        """Return a verified copy of ``edge`` on PASS; otherwise ``None``."""
        candidate = (
            edge if isinstance(edge, CandidateEdge) else CandidateEdge.model_validate(edge)
        )
        result = self.review(candidate, source_sql)
        if result.verdict != "PASS" or not result.verified:
            return None
        published = candidate.model_copy(
            update={
                "verified": True,
                "sql_fragment": result.sql_fragment or candidate.sql_fragment,
                "provenance": "llm_verified",
            }
        )
        return published

    def _resolve_fragment(
        self,
        edge: CandidateEdge,
        source_sql: str,
        src_hit: str,
        dst_hit: str,
    ) -> str:
        # Prefer the candidate's own fragment when it still appears in SQL.
        if edge.sql_fragment and edge.sql_fragment.lower() in source_sql.lower():
            return edge.sql_fragment

        if self.use_llm and self.llm is not None:
            suggested = self._ask_llm_for_fragment(edge, source_sql)
            if suggested and suggested.lower() in source_sql.lower():
                return suggested

        # Deterministic fallback: a window around the first evidence hit.
        return self._window_around(source_sql, src_hit) or src_hit or dst_hit

    def _ask_llm_for_fragment(self, edge: CandidateEdge, source_sql: str) -> str:
        prompt = (
            "You are a SQL lineage Reviewer. Locate the smallest SQL fragment that "
            "supports the edge below. Return ONLY JSON: "
            '{"verdict":"PASS"|"FAIL","sql_fragment":"...","reason":"..."}\n\n'
            f"Edge: {json.dumps(edge.model_dump(), ensure_ascii=False)}\n\n"
            f"SQL:\n{source_sql[:12000]}\n"
        )
        try:
            data = extract_json(self.llm.invoke(prompt))
        except ValueError:
            return ""
        if not isinstance(data, dict):
            return ""
        return str(data.get("sql_fragment") or "")

    @staticmethod
    def _window_around(sql: str, needle: str, radius: int = 120) -> str:
        idx = sql.lower().find(needle.lower())
        if idx < 0:
            return needle
        start = max(0, idx - radius)
        end = min(len(sql), idx + len(needle) + radius)
        return sql[start:end].strip()
