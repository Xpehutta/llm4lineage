"""Resolver agent: propose candidate lineage edges for unresolved SQL fragments.

Honesty principle: when unsure, return low confidence or an empty candidate list —
never invent lineage that is not grounded in the provided SQL.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from Classes.agents._json_util import (
    DEFAULT_TOKEN_BUDGET,
    estimate_tokens,
    extract_json,
    truncate_to_token_budget,
)
from Classes.agents.models import CandidateEdge, UnresolvedInput
from Classes.llm_cache import LLMCache
from Classes.pipeline.core.llm_interface import LLMInterface

__all__ = ["PROMPT_VERSION", "ResolverAgent"]

PROMPT_VERSION = "resolver-v1"

_SYSTEM = (
    "You are a SQL lineage Resolver. Given a function body and unresolved "
    "fragments, propose ONLY candidate lineage edges that are supported by the "
    "SQL text. Do not invent tables or columns. If unsure, omit the edge or set "
    "confidence below 0.5. Return ONLY valid JSON:\n"
    '{"edges":[{"src":"...","dst":"...","transform_type":"...",'
    '"confidence":0.0,"reasoning":"...","sql_fragment":"..."}]}'
)


class ResolverAgent:
    """Turn unresolved fragments into candidate edges via :class:`LLMInterface`."""

    def __init__(
        self,
        llm: LLMInterface,
        *,
        cache: LLMCache | None = None,
        model_label: str = "mock",
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        prompt_version: str = PROMPT_VERSION,
    ):
        self.llm = llm
        self.cache = cache
        self.model_label = model_label or getattr(llm, "model_label", "mock")
        self.token_budget = token_budget
        self.prompt_version = prompt_version

    def resolve(
        self,
        function_sql: str,
        unresolved: list[dict[str, Any] | UnresolvedInput],
    ) -> list[CandidateEdge]:
        """Propose candidate edges for ``unresolved`` within the SQL of ``function_sql``."""
        items = [UnresolvedInput.from_mapping(item) for item in unresolved]
        if not items:
            return []

        prompt = self._build_prompt(function_sql, items)
        cache_key = None
        if self.cache is not None:
            cache_key = LLMCache.make_key(
                prompt,
                prompt_version=self.prompt_version,
                model=self.model_label,
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                return self._edges_from_payload(cached)

        raw = self.llm.invoke(prompt)
        edges = self._parse_response(raw)
        payload = {"edges": [edge.model_dump() for edge in edges]}
        if self.cache is not None and cache_key is not None:
            quality = max((e.confidence for e in edges), default=0.0)
            self.cache.set(
                cache_key,
                payload,
                quality_score=quality,
                entry_type="resolver",
            )
        return edges

    def _build_prompt(self, function_sql: str, items: list[UnresolvedInput]) -> str:
        report = [item.model_dump() for item in items]
        report_json = json.dumps(report, ensure_ascii=False, indent=2)
        # Reserve headroom for the system instructions and report.
        overhead = estimate_tokens(_SYSTEM) + estimate_tokens(report_json) + 256
        sql_budget = max(1_000, self.token_budget - overhead)
        sql_slice = truncate_to_token_budget(function_sql or "", sql_budget)
        prompt = (
            f"{_SYSTEM}\n\n"
            f"### Function SQL\n{sql_slice}\n\n"
            f"### Unresolved report\n{report_json}\n"
        )
        # Hard cap for the whole object.
        return truncate_to_token_budget(prompt, self.token_budget)

    def _parse_response(self, raw: str) -> list[CandidateEdge]:
        try:
            data = extract_json(raw)
        except ValueError:
            return []
        return self._edges_from_payload(data)

    @staticmethod
    def _edges_from_payload(data: Any) -> list[CandidateEdge]:
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            rows = data.get("edges") or data.get("candidates") or []
            if not rows and {"src", "dst"} <= set(data):
                rows = [data]
        else:
            return []

        edges: list[CandidateEdge] = []
        for row in rows:
            edge = ResolverAgent._coerce_edge(row)
            if edge is not None:
                edges.append(edge)
        return edges

    @staticmethod
    def _coerce_edge(row: Any) -> CandidateEdge | None:
        """Parse one candidate; return ``None`` when the row is unusable."""
        if not isinstance(row, dict):
            return None
        try:
            return CandidateEdge.model_validate(
                {
                    "src": row.get("src") or row.get("source") or "",
                    "dst": row.get("dst") or row.get("target") or "",
                    "transform_type": row.get("transform_type") or row.get("type") or "unknown",
                    "confidence": float(row.get("confidence", 0.0) or 0.0),
                    "reasoning": str(row.get("reasoning") or ""),
                    "sql_fragment": str(row.get("sql_fragment") or ""),
                    "verified": False,
                    "provenance": "llm",
                }
            )
        except (ValidationError, TypeError, ValueError):
            # Honesty: drop malformed candidates rather than inventing fields.
            return None
