"""Agent orchestrator: unresolved queue → Resolver → Reviewer → escalate.

Acceptance: the unresolved queue cannot grow forever — after ``max_attempts``
failures an item is escalated to a human and removed from the queue.
Only Reviewer-PASS edges (``verified=True``) are published.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from Classes.agents.models import (
    CandidateEdge,
    CoverageReport,
    OrchestratorResult,
    UnresolvedInput,
)
from Classes.agents.resolver_agent import ResolverAgent
from Classes.agents.reviewer_agent import ReviewerAgent

__all__ = ["AgentOrchestrator"]


class AgentOrchestrator:
    """Distribute unresolved items through Resolver + Reviewer with escalation."""

    def __init__(
        self,
        resolver: ResolverAgent,
        reviewer: ReviewerAgent,
        *,
        max_attempts: int = 3,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.resolver = resolver
        self.reviewer = reviewer
        self.max_attempts = max_attempts

    def run(
        self,
        function_sql: str,
        unresolved: list[dict[str, Any] | UnresolvedInput],
    ) -> OrchestratorResult:
        """Process the unresolved queue until empty or every item is escalated."""
        items = [UnresolvedInput.from_mapping(item) for item in unresolved]
        queue: deque[UnresolvedInput] = deque(items)
        attempts: dict[str, int] = {}
        published: list[CandidateEdge] = []
        escalated: list[UnresolvedInput] = []
        escalated_keys: set[str] = set()
        resolved_keys: set[str] = set()
        failed_reviews = 0
        # Safety bound: each item can be seen at most max_attempts times.
        max_iterations = max(1, len(items) * self.max_attempts + 1)
        iterations = 0

        while queue and iterations < max_iterations:
            iterations += 1
            item = queue.popleft()
            key = self._item_key(item)
            attempts[key] = attempts.get(key, 0) + 1

            if attempts[key] > self.max_attempts:
                if key not in escalated_keys and key not in resolved_keys:
                    escalated.append(item)
                    escalated_keys.add(key)
                continue

            candidates = self.resolver.resolve(function_sql, [item])
            accepted = 0
            for edge in candidates:
                publishable = self.reviewer.publishable(edge, function_sql)
                if publishable is None:
                    failed_reviews += 1
                    continue
                if publishable.verified is False:
                    # Defence in depth: never publish unverified edges.
                    failed_reviews += 1
                    continue
                published.append(publishable)
                accepted += 1

            if accepted > 0:
                resolved_keys.add(key)
                continue

            if attempts[key] >= self.max_attempts:
                if key not in escalated_keys:
                    escalated.append(item)
                    escalated_keys.add(key)
            else:
                queue.append(item)

        # Anything still queued after the safety bound is escalated.
        while queue:
            leftover = queue.popleft()
            key = self._item_key(leftover)
            if key not in resolved_keys and key not in escalated_keys:
                escalated.append(leftover)
                escalated_keys.add(key)

        coverage = CoverageReport(
            total_unresolved=len(items),
            resolved=len(resolved_keys),
            escalated=len(escalated),
            published_edges=len(published),
            failed_reviews=failed_reviews,
            attempts=dict(attempts),
        ).recompute()

        return OrchestratorResult(
            published_edges=published,
            escalated=escalated,
            coverage=coverage,
        )

    @staticmethod
    def _item_key(item: UnresolvedInput) -> str:
        return (
            f"{item.reason}|{item.line_start}|{item.line_end}|"
            f"{item.sql_fragment[:200]}|{item.detail[:120]}"
        )
