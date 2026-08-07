"""Shared Pydantic contracts for Resolver / Reviewer / Doc agents."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "CandidateEdge",
    "ColumnDocLabel",
    "CoverageReport",
    "DocLabels",
    "OrchestratorResult",
    "ReviewResult",
    "UnresolvedInput",
]


class UnresolvedInput(BaseModel):
    """One unresolved fragment handed to the Resolver."""

    sql_fragment: str = ""
    reason: str = ""
    detail: str = ""
    kind: str = ""
    line_start: int = 0
    line_end: int = 0
    function: str = ""

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | UnresolvedInput) -> UnresolvedInput:
        if isinstance(data, UnresolvedInput):
            return data
        return cls.model_validate(data)


class CandidateEdge(BaseModel):
    """A lineage edge proposed by the Resolver (unverified until Reviewer PASS)."""

    src: str
    dst: str
    transform_type: str = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""
    sql_fragment: str = ""
    verified: bool = False
    provenance: str = "llm"

    @field_validator("src", "dst")
    @classmethod
    def non_empty_endpoint(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("edge endpoint cannot be empty")
        return cleaned


class ReviewResult(BaseModel):
    """Reviewer verdict for a single candidate edge."""

    verdict: Literal["PASS", "FAIL"]
    sql_fragment: str = ""
    reason: str = ""
    verified: bool = False

    @classmethod
    def pass_(cls, sql_fragment: str, reason: str = "") -> ReviewResult:
        return cls(verdict="PASS", sql_fragment=sql_fragment, reason=reason, verified=True)

    @classmethod
    def fail(cls, reason: str, sql_fragment: str = "") -> ReviewResult:
        return cls(verdict="FAIL", sql_fragment=sql_fragment, reason=reason, verified=False)


class ColumnDocLabel(BaseModel):
    """Structured documentation labels for one column (or object)."""

    column: str = ""
    is_pii: bool = False
    owner: str = ""
    description: str = ""
    sensitivity: str = ""
    tags: list[str] = Field(default_factory=list)


class DocLabels(BaseModel):
    """Batch of column documentation labels produced by DocAgent."""

    columns: list[ColumnDocLabel] = Field(default_factory=list)
    owner: str = ""
    description: str = ""


class CoverageReport(BaseModel):
    """How many unresolved items the orchestrator closed vs escalated."""

    total_unresolved: int = 0
    resolved: int = 0
    escalated: int = 0
    published_edges: int = 0
    failed_reviews: int = 0
    coverage_ratio: float = 0.0
    attempts: dict[str, int] = Field(default_factory=dict)

    def recompute(self) -> CoverageReport:
        denom = self.total_unresolved or 0
        self.coverage_ratio = (self.resolved / denom) if denom else 1.0
        return self


class OrchestratorResult(BaseModel):
    """Final output of a Resolver→Reviewer run over an unresolved queue."""

    published_edges: list[CandidateEdge] = Field(default_factory=list)
    escalated: list[UnresolvedInput] = Field(default_factory=list)
    coverage: CoverageReport = Field(default_factory=CoverageReport)
