"""Pipeline result data class."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PipelineResult:
    """Immutable record of a single pipeline execution."""

    original_sql: str
    ast_json: Dict[str, Any] = field(default_factory=dict)
    column_lineage: List[Dict[str, Any]] = field(default_factory=list)
    llm_response: str = ""
    latency_seconds: float = 0.0
    model_used: str = ""
    error: Optional[str] = None
    #: The LLM response parsed against the lineage schema: target, sources,
    #: reasoning, confidence, provenance. Empty when no response was parsed.
    llm_structured: Dict[str, Any] = field(default_factory=dict)
    #: Why structured parsing fell back or failed. Never raised, always reported.
    parse_error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Convenience flag: True when no error was recorded."""
        return self.error is None

    @property
    def llm_confidence(self) -> float:
        """Confidence of the structured extraction; 0.0 when nothing was parsed."""
        return float(self.llm_structured.get("confidence", 0.0) or 0.0)
