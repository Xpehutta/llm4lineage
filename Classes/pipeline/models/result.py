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

    @property
    def success(self) -> bool:
        """Convenience flag: True when no error was recorded."""
        return self.error is None
