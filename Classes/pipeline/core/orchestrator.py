"""End-to-end coordinator for SQL analysis."""

import logging
import time
from typing import Any, Dict, List, Optional

from Classes.pipeline.core.chain import SQLAnalysisChain
from Classes.pipeline.core.lineage import ColumnLineageExtractor
from Classes.pipeline.core.llm_factory import LLMFactory
from Classes.pipeline.core.llm_interface import LLMInterface, adapt_llm
from Classes.pipeline.core.parser import SQLParser
from Classes.pipeline.core.serializer import ASTSerializer
from Classes.pipeline.exceptions import (
    InvalidResponseError,
    LineageExtractionError,
    LLMCommunicationError,
    ParsingError,
    SerializationError,
)
from Classes.pipeline.models.config import Config
from Classes.pipeline.models.result import PipelineResult

logger = logging.getLogger(__name__)


def _resolve_model_label(llm: Any) -> str:
    """Return a safe model identifier without serializing credentials."""
    label = getattr(llm, "model_label", None)
    if isinstance(label, str) and label:
        return label

    for attr in ("model_name", "model_id", "model"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value

    inner = getattr(llm, "llm", None)
    if inner is not None:
        for attr in ("repo_id", "model", "model_id"):
            value = getattr(inner, attr, None)
            if isinstance(value, str) and value:
                return value

    return type(llm).__name__


class PipelineOrchestrator:
    """End-to-end coordinator for SQL analysis."""

    def __init__(
        self,
        config: Config,
        llm: Optional[LLMInterface] = None,
        schema_catalog: Optional[Dict[str, List[str]]] = None,
    ):
        self.config = config
        self.schema_catalog = schema_catalog
        self.parser = SQLParser(
            dialect=config.sql_dialect,
            error_on_incomplete=config.error_on_incomplete,
        )
        self.serializer = ASTSerializer(max_depth=config.ast_max_depth)
        self.lineage_extractor = ColumnLineageExtractor(
            dialect=config.sql_dialect,
            include_intermediate=config.lineage_include_intermediate_columns,
            schema_catalog=schema_catalog,
        )
        self.llm = adapt_llm(llm) if llm is not None else LLMFactory.create(config)
        self.chain = SQLAnalysisChain(config, self.llm)

    @staticmethod
    def _parse_structured(llm_response: str) -> Dict[str, Any]:
        """Parse the response against the lineage schema.

        Parsing never raises: a malformed response comes back with a
        ``parse_error`` and a confidence low enough to act on.
        """
        from Classes.model_classes import SQLLineageOutputParser

        return SQLLineageOutputParser().parse(llm_response).model_dump()

    def run(self, sql: str, instruction: str = "") -> PipelineResult:
        """Execute the full pipeline and always return a PipelineResult.

        Flow:
            1. sqlglot parse + column lineage (deterministic)
            2. LLM verifies the deterministic lineage against the AST
            3. LLM suggests enhancements only where needed

        On failure the ``error`` field is populated and the remaining
        fields carry sensible defaults (empty dict / list / string).
        """
        model_label = _resolve_model_label(self.llm)

        if not instruction:
            instruction = (
                "Verify the sqlglot column lineage against the SQL. "
                "List any missing dependencies or semantic gaps and propose corrections."
            )
        try:
            ast = self.parser.parse(sql)
            ast_json = self.serializer.serialize(ast)
            column_lineage = self.lineage_extractor.extract(ast)

            start = time.perf_counter()
            llm_response = self.chain.run(ast_json, column_lineage, instruction)
            latency = time.perf_counter() - start

            structured = self._parse_structured(llm_response)

            logger.info(
                "Pipeline completed in %.3fs for query: %.60s…",
                latency,
                sql.replace("\n", " "),
            )

            return PipelineResult(
                original_sql=sql,
                ast_json=ast_json,
                column_lineage=column_lineage,
                llm_response=llm_response,
                latency_seconds=round(latency, 4),
                model_used=model_label,
                llm_structured=structured,
                parse_error=structured.get("parse_error"),
            )

        except (
            ParsingError,
            SerializationError,
            LineageExtractionError,
            LLMCommunicationError,
            InvalidResponseError,
        ) as exc:
            logger.error("Pipeline error: %s", exc)
            return PipelineResult(
                original_sql=sql,
                ast_json={},
                column_lineage=[],
                llm_response="",
                latency_seconds=0.0,
                model_used=model_label,
                error=str(exc),
            )
        except Exception as exc:
            logger.exception("Unexpected pipeline failure")
            return PipelineResult(
                original_sql=sql,
                ast_json={},
                column_lineage=[],
                llm_response="",
                latency_seconds=0.0,
                model_used=model_label,
                error=f"Unexpected error: {exc}",
            )
