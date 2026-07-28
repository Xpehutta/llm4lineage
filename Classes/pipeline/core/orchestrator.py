"""End-to-end coordinator for SQL analysis."""

import logging
import time
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from Classes.pipeline.core.chain import SQLAnalysisChain
from Classes.pipeline.core.lineage import ColumnLineageExtractor
from Classes.pipeline.core.llm_factory import LLMFactory
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


def _resolve_model_label(llm: BaseChatModel) -> str:
    """Return a safe model identifier without serializing credentials."""
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
        llm: Optional[BaseChatModel] = None,
    ):
        self.config = config
        self.parser = SQLParser(
            dialect=config.sql_dialect,
            error_on_incomplete=config.error_on_incomplete,
        )
        self.serializer = ASTSerializer(max_depth=config.ast_max_depth)
        self.lineage_extractor = ColumnLineageExtractor(
            dialect=config.sql_dialect,
            include_intermediate=config.lineage_include_intermediate_columns,
        )
        self.llm = llm or LLMFactory.create(config)
        self.chain = SQLAnalysisChain(config, self.llm)

    def run(self, sql: str, instruction: str = "") -> PipelineResult:
        """Execute the full pipeline and always return a PipelineResult.

        On failure the ``error`` field is populated and the remaining
        fields carry sensible defaults (empty dict / list / string).
        """
        model_label = _resolve_model_label(self.llm)

        try:
            ast = self.parser.parse(sql)
            ast_json = self.serializer.serialize(ast)
            column_lineage = self.lineage_extractor.extract(ast)

            start = time.perf_counter()
            llm_response = self.chain.run(ast_json, column_lineage, instruction)
            latency = time.perf_counter() - start

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
