"""Backward-compatible import path for SQL2Graph classes.

Prefer ``from Classes.sql2graph import ...``. This module re-exports the package API.
"""
from Classes.sql2graph import (
    ColumnRef,
    FilterSpec,
    JoinSpec,
    OutputColumn,
    SQL2GraphBuilder,
    SQL2GraphExtraction,
    SQL2GraphExtractionCTE,
    SQL2GraphLLMExtractor,
    SQL2GraphParser,
    SQL2GraphPipeline,
    SQL2GraphValidator,
    SQL2GraphVisualizer,
    pipeline_result_quality,
)

__all__ = [
    "ColumnRef",
    "OutputColumn",
    "FilterSpec",
    "JoinSpec",
    "SQL2GraphExtraction",
    "SQL2GraphExtractionCTE",
    "SQL2GraphParser",
    "SQL2GraphLLMExtractor",
    "SQL2GraphBuilder",
    "SQL2GraphVisualizer",
    "SQL2GraphValidator",
    "SQL2GraphPipeline",
    "pipeline_result_quality",
]
