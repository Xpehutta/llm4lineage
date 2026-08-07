"""SQL2Graph package — column-level lineage pipeline.

Public API is re-exported for ``from Classes.sql2graph import ...``.
``Classes.sql2graph_classes`` remains a compatibility shim.
"""
from Classes.sql2graph.builder import SQL2GraphBuilder
from Classes.sql2graph.llm_extractor import SQL2GraphLLMExtractor
from Classes.sql2graph.models import (
    ColumnRef,
    FilterSpec,
    JoinSpec,
    OutputColumn,
    SQL2GraphExtraction,
    SQL2GraphExtractionCTE,
)
from Classes.sql2graph.parser import SQL2GraphParser
from Classes.sql2graph.pipeline import SQL2GraphPipeline, pipeline_result_quality
from Classes.sql2graph.validator import SQL2GraphValidator
from Classes.sql2graph.visualizer import SQL2GraphVisualizer

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
