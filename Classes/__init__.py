"""
SQL Lineage Tool – Classes Package

This package contains the core classes for SQL lineage extraction and refinement.
"""

__version__ = "0.2.0"

from .model_classes import (
    SQLLineageExtractor,
    SQLLineageResult,
    SQLDependencies,
    SQLLineageOutputParser,
    create_sql_lineage_extractor,
    ViewOutputColumn,
    ViewJoin,
    SourceTableStructure,
    ViewStructure,
)
from .sql2graph_classes import (
    SQL2GraphParser,
    SQL2GraphLLMExtractor,
    SQL2GraphBuilder,
    SQL2GraphVisualizer,
    SQL2GraphValidator,
    SQL2GraphPipeline,
    SQL2GraphExtraction,
)
from .sql_chunk_classes import (
    SQLChunk,
    SQLChunkLink,
    SQLChunkEdge,
    SQLChunkGraph,
    SQLLogicalChunkPreParser,
    SQLLogicalChunkParser,
)
from .pipeline import (
    ASTSerializer,
    ColumnLineageExtractor,
    Config,
    LLMFactory,
    PipelineOrchestrator,
    PipelineResult,
    SQLAnalysisChain,
    SQLParser,
    create_chat_model,
    setup_logging,
)

__all__ = [
    "SQLLineageExtractor",
    "SQLLineageResult",
    "SQLDependencies",
    "SQLLineageOutputParser",
    "create_sql_lineage_extractor",
    "ViewOutputColumn",
    "ViewJoin",
    "SourceTableStructure",
    "ViewStructure",
    "SQL2GraphParser",
    "SQL2GraphLLMExtractor",
    "SQL2GraphBuilder",
    "SQL2GraphVisualizer",
    "SQL2GraphValidator",
    "SQL2GraphPipeline",
    "SQL2GraphExtraction",
    "SQLChunk",
    "SQLChunkLink",
    "SQLChunkEdge",
    "SQLChunkGraph",
    "SQLLogicalChunkPreParser",
    "SQLLogicalChunkParser",
    "ASTSerializer",
    "ColumnLineageExtractor",
    "Config",
    "LLMFactory",
    "PipelineOrchestrator",
    "PipelineResult",
    "SQLAnalysisChain",
    "SQLParser",
    "create_chat_model",
    "setup_logging",
]
