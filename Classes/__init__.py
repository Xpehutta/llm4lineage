"""
SQL Lineage Tool – Classes Package

This package contains the core classes for SQL lineage extraction and refinement.
"""

__version__ = "0.2.0"

from .model_classes import (
    SourceTableStructure,
    SQLDependencies,
    SQLLineageExtractor,
    SQLLineageOutputParser,
    SQLLineageResult,
    ViewJoin,
    ViewOutputColumn,
    ViewStructure,
    create_sql_lineage_extractor,
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
from .plpgsql_lineage import (
    PlpgsqlLineageExtractor,
    UnresolvedItem,
    contains_plpgsql_function,
    extract_plpgsql_lineage,
)
from .plpgsql_splitter import (
    PlpgsqlStmt,
    extract_function_def,
    find_function_defs,
    is_plpgsql_function,
    split_function_body,
)
from .schema_registry import DDLParser, SchemaRegistry
from .sql2graph_classes import (
    SQL2GraphBuilder,
    SQL2GraphExtraction,
    SQL2GraphLLMExtractor,
    SQL2GraphParser,
    SQL2GraphPipeline,
    SQL2GraphValidator,
    SQL2GraphVisualizer,
)
from .sql_chunk_classes import (
    SQLChunk,
    SQLChunkEdge,
    SQLChunkGraph,
    SQLChunkLink,
    SQLLogicalChunkParser,
    SQLLogicalChunkPreParser,
)
from .view_expander import ViewExpander

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
    "DDLParser",
    "SchemaRegistry",
    "ViewExpander",
    "PlpgsqlStmt",
    "split_function_body",
    "extract_function_def",
    "find_function_defs",
    "is_plpgsql_function",
    "PlpgsqlLineageExtractor",
    "UnresolvedItem",
    "extract_plpgsql_lineage",
    "contains_plpgsql_function",
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
