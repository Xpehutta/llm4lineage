"""
SQL Lineage Tool – Classes Package

This package contains the core classes for SQL lineage extraction and refinement.
"""

# Version of the package (optional, but useful)
__version__ = "0.1.0"

# Import main classes for easy access
from .model_classes import (
    SQLLineageExtractor,
    SQLLineageResult,
    SQLDependencies,
    SQLLineageOutputParser,
    create_sql_lineage_extractor,   # if you have this factory function
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
from .dellm_classes import (
    DELLMKnowledge,
    DELLMGenerator,
)
from .views_structure_classes import (
    ViewOutputColumn,
    ViewJoin,
    SourceTableStructure,
    ViewStructure,
    ViewsStructureExtractor,
)
from .sql_chunk_classes import (
    SQLChunk,
    SQLChunkLink,
    SQLChunkEdge,
    SQLChunkGraph,
    SQLLogicalChunkPreParser,
    SQLLogicalChunkParser,
)

# Define what gets exported with "from Classes import *"
__all__ = [
    "SQLLineageExtractor",
    "SQLLineageResult",
    "SQLDependencies",
    "SQLLineageOutputParser",
    "create_sql_lineage_extractor",
    "SQL2GraphParser",
    "SQL2GraphLLMExtractor",
    "SQL2GraphBuilder",
    "SQL2GraphVisualizer",
    "SQL2GraphValidator",
    "SQL2GraphPipeline",
    "SQL2GraphExtraction",
    "DELLMKnowledge",
    "DELLMGenerator",
    "ViewOutputColumn",
    "ViewJoin",
    "SourceTableStructure",
    "ViewStructure",
    "ViewsStructureExtractor",
    "SQLChunk",
    "SQLChunkLink",
    "SQLChunkEdge",
    "SQLChunkGraph",
    "SQLLogicalChunkPreParser",
    "SQLLogicalChunkParser",
]
