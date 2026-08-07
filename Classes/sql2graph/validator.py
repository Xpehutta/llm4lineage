"""Graph integrity checks for SQL2Graph outputs."""
from __future__ import annotations

from typing import Any

import networkx as nx
from pydantic import ValidationError

from Classes.sql2graph.builder import SQL2GraphBuilder
from Classes.sql2graph.models import SQL2GraphExtraction


class SQL2GraphValidator:
    """Deterministic checks for extraction payload and graph integrity."""

    @staticmethod
    def validate_extraction(extraction: dict[str, Any]) -> tuple[bool, str]:
        try:
            SQL2GraphExtraction.model_validate(extraction)
            return True, "valid"
        except ValidationError as exc:
            return False, str(exc)

    @staticmethod
    def validate_graph(graph: nx.MultiDiGraph, schema: dict[str, Any] | None = None) -> list[str]:
        warnings = []
        for node, attrs in graph.nodes(data=True):
            node_type = attrs.get("node_type")
            if node_type and node_type not in SQL2GraphBuilder.ALL_NODE_TYPES:
                warnings.append(f"Unknown node_type: {node_type} ({node})")

        for source, target, attrs in graph.edges(data=True):
            edge_type = attrs.get("edge_type")
            if edge_type and edge_type not in SQL2GraphBuilder.ALL_EDGE_TYPES:
                warnings.append(f"Unknown edge_type: {edge_type} ({source} -> {target})")
            if source not in graph.nodes:
                warnings.append(f"Dangling edge source: {source}")
            if target not in graph.nodes:
                warnings.append(f"Dangling edge target: {target}")
            # Consumers weigh edges by these two; an edge without them is
            # indistinguishable from a confident, code-confirmed one.
            for required in ("confidence", "provenance"):
                if required not in attrs:
                    warnings.append(f"Edge missing {required}: {source} -> {target}")

        if schema and isinstance(schema, dict):
            alias_columns: dict[str, set] = {}
            for table in schema.get("tables", []):
                alias = table.get("alias") or table.get("name")
                cols = {col.get("name") for col in table.get("columns", []) if col.get("name")}
                if alias:
                    alias_columns[alias] = cols

            for node, attrs in graph.nodes(data=True):
                if attrs.get("node_type") != "source_column":
                    continue
                alias = attrs.get("table_alias")
                column = attrs.get("column")
                if alias in alias_columns and column not in alias_columns[alias]:
                    warnings.append(f"Unknown column reference: {node}")

        if graph.number_of_edges() > 0 and not nx.is_directed_acyclic_graph(graph):
            try:
                cycle = nx.find_cycle(graph)
                warnings.append(f"Graph contains a directed cycle: {cycle[:3]}")
            except nx.NetworkXNoCycle:
                pass

        return warnings
