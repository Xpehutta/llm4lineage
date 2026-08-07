"""In-memory lineage store used by the FastAPI app.

Production deployments can replace this with a durable edge store; the REST
handlers only depend on the small protocol below so tests stay self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LineageStore:
    """Hold node-link graphs keyed by object name (usually the target table)."""

    graphs: dict[str, dict[str, Any]] = field(default_factory=dict)
    column_meta: dict[str, dict[str, Any]] = field(default_factory=dict)

    def put_graph(self, object_name: str, graph: dict[str, Any]) -> None:
        self.graphs[object_name.lower()] = graph

    def get_graph(self, object_name: str) -> dict[str, Any] | None:
        return self.graphs.get(object_name.lower())

    def put_column_meta(self, object_name: str, column: str, meta: dict[str, Any]) -> None:
        key = f"{object_name.lower()}.{column.lower()}"
        self.column_meta[key] = meta

    def coverage(self) -> dict[str, Any]:
        total_edges = 0
        verified = 0
        unresolved = 0
        for graph in self.graphs.values():
            for link in graph.get("links") or graph.get("edges") or []:
                total_edges += 1
                if link.get("verified") is True:
                    verified += 1
                provenance = str(link.get("provenance") or "")
                if provenance in {"unresolved", "regex"} or link.get("confidence", 1.0) < 0.5:
                    unresolved += 1
        return {
            "objects": len(self.graphs),
            "edges": total_edges,
            "verified_edges": verified,
            "unresolved_edges": unresolved,
            "coverage_ratio": (verified / total_edges) if total_edges else 1.0,
        }

    def pii_columns(self) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        for key, meta in self.column_meta.items():
            if meta.get("is_pii"):
                object_name, column = key.split(".", 1)
                hits.append(
                    {
                        "object": object_name,
                        "column": column,
                        "owner": meta.get("owner"),
                        "description": meta.get("description"),
                    }
                )
        return sorted(hits, key=lambda item: (item["object"], item["column"]))


STORE = LineageStore()
