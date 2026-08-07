"""FastAPI surface for lineage impact, coverage, PII, and graph rendering.

Run with::

    uvicorn Web.api.main:app --reload

The handlers operate on an in-memory :class:`LineageStore`. Load graphs via
``POST /lineage/{object}`` (test helper) or wire a durable store behind the same
endpoints in production.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field

from Classes.impact_analyzer import analyze_impact, table_level_impact
from Web.api.render import graph_to_dot, graph_to_mermaid
from Web.api.store import STORE, LineageStore

app = FastAPI(title="llm4lineage API", version="0.1.0")


class GraphPayload(BaseModel):
    graph: dict[str, Any]
    column_meta: dict[str, dict[str, Any]] = Field(default_factory=dict)


def get_store() -> LineageStore:
    return STORE


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/lineage/{object_name}")
def put_lineage(object_name: str, payload: GraphPayload) -> dict[str, Any]:
    """Load a node-link graph into the store (primarily for tests / bootstrap)."""
    store = get_store()
    store.put_graph(object_name, payload.graph)
    for column, meta in payload.column_meta.items():
        store.put_column_meta(object_name, column, meta)
    return {"object": object_name, "nodes": len(payload.graph.get("nodes") or [])}


@app.get("/lineage/{object_name}")
def get_lineage(
    object_name: str,
    format: Literal["json", "dot", "mermaid"] = Query(default="json"),
) -> Any:
    graph = get_store().get_graph(object_name)
    if graph is None:
        raise HTTPException(status_code=404, detail=f"Unknown object: {object_name}")
    if format == "dot":
        return Response(content=graph_to_dot(graph), media_type="text/plain")
    if format == "mermaid":
        return Response(content=graph_to_mermaid(graph), media_type="text/plain")
    return graph


@app.get("/impact/{object_name}/{column}")
def get_impact(
    object_name: str,
    column: str,
    direction: Literal["up", "down", "both", "upstream", "downstream"] = "both",
) -> dict[str, Any]:
    graph = get_store().get_graph(object_name)
    if graph is None:
        raise HTTPException(status_code=404, detail=f"Unknown object: {object_name}")

    target = column if "." in column else f"output.{column}"
    # Fall back to bare column id or table.column forms present in the graph.
    node_ids = {str(node.get("id")) for node in graph.get("nodes") or []}
    if target not in node_ids:
        for candidate in (column, f"{object_name}.{column}", f"output.{column}"):
            if candidate in node_ids:
                target = candidate
                break
    result = analyze_impact(graph, target, direction=direction)
    result["object"] = object_name
    result["table_impact"] = table_level_impact(graph, object_name)
    return result


@app.get("/coverage")
def get_coverage() -> dict[str, Any]:
    return get_store().coverage()


@app.get("/pii")
def get_pii() -> dict[str, Any]:
    return {"columns": get_store().pii_columns()}
