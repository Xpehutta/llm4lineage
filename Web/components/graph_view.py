"""Graphviz lineage diagram builders for the Streamlit UI."""

from __future__ import annotations

from typing import Any

import graphviz

from Classes.sql2graph_classes import SQL2GraphVisualizer
from Web.services.pipeline_service import (
    resolve_output_node,
    shorten_text,
    upstream_lineage_nodes,
)


def build_column_lineage_dot(graph_json: dict[str, Any], column_alias: str) -> graphviz.Digraph | None:
    graph = SQL2GraphVisualizer.graph_from_node_link(graph_json)
    output_node = resolve_output_node(graph, column_alias)
    if not output_node:
        return None

    nodes_to_show = upstream_lineage_nodes(graph, output_node) | {output_node}
    dot = graphviz.Digraph(comment=f"Lineage for {column_alias}")
    dot.attr(rankdir="LR")

    out_attrs = graph.nodes[output_node]
    dot.node(
        output_node,
        out_attrs.get("alias") or column_alias,
        shape="box",
        style="filled",
        fillcolor="#ADD8E6",
    )

    for node in sorted(nodes_to_show - {output_node}):
        attrs = graph.nodes[node]
        label = attrs.get("column") or node
        if attrs.get("table_alias"):
            label = f"{attrs['table_alias']}.{label}"
        dot.node(node, label, shape="ellipse", style="filled", fillcolor="#90EE90")

    for source, target, data in graph.edges(data=True):
        if data.get("edge_type") != "DERIVED_FROM":
            continue
        if source in nodes_to_show and target in nodes_to_show:
            dot.edge(source, target)

    return dot


def build_table_lineage_dot(
    target: str,
    sources: list[str],
    *,
    highlight: str | None = None,
) -> graphviz.Digraph:
    """Simple source-table → target-table lineage graph."""
    dot = graphviz.Digraph(comment="Table lineage")
    dot.attr(rankdir="LR")

    if target:
        dot.node(
            target,
            shorten_text(target, 40),
            shape="box",
            style="filled",
            fillcolor="#ADD8E6",
        )

    for source in sources:
        is_highlight = highlight and source == highlight
        dot.node(
            source,
            shorten_text(source, 40),
            shape="ellipse",
            style="filled",
            fillcolor="#FFD700" if is_highlight else "#90EE90",
        )
        if target:
            dot.edge(source, target)

    return dot
