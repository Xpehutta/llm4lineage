"""Render lineage graphs as DOT or Mermaid text."""

from __future__ import annotations

from typing import Any


def graph_to_dot(graph: dict[str, Any], *, highlight: str | None = None) -> str:
    """Emit a Graphviz DOT digraph from node-link JSON."""
    lines = ["digraph lineage {", "  rankdir=LR;"]
    for node in graph.get("nodes") or []:
        node_id = str(node.get("id") or "")
        if not node_id:
            continue
        label = node.get("alias") or node.get("column") or node_id
        fill = "#ADD8E6" if node.get("node_type") == "output_column" else "#90EE90"
        if highlight and node_id == highlight:
            fill = "#FFD700"
        safe_id = _dot_id(node_id)
        lines.append(
            f'  {safe_id} [label="{_escape(str(label))}", shape=box, style=filled, fillcolor="{fill}"];'
        )
    for link in graph.get("links") or graph.get("edges") or []:
        source = link.get("source")
        target = link.get("target")
        if source is None or target is None:
            continue
        edge_type = link.get("edge_type") or ""
        lines.append(
            f'  {_dot_id(str(source))} -> {_dot_id(str(target))} [label="{_escape(str(edge_type))}"];'
        )
    lines.append("}")
    return "\n".join(lines)


def graph_to_mermaid(graph: dict[str, Any]) -> str:
    """Emit a Mermaid flowchart from node-link JSON."""
    lines = ["flowchart LR"]
    for node in graph.get("nodes") or []:
        node_id = str(node.get("id") or "")
        if not node_id:
            continue
        label = node.get("alias") or node.get("column") or node_id
        lines.append(f"  {_mermaid_id(node_id)}[\"{_escape(str(label))}\"]")
    for link in graph.get("links") or graph.get("edges") or []:
        source = link.get("source")
        target = link.get("target")
        if source is None or target is None:
            continue
        edge_type = link.get("edge_type") or ""
        lines.append(
            f"  {_mermaid_id(str(source))} -->|{_escape(str(edge_type))}| {_mermaid_id(str(target))}"
        )
    return "\n".join(lines)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _dot_id(value: str) -> str:
    return '"' + _escape(value) + '"'


def _mermaid_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value)
    return cleaned or "node"
