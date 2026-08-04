"""Impact / downstream analysis over SQL2Graph node-link JSON."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set

import networkx as nx
from networkx.readwrite import json_graph

LINEAGE_EDGE_TYPES = frozenset(
    {
        "DERIVED_FROM",
        "GROUPED_BY",
        "FILTERED_BY",
        "ROW_FLOW_IN",
        "ROW_FLOW_OUT",
        "VALUE_FLOW",
        "AGGREGATES_ON",
        "WINDOW_OVER",
    }
)

EDGE_REASONS = {
    "DERIVED_FROM": "column_derivation",
    "FILTERED_BY": "filter",
    "JOINS_ON": "join",
    "GROUPED_BY": "aggregation_group_key",
    "AGGREGATES_ON": "aggregation_input",
    "WINDOW_OVER": "window_input",
    "ROW_FLOW_IN": "union_or_row_flow",
    "ROW_FLOW_OUT": "union_or_row_flow",
    "VALUE_FLOW": "expression_transform",
}


def graph_from_payload(graph_json: Dict[str, Any]) -> nx.MultiDiGraph:
  try:
    return json_graph.node_link_graph(graph_json, edges="links")
  except TypeError:
    return json_graph.node_link_graph(graph_json)


def _walk(
    graph: nx.MultiDiGraph,
    start: str,
    *,
    direction: str,
    edge_types: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
  allowed = edge_types or LINEAGE_EDGE_TYPES
  visited: Set[str] = set()
  queue = deque([(start, [])])
  hits: List[Dict[str, Any]] = []

  while queue:
    node, path = queue.popleft()
    if node in visited:
      continue
    visited.add(node)
    if node != start:
      attrs = graph.nodes.get(node, {})
      hits.append(
        {
          "node": node,
          "node_type": attrs.get("node_type"),
          "path": path,
          "reason": path[-1]["reason"] if path else None,
        }
      )

    if direction == "upstream":
      for pred in graph.predecessors(node):
        edge_data = graph.get_edge_data(pred, node) or {}
        for data in edge_data.values():
          edge_type = data.get("edge_type", "")
          if edge_type not in allowed:
            continue
          step = {
            "from": pred,
            "to": node,
            "edge_type": edge_type,
            "reason": EDGE_REASONS.get(edge_type, edge_type),
          }
          queue.append((pred, path + [step]))
          break
    else:
      for succ in graph.successors(node):
        edge_data = graph.get_edge_data(node, succ) or {}
        for data in edge_data.values():
          edge_type = data.get("edge_type", "")
          if edge_type not in allowed:
            continue
          step = {
            "from": node,
            "to": succ,
            "edge_type": edge_type,
            "reason": EDGE_REASONS.get(edge_type, edge_type),
          }
          queue.append((succ, path + [step]))
          break

  return hits


def analyze_impact(
    graph_json: Dict[str, Any],
    target_node: str,
    *,
    direction: str = "both",
) -> Dict[str, Any]:
  graph = graph_from_payload(graph_json)
  result: Dict[str, Any] = {"target": target_node, "upstream": [], "downstream": []}
  if target_node not in graph:
    return result
  if direction in {"up", "upstream", "both"}:
    result["upstream"] = _walk(graph, target_node, direction="upstream")
  if direction in {"down", "downstream", "both"}:
    result["downstream"] = _walk(graph, target_node, direction="downstream")
  return result


def table_level_impact(graph_json: Dict[str, Any], table_name: str) -> Dict[str, List[str]]:
  graph = graph_from_payload(graph_json)
  table_name = table_name.lower()
  impacted_tables: Set[str] = set()
  for node, attrs in graph.nodes(data=True):
    if attrs.get("node_type") != "output_column":
      continue
    physical = str(attrs.get("physical_table") or attrs.get("table_alias") or "").lower()
    if table_name in physical or table_name in node.lower():
      for hit in _walk(graph, node, direction="downstream"):
        impacted_tables.add(hit["node"].split(".")[0])
  return {"table": table_name, "downstream_tables": sorted(impacted_tables)}
