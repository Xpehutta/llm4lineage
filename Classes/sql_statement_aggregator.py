"""Aggregate lineage across multiple SQL statements (temp tables / rename chains)."""

from __future__ import annotations

from typing import Any

import networkx as nx
from networkx.readwrite import json_graph


class SqlStatementAggregator:
  """Resolve lineage across a batch of INSERT/CTAS statements."""

  def __init__(self):
    self.statements: list[dict[str, Any]] = []
    self.logical_to_physical: dict[str, str] = {}

  def add_statement(self, sql: str, result: dict[str, Any]) -> None:
    simplified = result.get("simplified_query") or {}
    target = (simplified.get("target_table") or "").strip().lower()
    sources: set[str] = set()
    for item in simplified.get("from") or []:
      table = str(item.get("table") or "").strip().lower()
      if table:
        sources.add(self.resolve_table(table))
    for join in simplified.get("joins") or []:
      table = str(join.get("right_table") or "").strip().lower()
      if table:
        sources.add(self.resolve_table(table))

    record = {"sql": sql, "target": target, "sources": sorted(sources)}
    self.statements.append(record)
    if target:
      for source in sources:
        self.logical_to_physical.setdefault(target, source)

  def register_mapping(self, logical: str, physical: str) -> None:
    self.logical_to_physical[logical.strip().lower()] = physical.strip().lower()

  def resolve_table(self, table_name: str) -> str:
    current = table_name.strip().lower()
    seen: set[str] = set()
    while current in self.logical_to_physical and current not in seen:
      seen.add(current)
      current = self.logical_to_physical[current]
    return current

  def merge_graphs(self, graphs: list[dict[str, Any]]) -> dict[str, Any]:
    combined = nx.MultiDiGraph()
    for payload in graphs:
      graph = json_graph.node_link_graph(payload, edges="links")
      combined = nx.compose(combined, graph)
    try:
      return json_graph.node_link_data(combined, edges="links")
    except TypeError:
      return json_graph.node_link_data(combined)
