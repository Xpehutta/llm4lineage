"""Build networkx MultiDiGraph from SQL2Graph extraction."""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph

from Classes.sql2graph.models import (
    ColumnRef,
    FilterSpec,
    JoinSpec,
    OutputColumn,
    SQL2GraphExtraction,
    SQL2GraphExtractionCTE,
)

logger = logging.getLogger(__name__)

class SQL2GraphBuilder:
    """Build a column-level lineage graph from structured extraction JSON."""

    OPERATOR_NODE_TYPES = frozenset(
        {"union", "aggregate", "window", "transformation", "rowset"}
    )
    COLUMN_NODE_TYPES = frozenset({"source_column", "output_column", "filter"})
    ALL_NODE_TYPES = COLUMN_NODE_TYPES | OPERATOR_NODE_TYPES | frozenset({"join"})

    OPERATOR_EDGE_TYPES = frozenset(
        {
            "ROW_FLOW_IN",
            "ROW_FLOW_OUT",
            "VALUE_FLOW",
            "AGGREGATES_ON",
            "WINDOW_OVER",
        }
    )
    COLUMN_EDGE_TYPES = frozenset(
        {"DERIVED_FROM", "FILTERED_BY", "USES_COLUMN", "JOINS_ON", "GROUPED_BY"}
    )
    ALL_EDGE_TYPES = COLUMN_EDGE_TYPES | OPERATOR_EDGE_TYPES

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    @staticmethod
    def _default_edge_attrs(**extra: Any) -> dict[str, Any]:
        # `verified` is True here because a deterministic edge was read straight
        # out of the parsed SQL. LLM-derived edges flip it back to False until a
        # Reviewer confirms them against the source.
        attrs = {"confidence": 1.0, "provenance": "deterministic", "verified": True}
        attrs.update(extra)
        return attrs

    def _add_edge(self, source: str, target: str, edge_type: str, **attrs: Any) -> None:
        self.graph.add_edge(source, target, **self._default_edge_attrs(edge_type=edge_type, **attrs))

    @staticmethod
    def _parse_aggregate_function(expression: str) -> str | None:
        match = re.search(r"\b(SUM|COUNT|AVG|MIN|MAX)\s*\(", expression or "", re.IGNORECASE)
        return match.group(1).upper() if match else None

    @staticmethod
    def _looks_transformation(expression: str) -> bool:
        return bool(re.search(r"\b(CASE|CAST|COALESCE|::)\b", expression or "", re.IGNORECASE))

    def _digest_id(self, prefix: str, seed: str) -> str:
        digest = hashlib.md5(seed.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
        return f"{prefix}_{digest}"

    def _add_source_column(self, ref: ColumnRef) -> str:
        node_id = ref.node_id()
        self.graph.add_node(
            node_id,
            node_type="source_column",
            table_alias=ref.table_alias or "unknown",
            column=ref.column,
        )
        return node_id

    def _add_filter_node(self, clause: str, condition: str) -> str:
        digest = hashlib.md5(
            f"{clause}:{condition}".encode(), usedforsecurity=False
        ).hexdigest()[:12]
        node_id = f"filter_{digest}"
        self.graph.add_node(node_id, node_type="filter", clause=clause, condition=condition)
        return node_id

    def _add_scope(
        self,
        scope: dict[str, Any],
        output_prefix: str,
        output_node_type: str,
    ) -> list[str]:
        output_nodes = []

        for output in scope.get("output_columns", []):
            output_obj = OutputColumn.model_validate(output)
            out_id = f"{output_prefix}.{output_obj.alias}"
            self.graph.add_node(
                out_id,
                node_type=output_node_type,
                alias=output_obj.alias,
                expression=output_obj.expression,
                aggregate=output_obj.aggregate,
                window_function=output_obj.window_function,
            )
            output_nodes.append(out_id)

            for dep in output_obj.dependencies:
                dep_node = self._add_source_column(dep)
                self._add_edge(dep_node, out_id, "DERIVED_FROM")

            self._add_operator_nodes(output_obj, out_id, scope)

            if output_obj.aggregate:
                for group_ref in scope.get("group_by_columns", []):
                    grp = ColumnRef.model_validate(group_ref)
                    grp_node = self._add_source_column(grp)
                    self._add_edge(grp_node, out_id, "GROUPED_BY")

        for filt in scope.get("filters", []):
            f = FilterSpec.model_validate(filt)
            filter_node = self._add_filter_node(f.clause, f.condition)
            for used in f.columns_used:
                col_node = self._add_source_column(used)
                self._add_edge(col_node, filter_node, "USES_COLUMN")
            for out in output_nodes:
                self._add_edge(filter_node, out, "FILTERED_BY")

        for join in scope.get("joins", []):
            j = JoinSpec.model_validate(join)
            left = self._add_source_column(j.join_columns[0])
            right = self._add_source_column(j.join_columns[1])
            self._add_edge(left, right, "JOINS_ON", join_type=j.type, condition=j.condition)

        for cte in scope.get("ctes", []):
            cte_obj = SQL2GraphExtractionCTE.model_validate(cte)
            rowset_id = f"rowset.{cte_obj.alias}"
            self.graph.add_node(
                rowset_id,
                node_type="rowset",
                cte_alias=cte_obj.alias,
            )
            cte_outputs = self._add_scope(
                cte_obj.model_dump(),
                output_prefix=cte_obj.alias,
                output_node_type="source_column",
            )
            for out_node in cte_outputs:
                self._add_edge(out_node, rowset_id, "ROW_FLOW_OUT")

        return output_nodes

    def _add_operator_nodes(
        self,
        output_obj: OutputColumn,
        out_id: str,
        scope: dict[str, Any],
    ) -> None:
        union_branches = output_obj.union_branches or []
        if len(union_branches) > 1:
            union_id = self._digest_id("union", out_id)
            union_type = "ALL"
            self.graph.add_node(
                union_id,
                node_type="union",
                union_type=union_type,
                branch_count=len(union_branches),
            )
            for branch in union_branches:
                branch_index = branch.get("branch_index")
                if branch.get("kind") == "literal":
                    literal_id = self._digest_id("literal", f"{out_id}:{branch_index}:{branch.get('literal_value')}")
                    self.graph.add_node(
                        literal_id,
                        node_type="transformation",
                        function="LITERAL",
                        expression_text=branch.get("literal_value") or "",
                    )
                    self._add_edge(literal_id, union_id, "ROW_FLOW_IN", branch_index=branch_index)
                elif branch.get("kind") == "column_ref":
                    ref = ColumnRef(
                        table_alias=branch.get("table_alias"),
                        column=branch.get("column") or "",
                        physical_table=branch.get("physical_table"),
                    )
                    branch_node = self._add_source_column(ref)
                    self._add_edge(branch_node, union_id, "ROW_FLOW_IN", branch_index=branch_index)
            self._add_edge(union_id, out_id, "ROW_FLOW_OUT")

        if output_obj.aggregate:
            agg_id = self._digest_id("agg", out_id)
            self.graph.add_node(
                agg_id,
                node_type="aggregate",
                function=self._parse_aggregate_function(output_obj.expression),
                expression_text=output_obj.expression,
            )
            for dep in output_obj.dependencies:
                dep_node = self._add_source_column(dep)
                self._add_edge(dep_node, agg_id, "AGGREGATES_ON")
            self._add_edge(agg_id, out_id, "VALUE_FLOW")

        if output_obj.window_function:
            window_id = self._digest_id("window", out_id)
            self.graph.add_node(
                window_id,
                node_type="window",
                expression_text=output_obj.expression,
            )
            for dep in output_obj.dependencies:
                dep_node = self._add_source_column(dep)
                self._add_edge(dep_node, window_id, "WINDOW_OVER")
            self._add_edge(window_id, out_id, "VALUE_FLOW")

        if (
            not output_obj.aggregate
            and not output_obj.window_function
            and len(union_branches) <= 1
            and output_obj.dependencies
            and self._looks_transformation(output_obj.expression)
        ):
            transform_id = self._digest_id("transform", out_id)
            self.graph.add_node(
                transform_id,
                node_type="transformation",
                function="EXPR",
                expression_text=output_obj.expression,
            )
            for dep in output_obj.dependencies:
                dep_node = self._add_source_column(dep)
                self._add_edge(dep_node, transform_id, "VALUE_FLOW")
            self._add_edge(transform_id, out_id, "VALUE_FLOW")

    def build(self, extraction: dict[str, Any]) -> nx.MultiDiGraph:
        validated = SQL2GraphExtraction.model_validate(extraction)
        self.graph = nx.MultiDiGraph()
        self._add_scope(validated.model_dump(), output_prefix="output", output_node_type="output_column")
        return self.graph

    def apply_edge_provenance(
        self, provenance: str, confidence: float, verified: bool = False
    ) -> int:
        """Stamp every edge with where it came from and how much to trust it.

        ``verified`` defaults to False: an edge only earns True once a Reviewer
        has confirmed it against the source code.
        """
        updated = 0
        for source, target, key in self.graph.edges(keys=True):
            data = self.graph.edges[source, target, key]
            data["provenance"] = provenance
            data["confidence"] = confidence
            data["verified"] = verified
            updated += 1
        return updated

    def link_cte_aliases(self, alias_map: dict[str, str]) -> int:
        """
        Connect CTE output nodes to alias-qualified references of the same column.

        When the main query aliases a CTE (e.g. ``JOIN recent_orders r``), the LLM
        dependencies reference ``r.total`` while the CTE scope produces
        ``recent_orders.total``. This links them so the lineage chain stays connected
        (spec section 7.9 / 8.3).
        """
        if not alias_map:
            return 0
        added = 0
        for node, attrs in list(self.graph.nodes(data=True)):
            if attrs.get("node_type") != "source_column":
                continue
            cte_name = alias_map.get(attrs.get("table_alias"))
            if not cte_name:
                continue
            cte_node = f"{cte_name}.{attrs.get('column')}"
            if cte_node != node and cte_node in self.graph.nodes:
                if not self.graph.has_edge(cte_node, node):
                    self._add_edge(cte_node, node, "DERIVED_FROM")
                    added += 1
                rowset_id = f"rowset.{cte_name}"
                if rowset_id in self.graph.nodes:
                    self._add_edge(rowset_id, node, "ROW_FLOW_IN")
        return added

    def materialize_transitive_derived_from(self, output_node_type: str = "output_column") -> int:
        """
        Add direct DERIVED_FROM edges from ultimate source columns to outputs.

        Implements spec section 7.10: when lineage passes through intermediate
        column nodes (e.g. CTE passthrough), materialize shortcut edges so
        downstream consumers can query source-to-target without walking the chain.
        """
        added = 0
        for target, attrs in list(self.graph.nodes(data=True)):
            if attrs.get("node_type") != output_node_type:
                continue

            stack = [target]
            visited = set()
            leaves: set = set()
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)

                predecessors = [
                    source
                    for source, _, edge_data in self.graph.in_edges(node, data=True)
                    if edge_data.get("edge_type") == "DERIVED_FROM"
                ]
                if not predecessors:
                    if node != target:
                        leaves.add(node)
                    continue
                stack.extend(predecessors)

            for source in leaves:
                if source == target:
                    continue
                has_direct = any(
                    tgt == target and edge_data.get("edge_type") == "DERIVED_FROM"
                    for _, tgt, edge_data in self.graph.out_edges(source, data=True)
                )
                if not has_direct:
                    self._add_edge(source, target, "DERIVED_FROM", transitive=True)
                    added += 1
        return added

    def ensure_acyclic(self) -> list[str]:
        """
        Break any remaining directed cycles so the graph is a DAG.

        Prefers removing transitive shortcut edges, then other edges participating
        in the first detected cycle.
        """
        warnings: list[str] = []
        while self.graph.number_of_edges() > 0 and not nx.is_directed_acyclic_graph(self.graph):
            try:
                cycle = nx.find_cycle(self.graph)
            except nx.NetworkXNoCycle:
                break

            removable = None
            for u, v, key in cycle:
                edge_data = self.graph.edges[u, v, key]
                if edge_data.get("transitive"):
                    removable = (u, v, key)
                    break
            if removable is None:
                removable = cycle[0]

            u, v, key = removable
            edge_type = self.graph.edges[u, v, key].get("edge_type", "")
            self.graph.remove_edge(u, v, key)
            warnings.append(f"Removed cyclic edge: {u} -> {v} ({edge_type})")

        return warnings

    def to_node_link(self) -> dict[str, Any]:
        # Keep "links" key for backward compatibility in notebook/UI code.
        try:
            return json_graph.node_link_data(self.graph, edges="links")
        except TypeError:
            return json_graph.node_link_data(self.graph)

    def to_dot(self) -> str:
        lines = ["digraph SQL2Graph {"]
        for node, attrs in self.graph.nodes(data=True):
            label = attrs.get("alias") or node
            lines.append(f'  "{node}" [label="{label}\\n({attrs.get("node_type", "node")})"];')
        for source, target, attrs in self.graph.edges(data=True):
            lines.append(f'  "{source}" -> "{target}" [label="{attrs.get("edge_type", "")}"];')
        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        lines = ["flowchart TD"]
        for source, target, attrs in self.graph.edges(data=True):
            edge_label = attrs.get("edge_type", "")
            lines.append(f'  "{source}" -->|{edge_label}| "{target}"')
        return "\n".join(lines)
