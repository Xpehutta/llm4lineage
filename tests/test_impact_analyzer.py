"""Tests for impact analysis."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from networkx.readwrite import json_graph

from Classes.impact_analyzer import (
    analyze_impact,
    graph_from_payload,
    table_level_impact,
)
from Classes.sql2graph_classes import SQL2GraphBuilder


def payload(nodes: list[dict], links: list[dict]) -> dict:
    """Build a node-link payload with explicit control over node/edge attributes."""
    return {
        "directed": True,
        "multigraph": True,
        "graph": {},
        "nodes": nodes,
        "links": [{"key": index, **link} for index, link in enumerate(links)],
    }


#: output.a -> output.b -> output.c, plus a non-lineage JOINS_ON edge off output.a.
CHAIN = payload(
    nodes=[
        {"id": "orders.amount", "node_type": "source_column", "physical_table": "raw.orders",
         "column": "amount"},
        {"id": "output.a", "node_type": "output_column", "alias": "a",
         "physical_table": "raw.orders"},
        {"id": "output.b", "node_type": "output_column", "alias": "b"},
        {"id": "output.c", "node_type": "output_column", "alias": "c"},
        {"id": "sidecar.flag", "node_type": "source_column"},
    ],
    links=[
        {"source": "orders.amount", "target": "output.a", "edge_type": "DERIVED_FROM"},
        {"source": "output.a", "target": "output.b", "edge_type": "DERIVED_FROM"},
        {"source": "output.b", "target": "output.c", "edge_type": "DERIVED_FROM"},
        {"source": "output.a", "target": "sidecar.flag", "edge_type": "JOINS_ON"},
    ],
)


class TestImpactAnalyzer(unittest.TestCase):
    def test_transitive_downstream_chain(self):
        extraction = {
            "ctes": [],
            "output_columns": [
                {
                    "alias": "y",
                    "expression": "x",
                    "dependencies": [{"table_alias": "t", "column": "x"}],
                    "aggregate": False,
                    "window_function": False,
                },
                {
                    "alias": "z",
                    "expression": "y",
                    "dependencies": [{"table_alias": "output", "column": "y"}],
                    "aggregate": False,
                    "window_function": False,
                },
            ],
            "filters": [],
            "joins": [],
            "group_by_columns": [],
        }
        builder = SQL2GraphBuilder()
        graph_json = builder.to_node_link()
        # manually add chain output.y -> output.z if not present
        builder.build(extraction)
        graph_json = builder.to_node_link()
        builder.graph.add_edge("t.x", "output.y", edge_type="DERIVED_FROM", confidence=1.0, provenance="deterministic")
        builder.graph.add_edge("output.y", "output.z", edge_type="DERIVED_FROM", confidence=1.0, provenance="deterministic")
        graph_json = builder.to_node_link()

        report = analyze_impact(graph_json, "output.y", direction="downstream")
        downstream_nodes = {item["node"] for item in report["downstream"]}
        self.assertIn("output.z", downstream_nodes)


class TestAnalyzeImpactDirections(unittest.TestCase):
    def test_upstream_walk_returns_transitive_ancestors(self):
        report = analyze_impact(CHAIN, "output.c", direction="up")

        self.assertEqual(
            [hit["node"] for hit in report["upstream"]],
            ["output.b", "output.a", "orders.amount"],
        )
        self.assertEqual(report["downstream"], [])

    def test_upstream_hits_carry_node_type_and_reason(self):
        report = analyze_impact(CHAIN, "output.c", direction="upstream")

        first = report["upstream"][0]
        self.assertEqual(first["node_type"], "output_column")
        self.assertEqual(first["reason"], "column_derivation")
        self.assertEqual(
            first["path"],
            [
                {
                    "from": "output.b",
                    "to": "output.c",
                    "edge_type": "DERIVED_FROM",
                    "reason": "column_derivation",
                }
            ],
        )

    def test_downstream_walk_accumulates_the_full_path(self):
        report = analyze_impact(CHAIN, "orders.amount", direction="down")

        hits = {hit["node"]: hit for hit in report["downstream"]}
        self.assertEqual(set(hits), {"output.a", "output.b", "output.c"})
        self.assertEqual(len(hits["output.c"]["path"]), 3)

    def test_both_directions_populates_upstream_and_downstream(self):
        report = analyze_impact(CHAIN, "output.b")

        self.assertEqual([hit["node"] for hit in report["upstream"]], ["output.a", "orders.amount"])
        self.assertEqual([hit["node"] for hit in report["downstream"]], ["output.c"])

    def test_non_lineage_edges_are_not_traversed(self):
        """A JOINS_ON edge is structural, not a value flow, so it is not impact."""
        report = analyze_impact(CHAIN, "output.a", direction="down")

        self.assertNotIn("sidecar.flag", [hit["node"] for hit in report["downstream"]])

    def test_unknown_target_yields_an_empty_report(self):
        self.assertEqual(
            analyze_impact(CHAIN, "output.missing"),
            {"target": "output.missing", "upstream": [], "downstream": []},
        )

    def test_a_cycle_does_not_loop_forever(self):
        cyclic = payload(
            nodes=[{"id": "a"}, {"id": "b"}],
            links=[
                {"source": "a", "target": "b", "edge_type": "DERIVED_FROM"},
                {"source": "b", "target": "a", "edge_type": "DERIVED_FROM"},
            ],
        )
        report = analyze_impact(cyclic, "a", direction="down")
        self.assertEqual([hit["node"] for hit in report["downstream"]], ["b"])


class TestGraphFromPayload(unittest.TestCase):
    def test_falls_back_when_the_edges_keyword_is_unsupported(self):
        """Older networkx releases reject ``edges=``; the shim retries without it."""
        real = json_graph.node_link_graph
        calls = []

        def old_networkx(data, **kwargs):
            calls.append(kwargs)
            if "edges" in kwargs:
                raise TypeError("node_link_graph() got an unexpected keyword argument 'edges'")
            return real(data, edges="links")

        with patch("Classes.impact_analyzer.json_graph.node_link_graph", side_effect=old_networkx):
            graph = graph_from_payload(CHAIN)

        self.assertEqual(calls, [{"edges": "links"}, {}])
        self.assertIn("output.a", graph)


class TestTableLevelImpact(unittest.TestCase):
    def test_downstream_table_prefixes_are_collected_for_a_matching_table(self):
        report = table_level_impact(CHAIN, "RAW.ORDERS")

        self.assertEqual(report["table"], "raw.orders")
        self.assertEqual(report["downstream_tables"], ["output"])

    def test_output_columns_are_matched_on_their_node_name_too(self):
        report = table_level_impact(CHAIN, "output.b")

        self.assertEqual(report["downstream_tables"], ["output"])

    def test_unknown_table_has_no_downstream(self):
        self.assertEqual(
            table_level_impact(CHAIN, "raw.nonexistent"),
            {"table": "raw.nonexistent", "downstream_tables": []},
        )


if __name__ == "__main__":
    unittest.main()
