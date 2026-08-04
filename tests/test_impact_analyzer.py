"""Tests for impact analysis."""

from __future__ import annotations

import unittest

from Classes.impact_analyzer import analyze_impact
from Classes.sql2graph_classes import SQL2GraphBuilder


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


if __name__ == "__main__":
    unittest.main()
