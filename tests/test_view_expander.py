"""Tests for view expansion and operator graph nodes."""

from __future__ import annotations

import unittest
from pathlib import Path

from Classes.schema_registry import SchemaRegistry
from Classes.sql2graph_classes import SQL2GraphBuilder, SQL2GraphParser, SQL2GraphPipeline
from Classes.view_expander import ViewExpander


class TestViewExpander(unittest.TestCase):
    def test_expand_view_into_subquery(self):
        parser = SQL2GraphParser(dialect="postgres")
        if not parser.sqlglot_available:
            self.skipTest("sqlglot not installed")

        registry = SchemaRegistry(dialect="postgres").load_ddl(
            """
            CREATE TABLE sales.orders (order_id INT, amount NUMERIC);
            CREATE VIEW sales.v_orders AS SELECT order_id, amount FROM sales.orders;
            """
        )
        self.assertTrue(registry.is_view("sales", "v_orders"))

        tree, _ = parser._parse_tree("SELECT order_id FROM sales.v_orders v", dialect="postgres")
        expanded = ViewExpander(dialect="postgres").expand(tree, registry)
        sql = expanded.sql(dialect="postgres").lower()
        self.assertIn("orders", sql)
        self.assertNotIn("v_orders", sql)

    def test_simplify_expands_views_before_qualify(self):
        parser = SQL2GraphParser(dialect="postgres")
        if not parser.sqlglot_available:
            self.skipTest("sqlglot not installed")

        registry = SchemaRegistry(dialect="postgres").load_ddl(
            """
            CREATE TABLE public.users (id INT, name TEXT);
            CREATE VIEW public.active_users AS SELECT id, name FROM public.users;
            """
        )
        parser.schema_registry = registry
        simplified = parser.simplify("SELECT * FROM public.active_users u", use_schema=True)
        self.assertTrue(simplified.get("views_expanded"))


class TestOperatorGraphNodes(unittest.TestCase):
    def test_aggregate_and_group_nodes(self):
        extraction = {
            "ctes": [],
            "output_columns": [
                {
                    "alias": "total",
                    "expression": "SUM(amount)",
                    "dependencies": [{"table_alias": "orders", "column": "amount"}],
                    "aggregate": True,
                    "window_function": False,
                }
            ],
            "filters": [],
            "joins": [],
            "group_by_columns": [{"table_alias": "orders", "column": "customer_id"}],
        }
        graph = SQL2GraphBuilder().build(extraction)
        node_types = {attrs.get("node_type") for _, attrs in graph.nodes(data=True)}
        edge_types = {attrs.get("edge_type") for _, _, attrs in graph.edges(data=True)}
        self.assertIn("aggregate", node_types)
        self.assertIn("AGGREGATES_ON", edge_types)
        self.assertIn("VALUE_FLOW", edge_types)

    def test_window_node(self):
        extraction = {
            "ctes": [],
            "output_columns": [
                {
                    "alias": "rn",
                    "expression": "ROW_NUMBER() OVER (PARTITION BY a ORDER BY b)",
                    "dependencies": [
                        {"table_alias": "t", "column": "a"},
                        {"table_alias": "t", "column": "b"},
                    ],
                    "aggregate": False,
                    "window_function": True,
                }
            ],
            "filters": [],
            "joins": [],
            "group_by_columns": [],
        }
        graph = SQL2GraphBuilder().build(extraction)
        node_types = {attrs.get("node_type") for _, attrs in graph.nodes(data=True)}
        self.assertIn("window", node_types)

    def test_union_and_rowset_nodes_on_ddls10(self):
        parser = SQL2GraphParser(dialect="postgres")
        if not parser.sqlglot_available:
            self.skipTest("sqlglot not installed")

        sql = (
            Path(__file__).resolve().parents[1].joinpath("data/DDLs_10.txt").read_text().split(";")[0].strip()
        )
        pipeline = SQL2GraphPipeline(parser=parser)
        result = pipeline.run(sql, dialect="postgres", use_llm_verify=False, use_llm_enhance=False)
        from networkx.readwrite import json_graph

        graph = json_graph.node_link_graph(result["graph"], edges="links")
        node_types = {attrs.get("node_type") for _, attrs in graph.nodes(data=True)}
        edge_types = {attrs.get("edge_type") for _, _, attrs in graph.edges(data=True)}
        self.assertIn("union", node_types)
        self.assertIn("rowset", node_types)
        self.assertIn("ROW_FLOW_IN", edge_types)
        self.assertIn("ROW_FLOW_OUT", edge_types)
        self.assertEqual(result["metadata"]["implementation_profile"], "column_level_v2")

        for _, _, attrs in graph.edges(data=True):
            self.assertIn("confidence", attrs)
            self.assertIn("provenance", attrs)


if __name__ == "__main__":
    unittest.main()
