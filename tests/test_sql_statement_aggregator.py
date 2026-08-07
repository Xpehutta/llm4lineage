"""Tests for SqlStatementAggregator (cross-statement lineage resolution)."""

from __future__ import annotations

import unittest

from Classes.sql_statement_aggregator import SqlStatementAggregator


def _simplified(target: str, from_tables=(), join_tables=()):
    return {
        "simplified_query": {
            "target_table": target,
            "from": [{"table": name} for name in from_tables],
            "joins": [{"right_table": name} for name in join_tables],
        }
    }


class TestResolveTable(unittest.TestCase):
    """`resolve_table` follows logical->physical mappings to a fixed point."""

    def test_unmapped_table_is_returned_normalised(self):
        aggregator = SqlStatementAggregator()
        self.assertEqual(aggregator.resolve_table("  Public.Orders  "), "public.orders")

    def test_single_mapping_is_followed(self):
        aggregator = SqlStatementAggregator()
        aggregator.register_mapping("tmp_orders", "public.orders")
        self.assertEqual(aggregator.resolve_table("tmp_orders"), "public.orders")

    def test_rename_chain_is_followed_to_the_physical_table(self):
        aggregator = SqlStatementAggregator()
        aggregator.register_mapping("stage_a", "stage_b")
        aggregator.register_mapping("stage_b", "public.orders")
        self.assertEqual(aggregator.resolve_table("stage_a"), "public.orders")

    def test_register_mapping_normalises_case_and_whitespace(self):
        aggregator = SqlStatementAggregator()
        aggregator.register_mapping("  TMP_A  ", "  Public.Orders ")
        self.assertEqual(aggregator.logical_to_physical["tmp_a"], "public.orders")

    def test_cyclic_mapping_terminates(self):
        aggregator = SqlStatementAggregator()
        aggregator.register_mapping("a", "b")
        aggregator.register_mapping("b", "a")
        self.assertIn(aggregator.resolve_table("a"), {"a", "b"})


class TestAddStatement(unittest.TestCase):
    def test_records_target_and_sorted_sources_from_from_and_joins(self):
        aggregator = SqlStatementAggregator()
        aggregator.add_statement(
            "INSERT INTO Mart.Sales SELECT * FROM Raw.Orders JOIN Raw.Customers ON 1=1",
            _simplified("Mart.Sales", from_tables=["Raw.Orders"], join_tables=["Raw.Customers"]),
        )

        record = aggregator.statements[0]
        self.assertEqual(record["target"], "mart.sales")
        self.assertEqual(record["sources"], ["raw.customers", "raw.orders"])

    def test_blank_table_names_are_ignored(self):
        aggregator = SqlStatementAggregator()
        aggregator.add_statement(
            "SELECT 1",
            _simplified("mart.sales", from_tables=["", "  "], join_tables=[None]),
        )
        self.assertEqual(aggregator.statements[0]["sources"], [])

    def test_sources_are_resolved_through_known_mappings(self):
        aggregator = SqlStatementAggregator()
        aggregator.register_mapping("tmp_orders", "raw.orders")
        aggregator.add_statement(
            "INSERT INTO mart.sales SELECT * FROM tmp_orders",
            _simplified("mart.sales", from_tables=["tmp_orders"]),
        )
        self.assertEqual(aggregator.statements[0]["sources"], ["raw.orders"])

    def test_first_source_becomes_the_mapping_for_the_target(self):
        aggregator = SqlStatementAggregator()
        aggregator.add_statement(
            "CREATE TEMP TABLE tmp_a AS SELECT * FROM raw.orders",
            _simplified("tmp_a", from_tables=["raw.orders"]),
        )
        aggregator.add_statement(
            "INSERT INTO mart.sales SELECT * FROM tmp_a",
            _simplified("mart.sales", from_tables=["tmp_a"]),
        )
        self.assertEqual(aggregator.logical_to_physical["tmp_a"], "raw.orders")
        self.assertEqual(aggregator.statements[1]["sources"], ["raw.orders"])

    def test_existing_mapping_is_not_overwritten_by_a_later_statement(self):
        aggregator = SqlStatementAggregator()
        aggregator.register_mapping("tmp_a", "raw.orders")
        aggregator.add_statement(
            "INSERT INTO tmp_a SELECT * FROM raw.returns",
            _simplified("tmp_a", from_tables=["raw.returns"]),
        )
        self.assertEqual(aggregator.logical_to_physical["tmp_a"], "raw.orders")

    def test_statement_without_simplified_query_records_empty_target(self):
        aggregator = SqlStatementAggregator()
        aggregator.add_statement("SELECT 1", {})
        self.assertEqual(aggregator.statements, [{"sql": "SELECT 1", "target": "", "sources": []}])
        self.assertEqual(aggregator.logical_to_physical, {})


class TestMergeGraphs(unittest.TestCase):
    def test_nodes_and_edges_from_all_payloads_are_combined(self):
        first = {
            "directed": True,
            "multigraph": True,
            "graph": {},
            "nodes": [{"id": "a"}, {"id": "b"}],
            "links": [{"source": "a", "target": "b", "edge_type": "DERIVED_FROM", "key": 0}],
        }
        second = {
            "directed": True,
            "multigraph": True,
            "graph": {},
            "nodes": [{"id": "b"}, {"id": "c"}],
            "links": [{"source": "b", "target": "c", "edge_type": "DERIVED_FROM", "key": 0}],
        }

        merged = SqlStatementAggregator().merge_graphs([first, second])

        self.assertEqual({node["id"] for node in merged["nodes"]}, {"a", "b", "c"})
        edges = {(link["source"], link["target"]) for link in merged["links"]}
        self.assertEqual(edges, {("a", "b"), ("b", "c")})

    def test_merging_no_graphs_yields_an_empty_graph(self):
        merged = SqlStatementAggregator().merge_graphs([])
        self.assertEqual(merged["nodes"], [])
        self.assertEqual(merged["links"], [])


if __name__ == "__main__":
    unittest.main()
