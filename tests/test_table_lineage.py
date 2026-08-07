"""Tests for deterministic table-level lineage extraction."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from Classes.table_lineage import extract_table_lineage


class TestStatementTypes(unittest.TestCase):
    def test_insert_with_a_column_list_unwraps_the_target(self):
        result = extract_table_lineage(
            "INSERT INTO mart.sales (a, b) SELECT o.a, o.b FROM raw.orders o"
        )
        self.assertEqual(result["statement_type"], "insert")
        self.assertEqual(result["target"], "mart.sales")
        self.assertEqual(result["sources"], ["raw.orders"])
        self.assertTrue(result["parser_used"])

    def test_insert_from_values_has_no_sources(self):
        result = extract_table_lineage("INSERT INTO mart.sales VALUES (1, 2)")
        self.assertEqual(result["target"], "mart.sales")
        self.assertEqual(result["sources"], [])

    def test_create_table_as_select(self):
        result = extract_table_lineage("CREATE TABLE mart.sales AS SELECT * FROM raw.orders")
        self.assertEqual(result["statement_type"], "create_table_as")
        self.assertEqual(result["target"], "mart.sales")
        self.assertEqual(result["sources"], ["raw.orders"])

    def test_plain_create_table_has_no_sources(self):
        result = extract_table_lineage("CREATE TABLE mart.sales (a int, b text)")
        self.assertEqual(result["statement_type"], "create_table")
        self.assertEqual(result["target"], "mart.sales")
        self.assertEqual(result["sources"], [])

    def test_create_view_as_select(self):
        result = extract_table_lineage(
            "CREATE OR REPLACE VIEW analytics.v_sales AS "
            "SELECT p.category, s.amount FROM products p JOIN sales s ON p.id = s.product_id"
        )
        self.assertEqual(result["statement_type"], "create_view")
        self.assertEqual(result["target"], "analytics.v_sales")
        self.assertEqual(result["sources"], ["products", "sales"])

    def test_create_materialized_view(self):
        result = extract_table_lineage(
            "CREATE MATERIALIZED VIEW analytics.mv_sales AS "
            "SELECT category, SUM(amount) AS total FROM sales GROUP BY category"
        )
        self.assertEqual(result["statement_type"], "create_materialized_view")
        self.assertEqual(result["target"], "analytics.mv_sales")
        self.assertEqual(result["sources"], ["sales"])

    def test_create_view_target_is_not_listed_as_source(self):
        result = extract_table_lineage(
            "CREATE VIEW analytics.v_users AS SELECT id, name FROM analytics.v_users_src"
        )
        self.assertEqual(result["target"], "analytics.v_users")
        self.assertNotIn("analytics.v_users", result["sources"])
        self.assertEqual(result["sources"], ["analytics.v_users_src"])

    def test_update_target_is_excluded_from_its_own_sources(self):
        result = extract_table_lineage(
            "UPDATE mart.sales SET x = 1 FROM raw.orders WHERE mart.sales.id = raw.orders.id"
        )
        self.assertEqual(result["statement_type"], "update")
        self.assertEqual(result["target"], "mart.sales")
        self.assertEqual(result["sources"], ["raw.orders"])

    def test_merge_target_is_excluded_from_its_own_sources(self):
        result = extract_table_lineage(
            "MERGE INTO mart.sales t USING raw.orders s ON t.id = s.id "
            "WHEN MATCHED THEN UPDATE SET x = 1"
        )
        self.assertEqual(result["statement_type"], "merge")
        self.assertEqual(result["target"], "mart.sales")
        self.assertEqual(result["sources"], ["raw.orders"])

    def test_bare_select_has_no_target(self):
        result = extract_table_lineage("SELECT * FROM raw.orders")
        self.assertEqual(result["statement_type"], "select")
        self.assertEqual(result["target"], "")
        self.assertEqual(result["sources"], ["raw.orders"])

    def test_three_part_names_are_kept_whole(self):
        result = extract_table_lineage("SELECT * FROM warehouse.raw.orders")
        self.assertEqual(result["sources"], ["warehouse.raw.orders"])

    def test_dialect_is_honoured(self):
        result = extract_table_lineage("SELECT * FROM `raw`.`orders`", dialect="spark")
        self.assertEqual(result["sources"], ["raw.orders"])


class TestCteHandling(unittest.TestCase):
    """CTE aliases are not physical tables and must never appear as sources."""

    def test_cte_in_a_select_is_excluded(self):
        result = extract_table_lineage(
            "WITH c AS (SELECT * FROM raw.orders) "
            "SELECT * FROM c JOIN raw.items i ON c.id = i.id"
        )
        self.assertEqual(result["sources"], ["raw.items", "raw.orders"])

    def test_cte_declared_before_the_insert_is_excluded(self):
        result = extract_table_lineage(
            "WITH c AS (SELECT * FROM raw.orders) INSERT INTO mart.sales SELECT * FROM c"
        )
        self.assertEqual(result["target"], "mart.sales")
        self.assertEqual(result["sources"], ["raw.orders"])

    def test_cte_declared_inside_the_insert_is_excluded(self):
        result = extract_table_lineage(
            "INSERT INTO mart.sales WITH c AS (SELECT * FROM raw.orders) SELECT * FROM c"
        )
        self.assertEqual(result["sources"], ["raw.orders"])

    def test_cte_before_a_ctas_is_excluded(self):
        result = extract_table_lineage(
            "WITH c AS (SELECT * FROM raw.orders) CREATE TABLE mart.sales AS SELECT * FROM c"
        )
        self.assertEqual(result["sources"], ["raw.orders"])

    def test_insert_reading_from_its_own_target_keeps_it_as_a_source(self):
        result = extract_table_lineage("INSERT INTO mart.sales SELECT * FROM mart.sales")
        self.assertEqual(result["target"], "mart.sales")
        self.assertEqual(result["sources"], ["mart.sales"])

    def test_derived_table_aliases_are_not_reported(self):
        result = extract_table_lineage("SELECT * FROM (SELECT 1 AS x) sub")
        self.assertEqual(result["sources"], [])


class TestDegradedModes(unittest.TestCase):
    def test_unparseable_sql_reports_the_error_and_flags_the_parser_as_unused(self):
        result = extract_table_lineage("NOT SQL AT ALL ((")
        self.assertFalse(result["parser_used"])
        self.assertEqual(result["statement_type"], "unknown")
        self.assertEqual(result["sources"], [])
        self.assertIn("error", result)

    def test_without_sqlglot_the_extractor_degrades_quietly(self):
        with patch("Classes.table_lineage.sqlglot", None):
            result = extract_table_lineage("SELECT * FROM raw.orders")

        self.assertEqual(
            result,
            {"target": "", "sources": [], "statement_type": "unknown", "parser_used": False},
        )


if __name__ == "__main__":
    unittest.main()
