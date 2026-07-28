"""Tests for ColumnLineageExtractor."""

import unittest

from Classes.pipeline.core.lineage import ColumnLineageExtractor
from Classes.pipeline.core.parser import SQLParser
from Classes.pipeline.exceptions import LineageExtractionError

LINEAGE_KEYS = {"target_column", "source_columns", "expression", "used_tables"}


class TestColumnLineageExtractor(unittest.TestCase):
    def setUp(self):
        self.parser = SQLParser()
        self.extractor = ColumnLineageExtractor()

    def _extract(self, sql: str):
        tree = self.parser.parse(sql)
        return self.extractor.extract(tree)

    def test_simple_select(self):
        lineage = self._extract("SELECT a, b FROM t")
        self.assertEqual(len(lineage), 2)
        self.assertEqual(lineage[0]["target_column"], "a")
        self.assertEqual(lineage[0]["source_columns"][0]["column"], "a")

    def test_join(self):
        sql = (
            "SELECT u.name, o.total FROM users u "
            "JOIN orders o ON u.id = o.user_id"
        )
        lineage = self._extract(sql)
        self.assertEqual(len(lineage), 2)
        tables = {c["table"] for entry in lineage for c in entry["source_columns"]}
        self.assertIn("u", tables)
        self.assertIn("o", tables)

    def test_aliased_expression(self):
        lineage = self._extract("SELECT x AS y FROM t")
        self.assertEqual(lineage[0]["target_column"], "y")
        self.assertEqual(lineage[0]["source_columns"][0]["column"], "x")

    def test_select_star(self):
        lineage = self._extract("SELECT * FROM users u JOIN orders o ON u.id = o.user_id")
        self.assertEqual(len(lineage), 1)
        self.assertEqual(lineage[0]["target_column"], "*")
        self.assertEqual(lineage[0]["source_columns"], [])
        self.assertIn("users", lineage[0]["used_tables"])
        self.assertIn("orders", lineage[0]["used_tables"])

    def test_select_star_with_schema_catalog(self):
        extractor = ColumnLineageExtractor(
            schema_catalog={
                "users": ["id", "name"],
                "orders": ["user_id", "total"],
            }
        )
        tree = self.parser.parse("SELECT * FROM users u JOIN orders o ON u.id = o.user_id")
        lineage = extractor.extract(tree)
        self.assertEqual(
            {entry["target_column"] for entry in lineage},
            {"id", "name", "user_id", "total"},
        )
        for entry in lineage:
            self.assertGreater(len(entry["source_columns"]), 0)

    def test_cte_transitive_lineage(self):
        sql = "WITH cte AS (SELECT id, amt FROM t) SELECT cte.id AS customer_id FROM cte"
        lineage = self._extract(sql)
        self.assertEqual(lineage[0]["target_column"], "customer_id")
        self.assertEqual(lineage[0]["source_columns"][0]["column"], "id")
        self.assertEqual(lineage[0]["source_columns"][0]["table"], "t")

    def test_aggregate_function(self):
        lineage = self._extract("SELECT SUM(amount) AS total FROM orders")
        self.assertEqual(lineage[0]["target_column"], "total")
        self.assertEqual(lineage[0]["source_columns"][0]["column"], "amount")

    def test_subquery_in_projection(self):
        sql = "SELECT (SELECT MAX(v) FROM inner_t) AS m FROM outer_t"
        lineage = self._extract(sql)
        self.assertEqual(lineage[0]["target_column"], "m")

    def test_no_select_raises(self):
        tree = self.parser.parse("CREATE TABLE t (id INT)")
        with self.assertRaises(LineageExtractionError):
            self.extractor.extract(tree)

    def test_lineage_contract_keys(self):
        lineage = self._extract("SELECT a FROM t")
        for entry in lineage:
            self.assertEqual(set(entry.keys()), LINEAGE_KEYS)


if __name__ == "__main__":
    unittest.main()
