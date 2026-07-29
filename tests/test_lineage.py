"""Tests for ColumnLineageExtractor."""

import unittest
from pathlib import Path

from Classes.pipeline.core.lineage import ColumnLineageExtractor
from Classes.pipeline.core.parser import SQLParser
from Classes.pipeline.exceptions import LineageExtractionError

LINEAGE_KEYS = {"target_column", "source_columns", "expression", "used_tables", "union_branches", "literal_values"}


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

    def test_union_literal_column_collects_branch_literals(self):
        sql = """
        SELECT 'a'::text AS attr_name
        UNION ALL
        SELECT 'b'::text AS attr_name
        """
        tree = self.parser.parse(sql)
        lineage = ColumnLineageExtractor(dialect="postgres").extract(tree)
        attr = next(entry for entry in lineage if entry["target_column"] == "attr_name")
        self.assertEqual(
            attr["literal_values"],
            [
                "'a'::text AS attr_name",
                "'b'::text AS attr_name",
            ],
        )
        self.assertEqual(len(attr["union_branches"]), 2)
        self.assertTrue(all(branch.get("kind") == "literal" for branch in attr["union_branches"]))

    def test_union_literal_column_has_no_positional_source(self):
        sql = """
        WITH p AS (
          SELECT 'a'::text AS attr_name
          UNION ALL
          SELECT 'b'::text AS attr_name
        )
        SELECT p.attr_name FROM p
        """
        lineage = self._extract(sql)
        attr = next(entry for entry in lineage if entry["target_column"] == "attr_name")
        self.assertEqual(attr["source_columns"], [{"table": "p", "column": "attr_name"}])

    def test_union_mixed_literal_and_column_sources(self):
        sql = """
        SELECT 1::numeric AS val
        UNION ALL
        SELECT mkt_price_amt FROM t
        """
        lineage = self._extract(sql)
        val = next(entry for entry in lineage if entry["target_column"] == "val")
        columns = {ref["column"] for ref in val["source_columns"]}
        self.assertIn("mkt_price_amt", columns)
        self.assertNotIn("1", columns)

    def test_ddls10_attr_name_not_positional_union_index(self):
        sql = Path(__file__).resolve().parents[1].joinpath("data/DDLs_10.txt").read_text().split(";")[0].strip()
        lineage = self._extract(sql)
        attr = next(entry for entry in lineage if entry["target_column"] == "attr_name")
        for ref in attr["source_columns"]:
            self.assertFalse(str(ref["column"]).isdigit())
        self.assertEqual(attr["source_columns"], [{"table": "t1", "column": "attr_name"}])

    def test_ddls10_attr_name_resolves_through_cte_to_literals(self):
        from Classes import SQL2GraphParser

        sql = Path(__file__).resolve().parents[1].joinpath("data/DDLs_10.txt").read_text().split(";")[0].strip()
        parser = SQL2GraphParser(dialect="postgres")
        extraction = parser.build_deterministic_extraction(parser.simplify(sql, dialect="postgres"), dialect="postgres")
        attr = next(col for col in extraction["output_columns"] if col["alias"] == "attr_name")
        self.assertEqual(attr["dependencies"], [])
        self.assertEqual(attr["derivation_kind"], "literal")
        self.assertEqual(
            attr["literal_values"],
            [
                "'mkt_price_amt'::text AS attr_name",
                "'mkt_price_rub'::text AS attr_name",
                "'mkt_crncy_id'::text AS attr_name",
                "'agr_collat_qlty_cat_type_id'::text AS attr_name",
                "'asset_collat_type_id'::text AS attr_name",
            ],
        )
        self.assertTrue(all(branch.get("kind") == "literal" for branch in attr["union_branches"]))

    def test_ddls10_end_dt_uses_cte_passthrough_and_physical_union_sources(self):
        from Classes import SQL2GraphParser

        sql = Path(__file__).resolve().parents[1].joinpath("data/DDLs_10.txt").read_text().split(";")[0].strip()
        parser = SQL2GraphParser(dialect="postgres")
        extraction = parser.build_deterministic_extraction(parser.simplify(sql, dialect="postgres"), dialect="postgres")
        end_dt = next(col for col in extraction["output_columns"] if col["alias"] == "end_dt")
        self.assertEqual(end_dt["derivation_kind"], "cte_passthrough")
        physical_tables = {
            dep.get("physical_table") or dep.get("table_alias")
            for dep in end_dt["dependencies"]
        }
        self.assertEqual(
            physical_tables,
            {
                "s_grnplm_vd_t_bvd_db_dmcl.a_agr_collat_mkt_period",
                "s_grnplm_vd_t_bvd_db_dmcl.a_agr_collat_qlty_period",
            },
        )
        self.assertEqual(len(end_dt["union_branches"]), 5)

    def test_lineage_contract_keys(self):
        lineage = self._extract("SELECT a FROM t")
        for entry in lineage:
            self.assertEqual(set(entry.keys()), LINEAGE_KEYS)


if __name__ == "__main__":
    unittest.main()
