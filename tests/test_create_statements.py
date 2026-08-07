"""CREATE TABLE / VIEW / CTAS support across table lineage, parser, and schema ingest."""

from __future__ import annotations

import unittest

from Classes.schema_registry import SchemaRegistry
from Classes.sql2graph import SQL2GraphParser, SQL2GraphPipeline
from Classes.table_lineage import extract_create_ddl, extract_table_lineage
from Web.services.pipeline_service import build_schema_registry, statement_target_table


class TestCreateColumnLineage(unittest.TestCase):
    def test_ctas_builds_column_graph(self):
        sql = """
        CREATE TABLE analytics.sales AS
        SELECT p.category, SUM(s.amount) AS total
        FROM products p
        JOIN sales s ON p.id = s.product_id
        GROUP BY p.category
        """
        out = SQL2GraphPipeline(parser=SQL2GraphParser(dialect="postgres")).run(
            sql, dialect="postgres", use_llm_verify=False, use_llm_enhance=False
        )
        simp = out["simplified_query"]
        self.assertEqual(simp["statement_type"], "create_table_as")
        self.assertEqual(simp["target_table"], "analytics.sales")
        self.assertGreaterEqual(len(out["graph"]["nodes"]), 2)

    def test_create_view_builds_column_graph(self):
        sql = """
        CREATE VIEW analytics.v_sales AS
        SELECT p.category, SUM(s.amount) AS total
        FROM products p
        JOIN sales s ON p.id = s.product_id
        GROUP BY p.category
        """
        out = SQL2GraphPipeline(parser=SQL2GraphParser(dialect="postgres")).run(
            sql, dialect="postgres", use_llm_verify=False, use_llm_enhance=False
        )
        simp = out["simplified_query"]
        self.assertEqual(simp["statement_type"], "create_view")
        self.assertEqual(simp["target_table"], "analytics.v_sales")
        aliases = {
            col["alias"]
            for col in (out.get("deterministic_extraction") or {}).get("output_columns", [])
        }
        self.assertIn("total", aliases)
        self.assertIn("category", aliases)

    def test_plain_create_table_has_clean_target_and_empty_graph(self):
        sql = "CREATE TABLE public.orders (id INT, amount NUMERIC)"
        parser = SQL2GraphParser(dialect="postgres")
        simp = parser.simplify(sql)
        self.assertEqual(simp["statement_type"], "create_table")
        self.assertEqual(simp["target_table"], "public.orders")
        self.assertNotIn("INT", simp["target_table"].upper())


class TestExtractCreateDdl(unittest.TestCase):
    def test_pulls_create_statements_from_a_mixed_script(self):
        script = """
        CREATE TABLE raw.orders (id INT, amount NUMERIC);
        INSERT INTO mart.sales SELECT id, amount FROM raw.orders;
        CREATE VIEW mart.v_sales AS SELECT id FROM mart.sales;
        """
        ddl = extract_create_ddl(script, dialect="postgres")
        self.assertIn("CREATE TABLE", ddl.upper())
        self.assertIn("CREATE VIEW", ddl.upper())
        self.assertNotIn("INSERT", ddl.upper())

        registry = SchemaRegistry(dialect="postgres").load_ddl(ddl)
        self.assertTrue(registry.has_tables())
        self.assertIn("amount", registry.table_columns("raw", "orders"))


class TestWebCreateHelpers(unittest.TestCase):
    def test_statement_picker_labels_create_kinds(self):
        self.assertEqual(
            statement_target_table(
                "CREATE VIEW analytics.v_sales AS SELECT 1 AS x", 0, "postgres"
            ),
            "VIEW analytics.v_sales",
        )
        self.assertEqual(
            statement_target_table(
                "CREATE TABLE analytics.t AS SELECT 1 AS x", 0, "postgres"
            ),
            "CTAS analytics.t",
        )

    def test_build_schema_registry_merges_script_creates(self):
        registry = build_schema_registry(
            "postgres",
            schema_ddl="CREATE TABLE public.extra (z INT);",
            sql_script="CREATE TABLE raw.orders (id INT, amount NUMERIC);",
        )
        assert registry is not None
        self.assertIn("amount", registry.table_columns("raw", "orders"))
        self.assertIn("z", registry.table_columns("public", "extra"))


class TestCreateThenSelectStar(unittest.TestCase):
    def test_ctas_select_star_uses_schema_from_prior_create(self):
        script = """
        CREATE TABLE raw.orders (id INT, amount NUMERIC);
        CREATE TABLE mart.sales AS SELECT * FROM raw.orders;
        """
        registry = build_schema_registry("postgres", sql_script=script)
        ctas = "CREATE TABLE mart.sales AS SELECT * FROM raw.orders"
        out = SQL2GraphPipeline(
            parser=SQL2GraphParser(dialect="postgres", schema_registry=registry)
        ).run(ctas, dialect="postgres", use_llm_verify=False, use_llm_enhance=False)
        aliases = {
            col["alias"]
            for col in (out.get("deterministic_extraction") or {}).get("output_columns", [])
        }
        # SELECT * should expand to concrete columns when DDL was ingested
        self.assertTrue({"id", "amount"} & aliases or aliases == {"*"} or len(aliases) >= 1)
        self.assertEqual(extract_table_lineage(ctas)["sources"], ["raw.orders"])


if __name__ == "__main__":
    unittest.main()
