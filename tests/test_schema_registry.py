"""Tests for schema registry and column qualification."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sqlglot

from Classes.schema_registry import DDLParser, SchemaRegistry, _normalize_table_key
from Classes.sql2graph_classes import SQL2GraphParser


class TestSchemaRegistry(unittest.TestCase):
    def test_ddl_parser_extracts_columns(self):
        ddl = """
        CREATE TABLE sales.orders (
            order_id INT,
            customer_id INT,
            amount NUMERIC
        );
        """
        schema = DDLParser(dialect="postgres").parse_text(ddl)
        self.assertIn("sales", schema)
        self.assertIn("orders", schema["sales"])
        self.assertEqual(
            set(schema["sales"]["orders"].keys()),
            {"order_id", "customer_id", "amount"},
        )

    def test_registry_merge_and_csv(self):
        registry = SchemaRegistry(dialect="postgres")
        registry.load_ddl("CREATE TABLE public.users (id INT, name TEXT);")
        registry.load_csv("schema,table,column,type\npublic,users,email,TEXT\n")
        columns = registry.table_columns("public", "users")
        self.assertIn("id", columns)
        self.assertIn("email", columns)

    def test_qualify_select_star_with_schema(self):
        parser = SQL2GraphParser(dialect="postgres")
        if not parser.sqlglot_available:
            self.skipTest("sqlglot not installed")

        registry = SchemaRegistry(dialect="postgres").load_ddl(
            "CREATE TABLE public.users (id INT, name TEXT);"
        )
        parser.schema_registry = registry
        simplified = parser.simplify("SELECT * FROM public.users u", use_schema=True)
        self.assertTrue(simplified.get("parser_used"))
        self.assertTrue(simplified.get("schema_applied"))
        aliases = simplified.get("select", {}).get("aliases") or []
        self.assertGreaterEqual(len(aliases), 2)

    def test_view_ddl_is_registered(self):
        registry = SchemaRegistry(dialect="postgres").load_ddl(
            "CREATE VIEW sales.v_orders AS SELECT order_id FROM sales.orders;"
        )
        self.assertTrue(registry.is_view("sales", "v_orders"))
        self.assertIn("order_id", registry.table_columns("sales", "v_orders"))

    def test_without_schema_keeps_current_behavior(self):
        parser = SQL2GraphParser(dialect="postgres")
        if not parser.sqlglot_available:
            self.skipTest("sqlglot not installed")

        simplified = parser.simplify("SELECT a, b FROM t", use_schema=True)
        self.assertTrue(simplified.get("parser_used"))
        self.assertFalse(simplified.get("schema_applied"))


class TestNormalizeTableKey(unittest.TestCase):
    def test_bare_table_defaults_to_the_public_schema(self):
        self.assertEqual(_normalize_table_key(None, "Orders"), ("public", "orders"))

    def test_dotted_table_wins_over_the_schema_argument(self):
        self.assertEqual(_normalize_table_key("ignored", "Sales.Orders"), ("sales", "orders"))

    def test_three_part_names_keep_the_last_two_segments(self):
        self.assertEqual(_normalize_table_key(None, "wh.sales.orders"), ("sales", "orders"))

    def test_quotes_and_whitespace_are_stripped(self):
        self.assertEqual(_normalize_table_key(' "Sales" ', ' "Orders" '), ("sales", "orders"))


class TestDDLParser(unittest.TestCase):
    def test_empty_ddl_yields_nothing(self):
        self.assertEqual(DDLParser().parse_registry(""), ({}, {}))

    def test_empty_statements_between_semicolons_are_skipped(self):
        schema = DDLParser().parse_text(";; CREATE TABLE public.t (a INT); ;")
        self.assertEqual(schema, {"public": {"t": {"a": "INT"}}})

    def test_non_create_statements_are_ignored(self):
        self.assertEqual(DDLParser().parse_text("SELECT 1; INSERT INTO t VALUES (1);"), {})

    def test_column_types_are_rendered_in_the_configured_dialect(self):
        schema = DDLParser(dialect="postgres").parse_text(
            "CREATE TABLE public.t (a NUMERIC(10, 2), b VARCHAR(5));"
        )
        self.assertEqual(schema["public"]["t"]["a"], "DECIMAL(10, 2)")
        self.assertEqual(schema["public"]["t"]["b"], "VARCHAR(5)")

    def test_create_without_a_target_is_ignored(self):
        parser = DDLParser()
        schema: dict = {}
        statement = sqlglot.parse_one("CREATE TABLE public.t (a INT)", read="postgres")
        statement.set("this", None)

        parser._ingest_create(statement, schema)

        self.assertEqual(schema, {})


class TestRegistryLoading(unittest.TestCase):
    def test_csv_accepts_the_long_column_names(self):
        registry = SchemaRegistry().load_csv(
            "schema_name,table_name,column_name,data_type\npublic,users,id,INT\n"
        )
        self.assertEqual(registry.table_columns("public", "users"), {"id": "INT"})

    def test_csv_rows_without_a_table_or_column_are_skipped(self):
        registry = SchemaRegistry().load_csv(
            "schema,table,column,type\n"
            "public,,id,INT\n"
            "public,users,,INT\n"
            "public,users,id,INT\n"
        )
        self.assertEqual(registry.tables, {"public": {"users": {"id": "INT"}}})

    def test_csv_delimiter_can_be_overridden(self):
        registry = SchemaRegistry().load_csv("schema;table;column;type\npublic;t;a;INT\n", delimiter=";")
        self.assertEqual(registry.table_columns("public", "t"), {"a": "INT"})

    def test_merging_another_registry_unions_the_columns(self):
        first = SchemaRegistry().load_ddl("CREATE TABLE public.t (a INT);")
        second = SchemaRegistry().load_ddl("CREATE TABLE public.t (b TEXT);")

        first.merge(second)

        self.assertEqual(set(first.table_columns("public", "t")), {"a", "b"})

    def test_has_tables_reflects_content(self):
        self.assertFalse(SchemaRegistry().has_tables())
        self.assertTrue(SchemaRegistry().load_ddl("CREATE TABLE public.t (a INT);").has_tables())

    def test_from_ddl_file_reads_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schema.sql"
            path.write_text("CREATE TABLE public.t (a INT);", encoding="utf-8")

            registry = SchemaRegistry.from_ddl_file(str(path))

        self.assertEqual(registry.table_columns("public", "t"), {"a": "INT"})

    def test_unknown_table_has_no_columns(self):
        self.assertEqual(SchemaRegistry().table_columns("public", "missing"), {})


class TestSplitAndChunkedLoading(unittest.TestCase):
    DDL = "".join(f"CREATE TABLE public.t{index} (a INT);\n" for index in range(5))

    def test_split_statements_of_empty_text(self):
        self.assertEqual(SchemaRegistry().split_statements(""), [])

    def test_split_statements_returns_one_entry_per_create(self):
        statements = SchemaRegistry().split_statements(self.DDL)
        self.assertEqual(len(statements), 5)
        self.assertTrue(all(stmt.startswith("CREATE TABLE") for stmt in statements))

    def test_split_statements_falls_back_to_a_regex_without_sqlglot(self):
        with patch("Classes.schema_registry.sqlglot", None):
            statements = SchemaRegistry().split_statements("CREATE TABLE a (x INT); SELECT 1;")
        self.assertEqual(statements, ["CREATE TABLE a (x INT)", "SELECT 1"])

    def test_chunked_loading_below_the_threshold_loads_in_one_pass(self):
        registry = SchemaRegistry().load_ddl_chunked(self.DDL, chunk_size=200)
        self.assertEqual(len(registry.tables["public"]), 5)

    def test_chunked_loading_above_the_threshold_covers_every_statement(self):
        registry = SchemaRegistry().load_ddl_chunked(self.DDL, chunk_size=2)
        self.assertEqual(set(registry.tables["public"]), {f"t{index}" for index in range(5)})

    def test_chunked_loading_registers_views_from_every_chunk(self):
        ddl = self.DDL + "CREATE VIEW public.v AS SELECT a FROM public.t0;\n"
        registry = SchemaRegistry().load_ddl_chunked(ddl, chunk_size=2)
        self.assertTrue(registry.is_view("public", "v"))
        self.assertIsNotNone(registry.get_view_select("public", "v"))

    def test_chunked_loading_of_empty_text_is_a_no_op(self):
        registry = SchemaRegistry().load_ddl_chunked("")
        self.assertFalse(registry.has_tables())


class TestInference(unittest.TestCase):
    def test_select_aliases_are_recorded_as_inferred_output_columns(self):
        registry = SchemaRegistry().infer_from_sql("SELECT a, b AS renamed FROM t")
        self.assertEqual(
            set(registry.tables["inferred"]["query_output"]), {"a", "renamed"}
        )

    def test_unparseable_sql_is_ignored(self):
        self.assertFalse(SchemaRegistry().infer_from_sql("NOT SQL ((").has_tables())

    def test_blank_sql_is_ignored(self):
        self.assertFalse(SchemaRegistry().infer_from_sql("   ").has_tables())

    def test_a_statement_without_a_select_is_ignored(self):
        self.assertFalse(SchemaRegistry().infer_from_sql("CREATE TABLE t (a INT)").has_tables())

    def test_corpus_loading_walks_every_statement(self):
        registry = SchemaRegistry().load_sql_corpus(
            "SELECT a FROM t; SELECT b FROM u;", chunk_size=1
        )
        self.assertEqual(set(registry.tables["inferred"]["query_output"]), {"a", "b"})


class TestQualifyExpression(unittest.TestCase):
    def test_expression_is_returned_untouched_without_a_schema(self):
        expression = sqlglot.parse_one("SELECT * FROM public.users", read="postgres")
        self.assertIs(SchemaRegistry().qualify_expression(expression), expression)

    def test_none_is_passed_through(self):
        self.assertIsNone(SchemaRegistry().qualify_expression(None))

    def test_star_is_expanded_when_the_schema_is_known(self):
        registry = SchemaRegistry().load_ddl("CREATE TABLE public.users (id INT, name TEXT);")
        expression = sqlglot.parse_one("SELECT * FROM public.users", read="postgres")

        qualified = registry.qualify_expression(expression)

        self.assertIn("id", qualified.sql(dialect="postgres"))
        self.assertIn("name", qualified.sql(dialect="postgres"))

    def test_a_qualification_failure_returns_the_original_expression(self):
        registry = SchemaRegistry().load_ddl("CREATE TABLE public.users (id INT);")
        expression = sqlglot.parse_one("SELECT * FROM public.users", read="postgres")

        with patch(
            "sqlglot.optimizer.qualify_columns.qualify_columns",
            side_effect=RuntimeError("boom"),
        ):
            self.assertIs(registry.qualify_expression(expression), expression)


class TestTableKeysFromExpression(unittest.TestCase):
    def test_missing_expression_defaults_to_public(self):
        self.assertEqual(SchemaRegistry.table_keys_from_expression(None), ("public", ""))

    def test_qualified_table_expression(self):
        table = sqlglot.parse_one("SELECT * FROM Sales.Orders", read="postgres").find(
            sqlglot.exp.Table
        )
        self.assertEqual(
            SchemaRegistry.table_keys_from_expression(table), ("sales", "orders")
        )

    def test_unqualified_table_expression_defaults_to_public(self):
        table = sqlglot.parse_one("SELECT * FROM orders", read="postgres").find(
            sqlglot.exp.Table
        )
        self.assertEqual(
            SchemaRegistry.table_keys_from_expression(table), ("public", "orders")
        )


if __name__ == "__main__":
    unittest.main()
