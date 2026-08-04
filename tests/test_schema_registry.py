"""Tests for schema registry and column qualification."""

from __future__ import annotations

import unittest

from Classes.schema_registry import DDLParser, SchemaRegistry
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


if __name__ == "__main__":
    unittest.main()
