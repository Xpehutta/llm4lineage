"""Phase 1 integration tests on real DWH fixtures."""

from __future__ import annotations

import unittest
from pathlib import Path

from Classes.schema_registry import SchemaRegistry
from Classes.sql2graph_classes import SQL2GraphParser
from Classes.table_lineage import extract_table_lineage

ROOT = Path(__file__).resolve().parents[1]


class TestPhase1Integration(unittest.TestCase):
    def test_ddls_txt_chunked_registry_loads(self):
        ddl_path = ROOT / "data" / "DDLs.txt"
        if not ddl_path.exists():
            self.skipTest("DDLs.txt missing")

        ddl_text = ddl_path.read_text(encoding="utf-8")
        registry = SchemaRegistry(dialect="postgres").load_sql_corpus(ddl_text, chunk_size=50)
        self.assertTrue(registry.has_tables())
        self.assertIn("query_output", registry.tables.get("inferred", {}))

    def test_sql_txt_first_statement_parses_with_schema(self):
        sql_path = ROOT / "data" / "SQL.txt"
        ddl_path = ROOT / "data" / "DDLs.txt"
        if not sql_path.exists() or not ddl_path.exists():
            self.skipTest("SQL.txt or DDLs.txt missing")

        registry = SchemaRegistry(dialect="postgres").load_sql_corpus(
            ddl_path.read_text(encoding="utf-8"),
            chunk_size=50,
        )
        sql = sql_path.read_text(encoding="utf-8").split(";")[0].strip()
        parser = SQL2GraphParser(dialect="postgres", schema_registry=registry)
        simplified = parser.simplify(sql, use_schema=True)
        self.assertTrue(simplified.get("parser_used"))
        self.assertTrue(simplified.get("target_table"))

    def test_merge_and_update_table_lineage(self):
        merge = extract_table_lineage(
            "MERGE INTO schema.tgt t USING schema.src s ON t.id = s.id WHEN MATCHED THEN UPDATE SET x = 1",
            dialect="postgres",
        )
        self.assertEqual(merge["statement_type"], "merge")
        self.assertEqual(merge["target"], "schema.tgt")
        self.assertIn("schema.src", merge["sources"])

        update = extract_table_lineage(
            "UPDATE schema.t SET x = 1 FROM schema.s WHERE t.id = s.id",
            dialect="postgres",
        )
        self.assertEqual(update["statement_type"], "update")
        self.assertEqual(update["target"], "schema.t")
        self.assertIn("schema.s", update["sources"])


if __name__ == "__main__":
    unittest.main()
