import json
import unittest

from Classes.sql_chunk_classes import (
    SQLChunkGraph,
    SQLLogicalChunkParser,
    SQLLogicalChunkPreParser,
)


class _DummyChatAdapter:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = 0

    def invoke_messages(self, _messages):
        payload = self.payloads[self.calls]
        self.calls += 1
        return payload


SAMPLE_SQL = """
WITH recent_orders AS (
    SELECT customer_id, SUM(amount) AS total
    FROM orders
    WHERE order_date > '2025-01-01'
    GROUP BY customer_id
)
SELECT c.name,
    CASE WHEN c.vip THEN 'VIP' ELSE 'REG' END AS tier,
    r.total
FROM customers c
JOIN recent_orders r ON c.id = r.customer_id
WHERE c.active = true
"""

LLM_CHUNK_PAYLOAD = {
    "chunks": [
        {
            "id": "recent_orders",
            "name": "recent_orders",
            "chunk_type": "cte",
            "sql": (
                "SELECT customer_id, SUM(amount) AS total "
                "FROM orders WHERE order_date > '2025-01-01' GROUP BY customer_id"
            ),
        },
        {
            "id": "main",
            "name": "main",
            "chunk_type": "query",
            "sql": (
                "SELECT c.name, CASE WHEN c.vip THEN 'VIP' ELSE 'REG' END AS tier, r.total "
                "FROM customers c JOIN recent_orders r ON c.id = r.customer_id "
                "WHERE c.active = true"
            ),
        },
    ],
    "links": [
        {
            "source": "main",
            "target": "recent_orders",
            "link_type": "JOIN",
            "condition": "customers.id = recent_orders.customer_id",
        }
    ],
}


class TestSQLLogicalChunkPreParser(unittest.TestCase):
    def setUp(self):
        self.pre_parser = SQLLogicalChunkPreParser()

    def test_preparse_builds_cte_and_main_chunks(self):
        if not self.pre_parser.parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        seed = self.pre_parser.preparse(SAMPLE_SQL)
        chunk_names = {chunk["name"] for chunk in seed["chunks"]}
        self.assertIn("recent_orders", chunk_names)
        self.assertIn("main", chunk_names)

        cte = next(chunk for chunk in seed["chunks"] if chunk["name"] == "recent_orders")
        main = next(chunk for chunk in seed["chunks"] if chunk["name"] == "main")

        self.assertEqual(cte["chunk_type"], "cte")
        self.assertEqual(main["chunk_type"], "query")
        self.assertIn("SUM(amount)", cte["sql"])
        self.assertIn("FROM customers", main["sql"])
        self.assertIn("JOIN recent_orders", main["sql"])

    def test_preparse_builds_join_link(self):
        if not self.pre_parser.parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        seed = self.pre_parser.preparse(SAMPLE_SQL)
        self.assertTrue(seed["links"])
        join_links = [link for link in seed["links"] if link["link_type"] == "JOIN"]
        self.assertTrue(join_links)
        self.assertEqual(join_links[0]["source"], "main")
        self.assertEqual(join_links[0]["target"], "recent_orders")
        self.assertIn("customers.id", join_links[0]["condition"])
        self.assertIn("recent_orders.customer_id", join_links[0]["condition"])

    def test_preparse_detects_insert_target_and_union_branches(self):
        if not self.pre_parser.parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        insert_sql = "INSERT INTO target_table SELECT id FROM source_table"
        seed = self.pre_parser.preparse(insert_sql)
        self.assertEqual(seed["statement_type"], "insert")
        chunk_names = {chunk["name"] for chunk in seed["chunks"]}
        self.assertIn("target_table", chunk_names)
        self.assertIn("main", chunk_names)
        insert_links = [link for link in seed["links"] if link["link_type"] == "INSERT"]
        self.assertTrue(insert_links)


class TestSQLLogicalChunkParser(unittest.TestCase):
    def test_preparse_returns_only_chunks_and_links(self):
        parser = SQLLogicalChunkParser(hf_token=None)
        parser.pre_parser = SQLLogicalChunkPreParser()

        if not parser.pre_parser.parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        result = parser.preparse(SAMPLE_SQL)
        self.assertIn("chunks", result)
        self.assertIn("links", result)
        self.assertIn("statement_type", result)
        self.assertEqual(result["statement_type"], "select")
        self.assertEqual(result["metadata"]["pipeline_stage"], "deterministic")
        self.assertNotIn("graph", result)
        for chunk in result["chunks"]:
            self.assertNotIn("code", chunk)
        SQLChunkGraph.model_validate(
            {
                "chunks": result["chunks"],
                "links": result["links"],
                "statement_type": result["statement_type"],
                "target_table": result.get("target_table"),
            }
        )

    def test_preparse_without_hf_token(self):
        parser = SQLLogicalChunkParser(hf_token=None)
        if not parser.pre_parser.parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")
        self.assertIsNone(parser.chat_model)
        result = parser.preparse(SAMPLE_SQL)
        self.assertGreaterEqual(len(result["chunks"]), 2)

    def test_preparse_nested_union_produces_leaf_branches(self):
        from pathlib import Path

        pre_parser = SQLLogicalChunkPreParser()
        if not pre_parser.parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        raw = Path("data/SQL.txt").read_text(encoding="utf-8")
        sql = [statement.strip() for statement in raw.split(";") if statement.strip()][0]
        seed = pre_parser.preparse(sql)
        branch_chunks = [chunk for chunk in seed["chunks"] if chunk["chunk_type"] == "query"]
        self.assertEqual(len(branch_chunks), 3)
        for chunk in branch_chunks:
            self.assertNotRegex(chunk["sql"], r"\bUNION\s+ALL\b", chunk["id"])
        union_links = [link for link in seed["links"] if "UNION" in link["link_type"]]
        self.assertEqual(len(union_links), 2)

    def test_merge_seed_with_llm_drops_union_seed_when_llm_adds_branches(self):
        pre_parser = SQLLogicalChunkPreParser()
        if not pre_parser.parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        union_sql = "SELECT 1 AS id UNION ALL SELECT 2 AS id UNION ALL SELECT 3 AS id"
        seed = pre_parser.preparse(union_sql)
        llm_payload = {
            "chunks": [
                {"id": "left_branch", "name": "left_branch", "chunk_type": "query", "sql": "SELECT 1 AS id"},
                {"id": "middle_branch", "name": "middle_branch", "chunk_type": "query", "sql": "SELECT 2 AS id"},
                {"id": "right_branch", "name": "right_branch", "chunk_type": "query", "sql": "SELECT 3 AS id"},
            ],
            "links": [
                {"source": "left_branch", "target": "middle_branch", "link_type": "UNION ALL", "condition": ""},
                {"source": "middle_branch", "target": "right_branch", "link_type": "UNION ALL", "condition": ""},
            ],
        }
        merged = SQLLogicalChunkParser.merge_seed_with_llm(seed, llm_payload)
        merged_ids = {chunk["id"] for chunk in merged["chunks"]}
        self.assertNotIn("branch_0", merged_ids)
        self.assertIn("left_branch", merged_ids)
        self.assertEqual(len(merged["chunks"]), 3)

    def test_merge_seed_with_llm_preserves_seed_ids(self):
        pre_parser = SQLLogicalChunkPreParser()
        if not pre_parser.parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        seed = pre_parser.preparse(SAMPLE_SQL)
        merged = SQLLogicalChunkParser.merge_seed_with_llm(seed, LLM_CHUNK_PAYLOAD)
        self.assertIn("recent_orders", {chunk["id"] for chunk in merged["chunks"]})
        self.assertIn("main", {chunk["id"] for chunk in merged["chunks"]})
        self.assertTrue(merged["links"])

    def test_parse_with_llm_merges_and_validates(self):
        pre_parser = SQLLogicalChunkPreParser()
        if not pre_parser.parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        parser = SQLLogicalChunkParser.__new__(SQLLogicalChunkParser)
        parser.pre_parser = pre_parser
        parser.max_retries = 1
        parser.system_prompt = "system"
        parser.chat_adapter = _DummyChatAdapter([json.dumps(LLM_CHUNK_PAYLOAD)])
        parser.chat_model = None

        result = parser.parse(SAMPLE_SQL, use_llm=True)
        self.assertNotIn("error", result)
        self.assertEqual(len(result["chunks"]), 2)
        self.assertEqual(result["links"][0]["link_type"], "JOIN")
        self.assertEqual(result["metadata"]["pipeline_stage"], "llm_verified")
        self.assertIn("deterministic", result)
        self.assertEqual(len(result["deterministic"]["chunks"]), 2)

    def test_to_node_link_conversion(self):
        parser = SQLLogicalChunkParser.__new__(SQLLogicalChunkParser)
        node_link = parser.to_node_link(LLM_CHUNK_PAYLOAD)
        self.assertEqual(len(node_link["nodes"]), 2)
        self.assertEqual(len(node_link["links"]), 1)


if __name__ == "__main__":
    unittest.main()
