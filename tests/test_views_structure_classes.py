import csv
import json
import tempfile
import unittest

from Classes.views_structure_classes import ViewsStructureExtractor


class _DummyChatAdapter:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = 0

    def invoke_messages(self, _messages):
        payload = self.payloads[self.calls]
        self.calls += 1
        return payload


class TestViewsStructureExtractor(unittest.TestCase):
    def test_extract_view_structure_success(self):
        payload = {
            "view_name": "v_sales",
            "source_tables": ["sales.orders", "sales.customers"],
            "output_columns": [
                {
                    "name": "customer_id",
                    "expression": "o.customer_id",
                    "source_columns": ["o.customer_id"],
                }
            ],
            "joins": [
                {
                    "join_type": "INNER",
                    "left": "o.customer_id",
                    "right": "c.customer_id",
                    "condition": "o.customer_id = c.customer_id",
                }
            ],
            "filters": ["o.created_at >= '2026-01-01'"],
            "ctes": ["base_orders"],
        }
        extractor = ViewsStructureExtractor.__new__(ViewsStructureExtractor)
        extractor.max_retries = 1
        extractor.llm_pause_seconds = 0.0
        extractor.system_prompt = "system"
        extractor.chat_adapter = _DummyChatAdapter([json.dumps(payload)])
        extractor.chat_model = None

        result = extractor.extract_view_structure(
            view_name="v_sales",
            view_sql="SELECT o.customer_id FROM sales.orders o JOIN sales.customers c ON o.customer_id = c.customer_id",
        )
        self.assertNotIn("error", result)
        self.assertEqual(result["view_name"], "v_sales")
        self.assertEqual(result["source_tables"], ["sales.orders", "sales.customers"])
        self.assertEqual(
            result["output_columns"][0]["source_columns"][0],
            "sales.orders.customer_id",
        )
        self.assertIn("source_tables_structure", result)
        self.assertEqual(len(result["source_tables_structure"]), 2)
        self.assertEqual(result["source_tables_structure"][0]["full_name"], "sales.orders")
        self.assertEqual(extractor.chat_adapter.calls, 1)

    def test_extract_view_structure_fallback_on_invalid_payload(self):
        extractor = ViewsStructureExtractor.__new__(ViewsStructureExtractor)
        extractor.max_retries = 1
        extractor.llm_pause_seconds = 0.0
        extractor.system_prompt = "system"
        extractor.chat_adapter = _DummyChatAdapter(['{"bad_json": true}'])
        extractor.chat_model = None

        sql = "SELECT * FROM sales.orders o JOIN sales.customers c ON o.customer_id = c.customer_id WHERE o.flag = 1"
        result = extractor.extract_view_structure(view_name="v_bad", view_sql=sql)
        self.assertEqual(result["view_name"], "v_bad")
        self.assertEqual(result["source_tables"], [])
        self.assertIn("source_tables_structure", result)

    def test_extract_from_csv_with_limit_and_filter(self):
        extractor = ViewsStructureExtractor.__new__(ViewsStructureExtractor)
        extractor.max_retries = 1
        extractor.llm_pause_seconds = 0.0
        extractor.system_prompt = "system"
        extractor.chat_adapter = _DummyChatAdapter(
            [
                json.dumps(
                    {
                        "view_name": "v_one",
                        "source_tables": ["schema.a"],
                        "output_columns": [],
                        "joins": [],
                        "filters": [],
                        "ctes": [],
                    }
                )
            ]
        )
        extractor.chat_model = None

        with tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False) as tmp:
            writer = csv.DictWriter(tmp, fieldnames=["table_name", "view_def"])
            writer.writeheader()
            writer.writerow({"table_name": "v_one", "view_def": "SELECT 1 FROM schema.a"})
            writer.writerow({"table_name": "v_two", "view_def": "SELECT 1 FROM schema.b"})
            csv_path = tmp.name

        out = extractor.extract_from_csv(csv_path=csv_path, limit=1, include_tables=["v_one"])
        self.assertEqual(out["views_count"], 1)
        self.assertEqual(out["views"][0]["view_name"], "v_one")
        self.assertEqual(out["views"][0]["source_tables_structure"][0]["table"], "a")


if __name__ == "__main__":
    unittest.main()
