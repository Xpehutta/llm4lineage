"""Tests for PipelineOrchestrator."""

import unittest

from Classes.pipeline.core.orchestrator import PipelineOrchestrator
from Classes.pipeline.models.config import Config

SAMPLE_SQL = (
    "SELECT u.name, SUM(o.amount) AS total "
    "FROM users u JOIN orders o ON u.id = o.user_id "
    "GROUP BY u.name"
)


class TestPipelineOrchestrator(unittest.TestCase):
    def setUp(self):
        self.config = Config(llm_provider="mock")
        self.orchestrator = PipelineOrchestrator(self.config)

    def test_full_pipeline_success(self):
        result = self.orchestrator.run(
            SAMPLE_SQL,
            instruction="Explain the query.",
        )
        self.assertTrue(result.success)
        self.assertEqual(result.original_sql, SAMPLE_SQL)
        self.assertIsInstance(result.ast_json, dict)
        self.assertGreater(len(result.column_lineage), 0)
        self.assertTrue(result.llm_response)
        self.assertGreater(result.latency_seconds, 0.0)

    def test_broken_sql_graceful_degradation(self):
        result = self.orchestrator.run("SELECT FROM")
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.ast_json, {})
        self.assertEqual(result.column_lineage, [])
        self.assertEqual(result.llm_response, "")

    def test_batch_never_crashes(self):
        queries = ["SELECT a FROM t", "SELECT FROM", "SELECT b FROM u"]
        for sql in queries:
            result = self.orchestrator.run(sql)
            self.assertIsNotNone(result)
            self.assertEqual(result.original_sql, sql)

    def test_schema_catalog_expands_select_star(self):
        orchestrator = PipelineOrchestrator(
            self.config,
            schema_catalog={"t": ["a", "b"]},
        )
        result = orchestrator.run("SELECT * FROM t")
        self.assertTrue(result.success)
        self.assertEqual(
            {entry["target_column"] for entry in result.column_lineage},
            {"a", "b"},
        )


if __name__ == "__main__":
    unittest.main()
