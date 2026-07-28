"""Tests for PipelineResult."""

import unittest

from Classes.pipeline.models.result import PipelineResult


class TestPipelineResult(unittest.TestCase):
    def test_success_when_no_error(self):
        result = PipelineResult(original_sql="SELECT 1")
        self.assertTrue(result.success)
        self.assertIsNone(result.error)

    def test_failure_when_error_set(self):
        result = PipelineResult(
            original_sql="bad",
            error="parse failed",
        )
        self.assertFalse(result.success)

    def test_defaults(self):
        result = PipelineResult(original_sql="SELECT 1")
        self.assertEqual(result.ast_json, {})
        self.assertEqual(result.column_lineage, [])
        self.assertEqual(result.llm_response, "")
        self.assertEqual(result.latency_seconds, 0.0)
        self.assertEqual(result.model_used, "")


if __name__ == "__main__":
    unittest.main()
