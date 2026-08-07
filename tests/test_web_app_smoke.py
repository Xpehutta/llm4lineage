"""Lightweight smoke tests for the decomposed Streamlit web UI modules."""

from __future__ import annotations

import unittest


class TestWebServicesSmoke(unittest.TestCase):
    def test_pipeline_service_helpers(self):
        from Web.services.pipeline_service import (
            plpgsql_table_lineage,
            shorten_text,
            split_sql_statements,
            target_columns_from_result,
        )

        self.assertEqual(shorten_text("abcdefghij", 6), "abcde…")
        stmts = split_sql_statements("SELECT 1; SELECT 2;")
        self.assertGreaterEqual(len(stmts), 2)

        rolled = plpgsql_table_lineage(
            {
                "function": "analytics.build_daily_summary",
                "temp_tables": ["tmp_daily_orders"],
                "table_lineage_statements": [
                    {"target": "tmp_daily_orders", "sources": ["sales.orders"]},
                    {"target": "analytics.daily_summary", "sources": ["tmp_daily_orders"]},
                ],
            }
        )
        self.assertEqual(rolled["target"], "analytics.daily_summary")
        self.assertEqual(rolled["sources"], ["sales.orders"])

        cols = target_columns_from_result(
            {"extraction": {"output_columns": [{"alias": "a"}, {"alias": "b"}]}}
        )
        self.assertEqual(cols, ["a", "b"])

    def test_cache_service_helpers(self):
        from Web.services.cache_service import cache_status_captions, llm_config_key, make_llm_cache

        key = llm_config_key("model-a", "provider-b", "token-c")
        self.assertIn("model-a", key)
        self.assertIn("provider-b", key)
        self.assertIsNone(make_llm_cache(False))
        self.assertIsNotNone(make_llm_cache(True))
        captions = cache_status_captions({"hit": True, "quality_score": 0.9})
        self.assertEqual(len(captions), 1)
        self.assertIn("Cache hit", captions[0])

    def test_graph_view_table_dot(self):
        from Web.components.graph_view import build_table_lineage_dot

        dot = build_table_lineage_dot("tgt", ["src_a", "src_b"], highlight="src_a")
        source = dot.source
        self.assertIn("tgt", source)
        self.assertIn("src_a", source)
        self.assertIn("src_b", source)

    def test_component_modules_importable(self):
        from Web.components import graph_view, results_panel, sidebar, uploader

        self.assertTrue(callable(graph_view.build_table_lineage_dot))
        self.assertTrue(callable(uploader.resolve_active_sql))
        self.assertTrue(callable(sidebar.render_sidebar))
        self.assertTrue(callable(results_panel.render_lineage_results))


if __name__ == "__main__":
    unittest.main()
