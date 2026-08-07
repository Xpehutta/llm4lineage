"""End-to-end checks for PL/pgSQL routing through SQL2GraphPipeline (Phase A4)."""

import unittest
from pathlib import Path

from Classes.sql2graph_classes import SQL2GraphParser, SQL2GraphPipeline

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "plpgsql_functions.sql"

PLAIN_SQL = "INSERT INTO analytics.t (a) SELECT o.a FROM sales.orders o"


def load_function(name: str) -> str:
    text = FIXTURE.read_text(encoding="utf-8")
    return text[text.index(f"CREATE OR REPLACE FUNCTION {name}") :]


def make_pipeline() -> SQL2GraphPipeline:
    return SQL2GraphPipeline(parser=SQL2GraphParser(dialect="postgres"))


class TestPlpgsqlRouting(unittest.TestCase):
    def test_function_produces_graph_and_unresolved_report(self):
        result = make_pipeline().run(
            load_function("staging.load_partition"),
            use_llm_verify=False,
            use_llm_enhance=False,
            parse_plpgsql=True,
        )
        self.assertEqual(result["pipeline_stage"], "plpgsql")
        self.assertTrue(result["graph"]["nodes"])
        self.assertTrue(result["unresolved"])
        self.assertEqual(result["function"], "staging.load_partition")
        self.assertNotIn("error", result)

    def test_temp_table_chain_survives_the_pipeline(self):
        result = make_pipeline().run(
            load_function("analytics.build_daily_summary"),
            use_llm_verify=False,
            use_llm_enhance=False,
            parse_plpgsql=True,
        )
        edges = {
            (link["source"], link["target"])
            for link in result["graph"]["links"]
            if link.get("edge_type") == "DERIVED_FROM"
        }
        self.assertIn(("sales.orders.amount", "tmp_daily_orders.amount"), edges)
        self.assertIn(
            ("tmp_daily_orders.amount", "analytics.daily_summary.total_amount"), edges
        )
        self.assertIn("tmp_daily_orders", result["temp_tables"])

    def test_all_five_steps_are_reported(self):
        seen = []
        make_pipeline().run(
            load_function("analytics.route_customer"),
            use_llm_verify=False,
            use_llm_enhance=False,
            parse_plpgsql=True,
            step_callback=lambda name, step, all_steps: seen.append(name),
        )
        for step in SQL2GraphPipeline.PIPELINE_STEP_ORDER:
            self.assertIn(step, seen)

    def test_flag_off_leaves_plain_sql_untouched(self):
        pipeline = make_pipeline()
        without = pipeline.run(PLAIN_SQL, use_llm_verify=False, use_llm_enhance=False)
        with_flag = pipeline.run(
            PLAIN_SQL, use_llm_verify=False, use_llm_enhance=False, parse_plpgsql=True
        )
        self.assertEqual(without["pipeline_stage"], "deterministic")
        self.assertEqual(with_flag["pipeline_stage"], "deterministic")
        self.assertEqual(
            [n["id"] for n in without["graph"]["nodes"]],
            [n["id"] for n in with_flag["graph"]["nodes"]],
        )

    def test_function_without_flag_is_not_routed(self):
        result = make_pipeline().run(
            load_function("staging.load_partition"),
            use_llm_verify=False,
            use_llm_enhance=False,
        )
        self.assertNotEqual(result.get("pipeline_stage"), "plpgsql")

    def test_response_keeps_the_standard_shape(self):
        result = make_pipeline().run(
            load_function("analytics.route_customer"),
            use_llm_verify=False,
            use_llm_enhance=False,
            parse_plpgsql=True,
        )
        for key in (
            "graph",
            "metadata",
            "warnings",
            "extraction",
            "pipeline_stage",
            "pipeline_steps",
            "chunks",
            "simplified_query",
            "cache",
        ):
            self.assertIn(key, result)


class TestWebRollup(unittest.TestCase):
    def test_plpgsql_table_lineage_rollup(self):
        import sys

        sys.path.insert(0, str(ROOT))
        from Web.app import plpgsql_table_lineage

        result = {
            "function": "analytics.build_daily_summary",
            "temp_tables": ["tmp_daily_orders"],
            "table_lineage_statements": [
                {"target": "tmp_daily_orders", "sources": ["sales.orders"]},
                {"target": "analytics.daily_summary", "sources": ["tmp_daily_orders"]},
            ],
        }
        rolled = plpgsql_table_lineage(result)
        self.assertEqual(rolled["target"], "analytics.daily_summary")
        self.assertEqual(rolled["sources"], ["sales.orders"])
        self.assertEqual(rolled["statement_type"], "plpgsql")


if __name__ == "__main__":
    unittest.main()
