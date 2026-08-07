"""Structured output, parse-error reporting and edge provenance (Phase D)."""

import json
import unittest
from pathlib import Path

from Classes.helper_classes import SQLDependencies
from Classes.model_classes import REGEX_FALLBACK_CONFIDENCE, SQLLineageOutputParser
from Classes.pipeline.core.llm_factory import LLMFactory
from Classes.pipeline.core.llm_interface import MockLLM
from Classes.pipeline.core.orchestrator import PipelineOrchestrator
from Classes.pipeline.models.config import Config
from Classes.sql2graph_classes import SQL2GraphParser, SQL2GraphPipeline

PROMPT_DIR = Path(__file__).resolve().parent.parent / "Classes" / "pipeline" / "prompts"

VALID_RESPONSE = json.dumps(
    {
        "target": "analytics.sales",
        "sources": ["raw.orders", "raw.customers"],
        "reasoning": "The INSERT reads both tables.",
        "confidence": 0.9,
    }
)


class TestPromptSchema(unittest.TestCase):
    """D1: the prompts must state the response contract."""

    def test_system_prompt_declares_every_field(self):
        text = (PROMPT_DIR / "system.txt").read_text(encoding="utf-8")
        for field in ("target", "sources", "reasoning", "confidence"):
            self.assertIn(f'"{field}"', text)

    def test_system_prompt_forbids_prose(self):
        text = (PROMPT_DIR / "system.txt").read_text(encoding="utf-8").lower()
        self.assertIn("json", text)
        self.assertIn("no prose", text)

    def test_human_prompt_keeps_its_placeholders(self):
        text = (PROMPT_DIR / "human.txt").read_text(encoding="utf-8")
        for placeholder in ("{instruction}", "{ast_json}", "{column_lineage}"):
            self.assertIn(placeholder, text)

    def test_schema_braces_survive_rendering(self):
        """The JSON schema in the prompt must not be eaten by placeholder substitution."""
        orchestrator = PipelineOrchestrator(Config(llm_provider="mock"))
        rendered = orchestrator.chain.chain.invoke  # smoke: chain is built
        self.assertTrue(callable(rendered))
        system_text = orchestrator.chain.chain.system_text
        self.assertIn('"target"', system_text)


class TestJsonModeConfiguration(unittest.TestCase):
    """D2: providers that can enforce JSON should be told to."""

    def test_json_mode_is_on_by_default(self):
        self.assertTrue(Config().llm_json_mode)

    def test_temperature_capped_while_json_mode_is_on(self):
        config = Config(llm_temperature=0.9, llm_json_mode=True)
        self.assertEqual(LLMFactory.effective_temperature(config), 0.1)

    def test_low_temperature_is_left_alone(self):
        config = Config(llm_temperature=0.0, llm_json_mode=True)
        self.assertEqual(LLMFactory.effective_temperature(config), 0.0)

    def test_temperature_untouched_when_json_mode_is_off(self):
        config = Config(llm_temperature=0.9, llm_json_mode=False)
        self.assertEqual(LLMFactory.effective_temperature(config), 0.9)


class TestStructuredParsing(unittest.TestCase):
    """D3: schema first, regex only as a marked fallback."""

    def setUp(self):
        self.parser = SQLLineageOutputParser()

    def test_valid_json_is_the_primary_path(self):
        parsed = self.parser.parse(VALID_RESPONSE)
        self.assertEqual(parsed.target, "analytics.sales")
        self.assertEqual(parsed.sources, ["raw.orders", "raw.customers"])
        self.assertEqual(parsed.reasoning, "The INSERT reads both tables.")
        self.assertEqual(parsed.confidence, 0.9)
        self.assertEqual(parsed.provenance, "json")
        self.assertIsNone(parsed.parse_error)

    def test_json_wrapped_in_prose_is_still_parsed(self):
        parsed = self.parser.parse(f"Here you go:\n```json\n{VALID_RESPONSE}\n```")
        self.assertEqual(parsed.provenance, "json")
        self.assertEqual(parsed.target, "analytics.sales")

    def test_regex_fallback_is_marked_and_downweighted(self):
        parsed = self.parser.parse(
            "target: analytics.sales, sources: [raw.orders, raw.customers]"
        )
        self.assertEqual(parsed.target, "analytics.sales")
        self.assertEqual(parsed.provenance, "regex")
        self.assertEqual(parsed.confidence, REGEX_FALLBACK_CONFIDENCE)
        self.assertLess(parsed.confidence, 0.5)
        self.assertTrue(parsed.parse_error)

    def test_garbage_reports_parse_error_without_raising(self):
        parsed = self.parser.parse("I'm sorry, I can't help with that.")
        self.assertEqual(parsed.target, "")
        self.assertEqual(parsed.sources, [])
        self.assertLess(parsed.confidence, 0.5)
        self.assertEqual(parsed.provenance, "none")
        self.assertTrue(parsed.parse_error)

    def test_empty_response_reports_parse_error(self):
        parsed = self.parser.parse("")
        self.assertEqual(parsed.confidence, 0.0)
        self.assertIn("empty", parsed.parse_error.lower())

    def test_malformed_json_falls_back_and_explains_why(self):
        parsed = self.parser.parse('{"target": "a.b", "sources": ["c.d",}')
        self.assertEqual(parsed.provenance, "regex")
        self.assertIn("json", parsed.parse_error.lower())

    def test_non_object_json_is_rejected(self):
        parsed = self.parser.parse("[1, 2, 3]")
        self.assertEqual(parsed.provenance, "none")
        self.assertTrue(parsed.parse_error)

    def test_string_source_is_coerced_to_a_list(self):
        parsed = self.parser.parse('{"target": "A.B", "sources": "C.D"}')
        self.assertEqual(parsed.sources, ["c.d"])


class TestConfidenceClamping(unittest.TestCase):
    def test_percentage_is_rescaled(self):
        self.assertEqual(SQLDependencies(target="t", sources=[], confidence=90).confidence, 0.9)

    def test_out_of_range_is_clamped(self):
        self.assertEqual(SQLDependencies(target="t", sources=[], confidence=500).confidence, 1.0)
        self.assertEqual(SQLDependencies(target="t", sources=[], confidence=-1).confidence, 0.0)

    def test_garbage_confidence_defaults_to_one(self):
        self.assertEqual(
            SQLDependencies(target="t", sources=[], confidence="high").confidence, 1.0
        )

    def test_defaults_preserve_backwards_compatibility(self):
        deps = SQLDependencies(target="a.b", sources=["c.d"])
        self.assertEqual(deps.confidence, 1.0)
        self.assertEqual(deps.reasoning, "")
        self.assertIsNone(deps.parse_error)
        self.assertEqual(deps.to_lineage_result().target, "a.b")


class TestOrchestratorReportsParseFailures(unittest.TestCase):
    """D3: a bad response surfaces in the result, never as an exception."""

    def test_structured_response_is_exposed(self):
        orchestrator = PipelineOrchestrator(
            Config(llm_provider="mock"), llm=MockLLM([VALID_RESPONSE])
        )
        result = orchestrator.run("SELECT a FROM t")
        self.assertTrue(result.success)
        self.assertEqual(result.llm_structured["target"], "analytics.sales")
        self.assertEqual(result.llm_confidence, 0.9)
        self.assertIsNone(result.parse_error)

    def test_garbage_response_sets_parse_error(self):
        orchestrator = PipelineOrchestrator(
            Config(llm_provider="mock"), llm=MockLLM(["not json at all"])
        )
        result = orchestrator.run("SELECT a FROM t")
        self.assertTrue(result.success, "a parse failure must not fail the run")
        self.assertTrue(result.parse_error)
        self.assertLess(result.llm_confidence, 0.5)


class TestEdgeProvenance(unittest.TestCase):
    """D4: every edge carries confidence, provenance and a verified flag."""

    SQL = "INSERT INTO analytics.t (a) SELECT o.a FROM sales.orders o WHERE o.a > 1"

    def run_pipeline(self):
        pipeline = SQL2GraphPipeline(parser=SQL2GraphParser(dialect="postgres"))
        return pipeline.run(self.SQL, use_llm_verify=False, use_llm_enhance=False)

    def test_all_edges_have_confidence_and_provenance(self):
        result = self.run_pipeline()
        self.assertTrue(result["graph"]["links"])
        for link in result["graph"]["links"]:
            self.assertIn("confidence", link)
            self.assertIn("provenance", link)
            self.assertIn("verified", link)

    def test_deterministic_edges_are_verified(self):
        result = self.run_pipeline()
        for link in result["graph"]["links"]:
            self.assertEqual(link["provenance"], "deterministic")
            self.assertTrue(link["verified"])

    def test_llm_provenance_marks_edges_unverified(self):
        from Classes.sql2graph_classes import SQL2GraphBuilder

        builder = SQL2GraphBuilder()
        builder.build(
            {
                "output_columns": [
                    {
                        "alias": "a",
                        "expression": "o.a",
                        "dependencies": [{"table_alias": "o", "column": "a"}],
                    }
                ]
            }
        )
        updated = builder.apply_edge_provenance("llm_verified", 0.9)
        self.assertGreater(updated, 0)
        for _, _, data in builder.graph.edges(data=True):
            self.assertEqual(data["provenance"], "llm_verified")
            self.assertEqual(data["confidence"], 0.9)
            self.assertFalse(data["verified"])

    def test_validator_flags_an_edge_without_provenance(self):
        import networkx as nx

        from Classes.sql2graph_classes import SQL2GraphValidator

        graph = nx.MultiDiGraph()
        graph.add_node("a", node_type="source_column")
        graph.add_node("b", node_type="output_column")
        graph.add_edge("a", "b", edge_type="DERIVED_FROM")
        warnings = SQL2GraphValidator.validate_graph(graph)
        self.assertTrue(any("missing confidence" in w for w in warnings))
        self.assertTrue(any("missing provenance" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
