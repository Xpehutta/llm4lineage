"""Phase G — Resolver / Reviewer / Doc agents + orchestrator (MockLLM)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Classes.agents import (
    AgentOrchestrator,
    CandidateEdge,
    DocAgent,
    ResolverAgent,
    ReviewerAgent,
    find_sql_evidence,
)
from Classes.agents._json_util import (
    DEFAULT_TOKEN_BUDGET,
    estimate_tokens,
    truncate_to_token_budget,
)
from Classes.llm_cache import LLMCache
from Classes.pipeline.core.llm_interface import MockLLM

FUNCTION_SQL = """
CREATE FUNCTION enrich_customers() RETURNS void AS $$
BEGIN
  INSERT INTO analytics.customers (id, email, full_name)
  SELECT c.id, c.email, c.full_name
  FROM raw.customers c;
END;
$$ LANGUAGE plpgsql;
"""

GOOD_EDGE = {
    "src": "raw.customers.email",
    "dst": "analytics.customers.email",
    "transform_type": "direct",
    "confidence": 0.85,
    "reasoning": "INSERT selects email from raw.customers",
    "sql_fragment": "SELECT c.id, c.email, c.full_name FROM raw.customers c",
}

BAD_EDGE = {
    "src": "imaginary.source.col",
    "dst": "ghost.target.col",
    "transform_type": "direct",
    "confidence": 0.9,
    "reasoning": "hallucinated",
    "sql_fragment": "",
}


class TestTokenBudget(unittest.TestCase):
    def test_estimate_is_chars_over_four(self):
        self.assertEqual(estimate_tokens("abcd"), 1)
        self.assertEqual(estimate_tokens("a" * 400), 100)

    def test_truncate_respects_30k_budget(self):
        huge = "x" * (DEFAULT_TOKEN_BUDGET * 4 + 500)
        clipped = truncate_to_token_budget(huge, DEFAULT_TOKEN_BUDGET)
        self.assertLessEqual(estimate_tokens(clipped), DEFAULT_TOKEN_BUDGET)
        self.assertEqual(len(clipped), DEFAULT_TOKEN_BUDGET * 4)


class TestResolverAgent(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = LLMCache(path=str(Path(self.tmp.name) / "cache.sqlite"))
        payload = json.dumps({"edges": [GOOD_EDGE]})
        self.llm = MockLLM([payload])
        self.agent = ResolverAgent(self.llm, cache=self.cache, model_label="mock")

    def tearDown(self):
        self.tmp.cleanup()

    def test_resolve_returns_candidate_edges(self):
        edges = self.agent.resolve(
            FUNCTION_SQL,
            [{"sql_fragment": "EXECUTE format(...)", "reason": "dynamic_execute"}],
        )
        self.assertEqual(len(edges), 1)
        edge = edges[0]
        self.assertEqual(edge.src, GOOD_EDGE["src"])
        self.assertEqual(edge.dst, GOOD_EDGE["dst"])
        self.assertEqual(edge.transform_type, "direct")
        self.assertAlmostEqual(edge.confidence, 0.85)
        self.assertTrue(edge.reasoning)
        self.assertFalse(edge.verified)

    def test_empty_unresolved_short_circuits(self):
        self.assertEqual(self.agent.resolve(FUNCTION_SQL, []), [])

    def test_malformed_llm_response_yields_no_edges(self):
        agent = ResolverAgent(MockLLM(["not json at all"]))
        edges = agent.resolve(
            FUNCTION_SQL,
            [{"sql_fragment": "x", "reason": "parse_failed"}],
        )
        self.assertEqual(edges, [])

    def test_cache_avoids_second_llm_call(self):
        unresolved = [{"sql_fragment": "frag", "reason": "dynamic_execute"}]
        first = self.agent.resolve(FUNCTION_SQL, unresolved)
        # Exhaust / change the mock so a second call would differ if uncached.
        self.llm._responses = [json.dumps({"edges": [BAD_EDGE]})]
        self.llm._index = 0
        second = self.agent.resolve(FUNCTION_SQL, unresolved)
        self.assertEqual(first[0].src, second[0].src)
        self.assertEqual(second[0].src, GOOD_EDGE["src"])

    def test_prompt_stays_within_token_budget(self):
        huge_sql = "SELECT 1;\n" * 50_000
        calls: list[str] = []

        class SpyLLM(MockLLM):
            def invoke(self, prompt: str) -> str:
                calls.append(prompt)
                return super().invoke(prompt)

        spy_agent = ResolverAgent(SpyLLM(["{}"]), token_budget=500)
        spy_agent.resolve(huge_sql, [{"sql_fragment": "x", "reason": "r"}])
        self.assertEqual(len(calls), 1)
        self.assertLessEqual(estimate_tokens(calls[0]), 500)


class TestReviewerAgent(unittest.TestCase):
    def setUp(self):
        self.reviewer = ReviewerAgent(llm=MockLLM([]), use_llm=False)

    def test_pass_when_src_and_dst_appear_in_sql(self):
        result = self.reviewer.review(GOOD_EDGE, FUNCTION_SQL)
        self.assertEqual(result.verdict, "PASS")
        self.assertTrue(result.verified)
        self.assertTrue(result.sql_fragment)

    def test_fail_when_evidence_missing(self):
        result = self.reviewer.review(BAD_EDGE, FUNCTION_SQL)
        self.assertEqual(result.verdict, "FAIL")
        self.assertFalse(result.verified)
        self.assertIn("No code evidence", result.reason)

    def test_fail_when_only_src_present(self):
        edge = {
            "src": "raw.customers.email",
            "dst": "missing.nowhere.col",
            "transform_type": "direct",
            "confidence": 0.7,
            "reasoning": "partial",
        }
        result = self.reviewer.review(edge, FUNCTION_SQL)
        self.assertEqual(result.verdict, "FAIL")
        self.assertFalse(result.verified)

    def test_publishable_sets_verified_true(self):
        published = self.reviewer.publishable(GOOD_EDGE, FUNCTION_SQL)
        self.assertIsNotNone(published)
        assert published is not None
        self.assertTrue(published.verified)
        self.assertEqual(published.provenance, "llm_verified")

    def test_publishable_returns_none_on_fail(self):
        self.assertIsNone(self.reviewer.publishable(BAD_EDGE, FUNCTION_SQL))

    def test_find_sql_evidence_matches_qualified_and_bare(self):
        self.assertIsNotNone(find_sql_evidence(FUNCTION_SQL, "raw.customers.email"))
        self.assertIsNotNone(find_sql_evidence(FUNCTION_SQL, "email"))
        self.assertIsNone(find_sql_evidence(FUNCTION_SQL, "does_not_exist_xyz"))


class TestDocAgent(unittest.TestCase):
    def test_structured_labels_include_pii_owner_description(self):
        response = json.dumps(
            {
                "columns": [
                    {
                        "column": "email",
                        "is_pii": True,
                        "owner": "crm-team",
                        "description": "Customer email address",
                        "tags": ["contact"],
                    }
                ],
                "owner": "crm-team",
                "description": "Customer PII fields",
            }
        )
        agent = DocAgent(MockLLM([response]))
        labels = agent.label(
            "email — personal contact of the customer",
            column_metadata=[{"name": "email", "type": "text"}],
        )
        self.assertEqual(len(labels.columns), 1)
        col = labels.columns[0]
        self.assertEqual(col.column, "email")
        self.assertTrue(col.is_pii)
        self.assertEqual(col.owner, "crm-team")
        self.assertIn("email", col.description.lower())

    def test_apply_to_columns_sets_is_pii(self):
        response = json.dumps(
            {
                "columns": [
                    {
                        "column": "email",
                        "is_pii": True,
                        "owner": "dpo",
                        "description": "PII",
                    }
                ]
            }
        )
        agent = DocAgent(MockLLM([response]))
        labels = agent.label("email is PII")
        columns = agent.apply_to_columns([{"name": "email"}, {"name": "id"}], labels)
        by_name = {c["name"]: c for c in columns}
        self.assertTrue(by_name["email"]["is_pii"])
        self.assertEqual(by_name["email"]["owner"], "dpo")
        self.assertFalse(by_name["id"].get("is_pii", False))

    def test_garbage_response_returns_empty_labels(self):
        labels = DocAgent(MockLLM(["???"])).label("whatever")
        self.assertEqual(labels.columns, [])


class TestAgentOrchestrator(unittest.TestCase):
    def _make_orch(self, resolver_responses: list[str], max_attempts: int = 2):
        resolver = ResolverAgent(MockLLM(resolver_responses), model_label="mock")
        reviewer = ReviewerAgent(use_llm=False)
        return AgentOrchestrator(resolver, reviewer, max_attempts=max_attempts)

    def test_publishes_only_verified_edges(self):
        orch = self._make_orch([json.dumps({"edges": [GOOD_EDGE, BAD_EDGE]})])
        result = orch.run(
            FUNCTION_SQL,
            [{"sql_fragment": "INSERT ...", "reason": "dynamic_execute"}],
        )
        self.assertEqual(len(result.published_edges), 1)
        self.assertTrue(all(e.verified for e in result.published_edges))
        self.assertEqual(result.published_edges[0].src, GOOD_EDGE["src"])
        self.assertEqual(result.coverage.resolved, 1)
        self.assertEqual(result.coverage.escalated, 0)

    def test_escalates_after_n_attempts_queue_does_not_grow_forever(self):
        # Always propose an unverifiable edge → Reviewer FAIL → escalate.
        orch = self._make_orch(
            [json.dumps({"edges": [BAD_EDGE]})],
            max_attempts=2,
        )
        result = orch.run(
            FUNCTION_SQL,
            [
                {"sql_fragment": "EXECUTE format('%s', v)", "reason": "dynamic_execute"},
                {"sql_fragment": "CALL mystery()", "reason": "unsupported_statement"},
            ],
        )
        self.assertEqual(result.published_edges, [])
        self.assertEqual(result.coverage.escalated, 2)
        self.assertEqual(result.coverage.resolved, 0)
        self.assertEqual(len(result.escalated), 2)
        # Attempts are bounded by max_attempts per item.
        for count in result.coverage.attempts.values():
            self.assertLessEqual(count, 2)
        # Queue drained — nothing left unresolved in the run result beyond escalations.
        self.assertEqual(
            result.coverage.resolved + result.coverage.escalated,
            result.coverage.total_unresolved,
        )

    def test_coverage_report_fields(self):
        orch = self._make_orch([json.dumps({"edges": [GOOD_EDGE]})])
        result = orch.run(
            FUNCTION_SQL,
            [{"sql_fragment": "x", "reason": "dynamic_execute"}],
        )
        cov = result.coverage
        self.assertEqual(cov.total_unresolved, 1)
        self.assertEqual(cov.published_edges, 1)
        self.assertGreater(cov.coverage_ratio, 0.0)

    def test_never_publishes_unverified(self):
        """Defence: even a sneaky CandidateEdge with verified=False is gated."""
        class SoftReviewer(ReviewerAgent):
            def publishable(self, edge, source_sql):
                # Intentionally return unverified — orchestrator must drop it.
                c = (
                    edge
                    if isinstance(edge, CandidateEdge)
                    else CandidateEdge.model_validate(edge)
                )
                return c.model_copy(update={"verified": False})

        resolver = ResolverAgent(
            MockLLM([json.dumps({"edges": [GOOD_EDGE]})]),
            model_label="mock",
        )
        orch = AgentOrchestrator(resolver, SoftReviewer(use_llm=False), max_attempts=1)
        result = orch.run(
            FUNCTION_SQL,
            [{"sql_fragment": "x", "reason": "dynamic_execute"}],
        )
        self.assertEqual(result.published_edges, [])


class TestNoLangchainImports(unittest.TestCase):
    def test_agents_package_has_no_langchain_imports(self):
        root = Path(__file__).resolve().parent.parent / "Classes" / "agents"
        offenders: list[str] = []
        for path in root.rglob("*.py"):
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                if "import langchain" in line or "from langchain" in line:
                    offenders.append(f"{path.name}:{lineno}:{stripped}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
