import hashlib
import json
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from Classes.lineage_graph_agent import LineageGraphAgent

CHUNK_RESULT = {
    "chunks": [
        {"id": "pprb_attr_val", "name": "pprb_attr_val", "chunk_type": "cte", "sql": "SELECT 1"},
        {"id": "d_agr_collat_dmcl_attr", "name": "d_agr_collat_dmcl_attr", "chunk_type": "target", "sql": "target"},
        {"id": "main", "name": "main", "chunk_type": "query", "sql": "SELECT 2 FROM pprb_attr_val"},
    ],
    "links": [
        {"source": "main", "target": "d_agr_collat_dmcl_attr", "link_type": "INSERT", "condition": ""},
    ],
    "statement_type": "insert",
    "target_table": "d_agr_collat_dmcl_attr",
    "warnings": ["Disconnected chunks: ['pprb_attr_val']"],
}

SIMPLIFY = {
    "parser_used": True,
    "statement_type": "insert",
    "target_table": "d_agr_collat_dmcl_attr",
    "ctes": [{"alias": "pprb_attr_val"}],
    "joins": [],
}


class TestLineageGraphAgent(unittest.TestCase):
    def test_compact_helpers(self):
        compact = LineageGraphAgent._compact_simplify(SIMPLIFY)
        self.assertEqual(compact["cte_aliases"], ["pprb_attr_val"])
        chunks = LineageGraphAgent._compact_chunks(CHUNK_RESULT["chunks"], sql_limit=10)
        self.assertTrue(all(len(c["sql_preview"]) <= 13 for c in chunks))

    def test_build_prompt_includes_warnings(self):
        agent = LineageGraphAgent(hf_token=None)
        prompt = agent._build_prompt(
            sql="INSERT INTO t SELECT 1",
            deterministic=CHUNK_RESULT,
            simplify=SIMPLIFY,
            ast_summary={"cte_names": ["pprb_attr_val"]},
        )
        self.assertIn("Disconnected chunks", prompt)
        self.assertIn("pprb_attr_val", prompt)

    def test_build_graph_requires_token(self):
        agent = LineageGraphAgent(hf_token=None)
        with self.assertRaises(ValueError):
            agent.build_graph(sql="SELECT 1", chunk_result=CHUNK_RESULT)

    def test_invoke_uses_agent_client_not_empty_chunk_parser(self):
        class _Adapter:
            def invoke_messages(self, messages):
                return '{"chunks":[],"links":[]}'

        agent = LineageGraphAgent(hf_token=None)
        agent.chat_adapter = _Adapter()
        text = agent._invoke_messages_text([])
        self.assertEqual(text, '{"chunks":[],"links":[]}')


class FakeAdapter:
    """Scripted stand-in for the HuggingFace chat adapter."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke_messages(self, messages):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response


#: A minimal correction that links the orphaned CTE into the main query. Links are
#: only kept when the payload also declares the chunks they reference.
LLM_PAYLOAD = json.dumps(
    {
        "chunks": CHUNK_RESULT["chunks"],
        "links": CHUNK_RESULT["links"]
        + [
            {
                "source": "main",
                "target": "pprb_attr_val",
                "link_type": "JOIN",
                "condition": "",
            }
        ],
    }
)


class TestCompactHelpers(unittest.TestCase):
    def test_compact_simplify_of_nothing_is_empty(self):
        self.assertEqual(LineageGraphAgent._compact_simplify(None), {})

    def test_compact_simplify_keeps_only_the_summary_fields(self):
        compact = LineageGraphAgent._compact_simplify(
            {
                "statement_type": "insert",
                "target_table": "t",
                "select": {"aliases": ["a", "b"], "raw": "ignored"},
                "subgraph_blocks": [{"id": "b1", "type": "cte", "name": "n", "sql": "ignored"}],
                "unrelated": "dropped",
            }
        )
        self.assertNotIn("unrelated", compact)
        self.assertEqual(compact["select_aliases"], ["a", "b"])
        self.assertEqual(compact["subgraph_blocks"], [{"id": "b1", "type": "cte", "name": "n"}])

    def test_compact_ast_of_nothing_is_empty(self):
        self.assertEqual(LineageGraphAgent._compact_ast(None), {})

    def test_compact_ast_replaces_column_refs_with_a_count(self):
        compact = LineageGraphAgent._compact_ast(
            {"tables": ["t"], "column_refs": ["t.a", "t.b", "t.c"]}
        )
        self.assertEqual(compact["tables"], ["t"])
        self.assertEqual(compact["column_ref_count"], 3)

    def test_compact_chunks_truncates_long_sql(self):
        compact = LineageGraphAgent._compact_chunks(
            [{"id": "c", "name": "c", "chunk_type": "cte", "sql": "x" * 500}], sql_limit=10
        )
        self.assertEqual(compact[0]["sql_preview"], "x" * 10 + "...")

    def test_compact_chunks_of_nothing_is_empty(self):
        self.assertEqual(LineageGraphAgent._compact_chunks([]), [])


class TestPromptAndStaticDelegates(unittest.TestCase):
    def test_prompt_repeats_the_previous_validation_error(self):
        agent = LineageGraphAgent(hf_token=None)
        prompt = agent._build_prompt(
            sql="SELECT 1",
            deterministic=CHUNK_RESULT,
            validation_error="link_type must be one of [...]",
        )
        self.assertIn("Validation error from previous attempt:", prompt)
        self.assertIn("link_type must be one of", prompt)

    def test_extract_json_recovers_an_object_from_surrounding_prose(self):
        self.assertEqual(
            LineageGraphAgent._extract_json('Here you go: {"chunks": []} — done'),
            {"chunks": []},
        )

    def test_auth_errors_are_recognised(self):
        self.assertTrue(LineageGraphAgent._is_auth_error(Exception("401 Unauthorized")))
        self.assertFalse(LineageGraphAgent._is_auth_error(Exception("read timeout")))


class TestInvokeMessagesText(unittest.TestCase):
    def test_an_adapter_without_invoke_messages_falls_back_to_invoke(self):
        class _InvokeOnly:
            def invoke(self, messages):
                return type("Response", (), {"content": "from-invoke"})()

        agent = LineageGraphAgent(hf_token=None)
        agent.chat_adapter = _InvokeOnly()
        self.assertEqual(agent._invoke_messages_text([]), "from-invoke")

    def test_the_raw_chat_model_is_used_when_there_is_no_adapter(self):
        class _ChatModel:
            def invoke(self, messages):
                return type("Response", (), {"content": "from-chat-model"})()

        agent = LineageGraphAgent(hf_token=None)
        agent.chat_model = _ChatModel()
        self.assertEqual(agent._invoke_messages_text([]), "from-chat-model")

    def test_without_any_client_the_agent_says_so(self):
        agent = LineageGraphAgent(hf_token=None)
        with self.assertRaises(ValueError):
            agent._invoke_messages_text([])


class TestConstruction(unittest.TestCase):
    def test_a_token_builds_a_chat_model_and_adapter(self):
        sentinel = object()
        with patch(
            "Classes.lineage_graph_agent.create_chat_model", return_value=sentinel
        ) as create:
            agent = LineageGraphAgent(hf_token="test-token", temperature=0.5, max_new_tokens=64)

        self.assertIs(agent.chat_model, sentinel)
        self.assertIs(agent.chat_adapter.wrapper, sentinel)
        self.assertEqual(create.call_args.kwargs["max_new_tokens"], 64)
        self.assertTrue(create.call_args.kwargs["do_sample"])

    def test_no_token_leaves_the_agent_without_a_client(self):
        agent = LineageGraphAgent(hf_token=None)
        self.assertIsNone(agent.chat_model)
        self.assertIsNone(agent.chat_adapter)


class TestBuildGraph(unittest.TestCase):
    """`build_graph` merges LLM corrections into the deterministic seed."""

    def _agent(self, responses):
        agent = LineageGraphAgent(hf_token=None)
        agent.chat_adapter = FakeAdapter(responses)
        return agent

    def test_llm_link_is_merged_into_the_deterministic_seed(self):
        agent = self._agent([LLM_PAYLOAD])

        result = agent.build_graph(sql="INSERT INTO t SELECT 1", chunk_result=CHUNK_RESULT)

        links = {(link["source"], link["target"]) for link in result["links"]}
        self.assertIn(("main", "pprb_attr_val"), links)
        self.assertIn(("main", "d_agr_collat_dmcl_attr"), links)
        self.assertEqual({chunk["id"] for chunk in result["chunks"]}, {c["id"] for c in CHUNK_RESULT["chunks"]})
        self.assertEqual(result["warnings"], [])

    def test_result_is_marked_as_llm_enriched(self):
        agent = self._agent([LLM_PAYLOAD])

        metadata = agent.build_graph(sql="SELECT 1", chunk_result=CHUNK_RESULT)["metadata"]

        self.assertEqual(metadata["seed_source"], "sqlglot+llm_graph_agent")
        self.assertEqual(metadata["pipeline_stage"], "llm_graph_agent")
        self.assertTrue(metadata["llm_enriched"])

    def test_a_transient_failure_is_retried(self):
        agent = self._agent([RuntimeError("connection reset"), LLM_PAYLOAD])

        with patch("Classes.lineage_graph_agent.time.sleep") as sleep:
            result = agent.build_graph(sql="SELECT 1", chunk_result=CHUNK_RESULT)

        self.assertEqual(agent.chat_adapter.calls, 2)
        sleep.assert_called_once()
        self.assertTrue(result["chunks"])

    def test_an_auth_failure_is_raised_without_retrying(self):
        agent = self._agent([Exception("401 Unauthorized")])

        with patch("Classes.lineage_graph_agent.time.sleep") as sleep:
            with self.assertRaises(Exception) as ctx:
                agent.build_graph(sql="SELECT 1", chunk_result=CHUNK_RESULT)

        self.assertIn("401", str(ctx.exception))
        self.assertEqual(agent.chat_adapter.calls, 1)
        sleep.assert_not_called()

    def test_persistent_failures_are_raised_after_the_retry_budget(self):
        agent = self._agent([RuntimeError("connection reset")])
        agent.max_retries = 2

        with patch("Classes.lineage_graph_agent.time.sleep"):
            with self.assertRaises(RuntimeError):
                agent.build_graph(sql="SELECT 1", chunk_result=CHUNK_RESULT)

        self.assertEqual(agent.chat_adapter.calls, 2)

    def test_an_invalid_graph_is_retried_with_the_error_in_the_prompt(self):
        bad = json.dumps(
            {"chunks": [{"id": "x", "name": "x", "chunk_type": "banana", "sql": "SELECT 1"}],
             "links": []}
        )
        agent = self._agent([bad, LLM_PAYLOAD])
        prompts = []
        original = agent._build_prompt

        def recording_build_prompt(**kwargs):
            prompts.append(kwargs)
            return original(**kwargs)

        agent._build_prompt = recording_build_prompt

        agent.build_graph(sql="SELECT 1", chunk_result=CHUNK_RESULT)

        self.assertIsNone(prompts[0]["validation_error"])
        self.assertIn("chunk_type must be one of", prompts[1]["validation_error"])

    def test_a_persistently_invalid_graph_surfaces_the_validation_error(self):
        bad = json.dumps(
            {"chunks": [{"id": "x", "name": "x", "chunk_type": "banana", "sql": "SELECT 1"}],
             "links": []}
        )
        agent = self._agent([bad])
        agent.max_retries = 1

        with self.assertRaises(ValidationError):
            agent.build_graph(sql="SELECT 1", chunk_result=CHUNK_RESULT)


class TestBuildGraphFromSnapshot(unittest.TestCase):
    def _agent(self):
        agent = LineageGraphAgent(hf_token=None)
        agent.chat_adapter = FakeAdapter([LLM_PAYLOAD])
        return agent

    def test_snapshot_fields_are_forwarded_to_build_graph(self):
        agent = self._agent()
        snapshot = {"chunks": CHUNK_RESULT, "simplify": SIMPLIFY, "sql": "INSERT INTO t SELECT 1"}

        result = agent.build_graph_from_snapshot(snapshot)

        self.assertEqual(result["target_table"], "d_agr_collat_dmcl_attr")
        self.assertTrue(result["metadata"]["llm_enriched"])

    def test_an_explicit_sql_argument_overrides_the_snapshot(self):
        override = "INSERT INTO other SELECT 2"
        snapshot = {"chunks": CHUNK_RESULT, "sql": "SELECT 1"}

        result = self._agent().build_graph_from_snapshot(snapshot, sql=override)

        self.assertEqual(
            result["metadata"]["source_sql_hash"],
            hashlib.sha256(override.encode("utf-8")).hexdigest(),
        )

    def test_a_snapshot_without_a_chunk_object_is_rejected(self):
        with self.assertRaises(ValueError):
            self._agent().build_graph_from_snapshot({"chunks": ["not", "a", "dict"]})

    def test_a_snapshot_referencing_a_file_must_carry_the_sql(self):
        with self.assertRaises(ValueError):
            self._agent().build_graph_from_snapshot(
                {"chunks": CHUNK_RESULT, "source_file": "queries.sql"}
            )


if __name__ == "__main__":
    unittest.main()
