import unittest

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


if __name__ == "__main__":
    unittest.main()
