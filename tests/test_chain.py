"""Tests for SQLAnalysisChain."""

import unittest
from unittest.mock import MagicMock

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from Classes.pipeline.core.chain import SQLAnalysisChain, _DEFAULT_HUMAN_TEMPLATE
from Classes.pipeline.models.config import Config


class TestSQLAnalysisChain(unittest.TestCase):
    def test_run_with_mock_llm(self):
        config = Config(llm_provider="mock")
        llm = GenericFakeChatModel(messages=iter(["Analysis complete."]))
        chain = SQLAnalysisChain(config, llm)
        response = chain.run(
            ast_json={"type": "Select"},
            column_lineage=[{"target_column": "a"}],
            instruction="Explain.",
        )
        self.assertEqual(response, "Analysis complete.")

    def test_prompt_placeholders_in_template(self):
        self.assertIn("{instruction}", _DEFAULT_HUMAN_TEMPLATE)
        self.assertIn("{ast_json}", _DEFAULT_HUMAN_TEMPLATE)
        self.assertIn("{column_lineage}", _DEFAULT_HUMAN_TEMPLATE)

    def test_missing_prompt_file_falls_back(self):
        config = Config(
            prompt_system_file="/nonexistent/system.txt",
            prompt_human_template_file="/nonexistent/human.txt",
        )
        llm = GenericFakeChatModel(messages=iter(["ok"]))
        chain = SQLAnalysisChain(config, llm)
        response = chain.run({}, [], "")
        self.assertEqual(response, "ok")

    def test_retry_on_transient_failure(self):
        config = Config(llm_provider="mock", llm_retry_attempts=3)
        llm = GenericFakeChatModel(messages=iter([]))
        chain = SQLAnalysisChain(config, llm)

        call_count = 0

        def flaky_invoke(_payload):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("transient")
            return "recovered"

        chain.chain = MagicMock()
        chain.chain.invoke = flaky_invoke

        response = chain.run({"type": "Select"}, [], "test")
        self.assertEqual(response, "recovered")
        self.assertEqual(call_count, 2)


if __name__ == "__main__":
    unittest.main()
