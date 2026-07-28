"""Tests for LLMFactory."""

import unittest

from Classes.pipeline.core.llm_factory import LLMFactory
from Classes.pipeline.models.config import Config


class TestLLMFactory(unittest.TestCase):
    def test_mock_provider(self):
        config = Config(llm_provider="mock")
        llm = LLMFactory.create(config)
        self.assertIsNotNone(llm)

    def test_unknown_provider_raises(self):
        config = Config(llm_provider="nonexistent_provider")
        with self.assertRaises(ValueError):
            LLMFactory.create(config)

    def test_provider_case_insensitive(self):
        config = Config(llm_provider="  MOCK  ")
        llm = LLMFactory.create(config)
        self.assertIsNotNone(llm)

    def test_secret_str_unwrapping_mock(self):
        config = Config(
            llm_provider="mock",
            openai_api_key="secret-key-value",
        )
        llm = LLMFactory.create(config)
        self.assertIsNotNone(llm)


if __name__ == "__main__":
    unittest.main()
