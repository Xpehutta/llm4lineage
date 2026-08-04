import tempfile
import unittest
from pathlib import Path

from Classes.llm_cache import LLMCache


class TestLLMCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = LLMCache(path=str(Path(self.tmp.name) / "cache.sqlite"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_set_if_better_replaces_lower_quality(self):
        key = "test-key"
        self.cache.set(key, {"value": 1}, quality_score=10.0)
        result = self.cache.set_if_better(key, {"value": 2}, quality_score=20.0)
        self.assertTrue(result["updated"])
        entry = self.cache.get_entry(key)
        self.assertEqual(entry["payload"]["value"], 2)
        self.assertEqual(entry["quality_score"], 20.0)

    def test_set_if_better_keeps_higher_quality(self):
        key = "test-key"
        self.cache.set(key, {"value": 1}, quality_score=30.0)
        result = self.cache.set_if_better(key, {"value": 2}, quality_score=15.0)
        self.assertFalse(result["updated"])
        entry = self.cache.get_entry(key)
        self.assertEqual(entry["payload"]["value"], 1)
        self.assertEqual(result["previous_quality_score"], 30.0)

    def test_make_pipeline_key_differs_by_flags(self):
        key_a = LLMCache.make_pipeline_key(
            "SELECT 1",
            prompt_version="v2.1",
            model="model-a",
            dialect="postgres",
            use_llm_verify=True,
            use_llm_enhance=False,
        )
        key_b = LLMCache.make_pipeline_key(
            "SELECT 1",
            prompt_version="v2.1",
            model="model-a",
            dialect="postgres",
            use_llm_verify=True,
            use_llm_enhance=True,
        )
        self.assertNotEqual(key_a, key_b)


if __name__ == "__main__":
    unittest.main()
