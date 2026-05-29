import unittest
from unittest.mock import patch

from Classes.model_classes import SQLLineageExtractor, SQLLineageOutputParser


class RaisingChain:
    def __init__(self, error: Exception):
        self.error = error
        self.calls = 0

    def invoke(self, _sql):
        self.calls += 1
        raise self.error


class SequenceChain:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = 0

    def invoke(self, _sql):
        value = self.outputs[self.calls]
        self.calls += 1
        if isinstance(value, Exception):
            raise value
        return value


class TestSQLLineageOutputParser(unittest.TestCase):
    def test_parse_valid_json_with_string_source(self):
        parser = SQLLineageOutputParser()
        parsed = parser.parse('{"target": "Analytics.Sales", "sources": "Raw.Orders"}')
        self.assertEqual(parsed.target, "analytics.sales")
        self.assertEqual(parsed.sources, ["raw.orders"])

    def test_parse_regex_fallback(self):
        parser = SQLLineageOutputParser()
        text = "target: analytics.sales, sources: [raw.orders, raw.customers]"
        parsed = parser.parse(text)
        self.assertEqual(parsed.target, "analytics.sales")
        self.assertEqual(parsed.sources, ["raw.orders", "raw.customers"])

    def test_parse_unstructured_text_returns_empty(self):
        parser = SQLLineageOutputParser()
        parsed = parser.parse("no lineage data here")
        self.assertEqual(parsed.target, "")
        self.assertEqual(parsed.sources, [])


class TestSQLLineageExtractorLogic(unittest.TestCase):
    def _make_extractor_without_init(self):
        extractor = SQLLineageExtractor.__new__(SQLLineageExtractor)
        extractor.max_retries = 3
        extractor.provider = "scaleway"
        return extractor

    def test_auth_error_detection(self):
        self.assertTrue(SQLLineageExtractor._is_auth_error(Exception("401 Unauthorized")))
        self.assertTrue(SQLLineageExtractor._is_auth_error(Exception("bad credentials")))
        self.assertFalse(SQLLineageExtractor._is_auth_error(Exception("timeout reached")))

    def test_extract_returns_immediately_on_auth_error(self):
        extractor = self._make_extractor_without_init()
        extractor.chain = RaisingChain(Exception("401 Client Error: Unauthorized"))

        with patch("Classes.model_classes.time.sleep") as sleep_mock:
            result = extractor.extract("SELECT 1")

        self.assertIn("authentication failed", result["error"].lower())
        self.assertEqual(extractor.chain.calls, 1)
        sleep_mock.assert_not_called()

    def test_extract_retries_and_returns_max_retries_error(self):
        extractor = self._make_extractor_without_init()
        extractor.chain = RaisingChain(Exception("temporary network issue"))

        with patch("Classes.model_classes.time.sleep") as sleep_mock:
            result = extractor.extract("SELECT 1")

        self.assertEqual(result["error"], "Max retries exceeded")
        self.assertEqual(extractor.chain.calls, 3)
        self.assertEqual(sleep_mock.call_count, 3)

    def test_extract_returns_success_on_valid_result(self):
        extractor = self._make_extractor_without_init()
        extractor.chain = SequenceChain([{"target": "a.b", "sources": ["c.d"]}])

        result = extractor.extract("SELECT 1")
        self.assertEqual(result, {"target": "a.b", "sources": ["c.d"]})

    def test_extract_with_result_wraps_dict(self):
        extractor = self._make_extractor_without_init()
        extractor.extract = lambda _sql: {"target": "a.b", "sources": ["c.d"]}

        wrapped = extractor.extract_with_result("SELECT 1")
        self.assertEqual(wrapped.target, "a.b")
        self.assertEqual(wrapped.sources, ["c.d"])


if __name__ == "__main__":
    unittest.main()
