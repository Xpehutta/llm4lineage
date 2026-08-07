import unittest
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from Classes.model_classes import (
    SourceTableStructure,
    SQLLineageExtractor,
    SQLLineageOutputParser,
    ViewOutputColumn,
    ViewStructure,
    create_sql_lineage_extractor,
)


class FakeChatModel:
    """Stands in for the provider chat model created by ``create_chat_model``."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return type("Response", (), {"content": response})()


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

    def test_batch_extract_returns_one_entry_per_query(self):
        extractor = self._make_extractor_without_init()
        extractor.chain = SequenceChain(
            [{"target": "a.b", "sources": ["c.d"]}, {"target": "e.f", "sources": []}]
        )

        results = extractor.batch_extract(["SELECT 1", "SELECT 2"])
        self.assertEqual([item["target"] for item in results], ["a.b", "e.f"])

    def test_batch_extract_captures_a_failure_without_aborting_the_batch(self):
        extractor = self._make_extractor_without_init()
        queries = ["BAD" * 60, "SELECT 2"]

        def flaky(sql):
            if sql.startswith("BAD"):
                raise RuntimeError("boom")
            return {"target": "e.f", "sources": []}

        extractor.extract = flaky
        results = extractor.batch_extract(queries)

        self.assertEqual(results[0]["error"], "boom")
        self.assertEqual(results[0]["target"], "")
        self.assertEqual(results[0]["query"], queries[0][:100])
        self.assertEqual(results[1]["target"], "e.f")

    def test_test_connection_reflects_whether_extract_errored(self):
        extractor = self._make_extractor_without_init()

        extractor.extract = lambda _sql: {"target": "a.b", "sources": []}
        self.assertTrue(extractor.test_connection())

        extractor.extract = lambda _sql: {"error": "Max retries exceeded", "target": ""}
        self.assertFalse(extractor.test_connection())

    def test_test_connection_swallows_unexpected_errors(self):
        extractor = self._make_extractor_without_init()

        def explode(_sql):
            raise RuntimeError("socket closed")

        extractor.extract = explode
        self.assertFalse(extractor.test_connection())


class TestSQLLineageExtractorConstruction(unittest.TestCase):
    """``__init__`` wires prompt, parser and chain around the provider client."""

    def _build(self, responses, **kwargs):
        self.chat_model = FakeChatModel(responses)
        patcher = patch(
            "Classes.model_classes.create_chat_model", return_value=self.chat_model
        )
        self.create_chat_model = patcher.start()
        self.addCleanup(patcher.stop)
        return SQLLineageExtractor(hf_token="test-token", **kwargs)

    def test_missing_token_is_rejected(self):
        with patch("Classes.model_classes.resolve_hf_token", return_value=None):
            with self.assertRaises(ValueError) as ctx:
                SQLLineageExtractor(hf_token=None)
        self.assertIn("Hugging Face token not found", str(ctx.exception))

    def test_config_info_reports_the_resolved_settings(self):
        extractor = self._build(["{}"], model="org/model", provider="scaleway", max_retries=2)

        self.assertEqual(
            extractor.get_config_info(),
            {
                "model": "org/model",
                "provider": "scaleway",
                "max_new_tokens": 2048,
                "do_sample": False,
                "max_retries": 2,
                "use_pydantic_parser": True,
            },
        )

    def test_sampling_temperature_follows_do_sample(self):
        self._build(["{}"], do_sample=True)
        self.assertEqual(self.create_chat_model.call_args.kwargs["temperature"], 0.1)

        self._build(["{}"], do_sample=False)
        self.assertEqual(self.create_chat_model.call_args.kwargs["temperature"], 0.005)

    def test_default_human_prompt_template_asks_for_target_and_sources(self):
        extractor = self._build(["{}"])
        self.assertIn('"target"', extractor.human_prompt_template)
        self.assertIn('"sources"', extractor.human_prompt_template)

    def test_custom_human_prompt_template_is_kept_verbatim(self):
        extractor = self._build(["{}"], human_prompt_template="just do it")
        self.assertEqual(extractor.human_prompt_template, "just do it")

    def test_non_pydantic_parser_is_the_custom_parser(self):
        extractor = self._build(["{}"], use_pydantic_parser=False)
        self.assertIsInstance(extractor.output_parser, SQLLineageOutputParser)

    def test_chain_sends_system_and_human_messages_containing_the_sql(self):
        extractor = self._build(['{"target": "a.b", "sources": ["c.d"]}'])

        extractor.extract("SELECT * FROM raw.orders")

        system, human = self.chat_model.calls[0]
        self.assertEqual(type(system).__name__, "SystemMessage")
        self.assertEqual(type(human).__name__, "HumanMessage")
        self.assertIn("SQL lineage extraction expert", system.content)
        self.assertIn("SELECT * FROM raw.orders", human.content)

    def test_chain_escapes_braces_so_the_sql_is_not_read_as_a_placeholder(self):
        extractor = self._build(['{"target": "a.b", "sources": ["c.d"]}'])

        extractor.extract("SELECT x FROM t WHERE j = '{a}'")

        _system, human = self.chat_model.calls[0]
        self.assertIn("'{{a}}'", human.content)

    def test_pydantic_parser_returns_the_structured_response(self):
        extractor = self._build(['{"target": "Analytics.Sales", "sources": ["Raw.Orders"]}'])

        result = extractor.extract("SELECT 1")

        self.assertEqual(result["target"], "analytics.sales")
        self.assertEqual(result["sources"], ["raw.orders"])

    def test_pydantic_parser_falls_back_to_the_salvage_parser(self):
        """A response the schema parser rejects is still salvaged by regex."""
        extractor = self._build(["target: analytics.sales, sources: [raw.orders]"])

        result = extractor.extract("SELECT 1")

        self.assertEqual(result["target"], "analytics.sales")
        self.assertEqual(result["sources"], ["raw.orders"])
        self.assertEqual(result["provenance"], "regex")

    def test_custom_parser_path_produces_the_same_lineage(self):
        extractor = self._build(
            ['{"target": "Analytics.Sales", "sources": ["Raw.Orders"]}'],
            use_pydantic_parser=False,
        )

        result = extractor.extract("SELECT 1")

        self.assertEqual(result["target"], "analytics.sales")
        self.assertEqual(result["provenance"], "json")

    def test_factory_forwards_arguments_to_the_extractor(self):
        chat_model = FakeChatModel(["{}"])
        with patch("Classes.model_classes.create_chat_model", return_value=chat_model):
            extractor = create_sql_lineage_extractor(
                model="org/model",
                provider="nebius",
                hf_token="test-token",
                max_retries=7,
            )

        self.assertIsInstance(extractor, SQLLineageExtractor)
        self.assertEqual(extractor.model, "org/model")
        self.assertEqual(extractor.provider, "nebius")
        self.assertEqual(extractor.max_retries, 7)


class TestParserEdgeCases(unittest.TestCase):
    def test_empty_response_is_reported_as_a_parse_error(self):
        parsed = SQLLineageOutputParser().parse("   ")
        self.assertEqual(parsed.provenance, "none")
        self.assertEqual(parsed.confidence, 0.0)
        self.assertEqual(parsed.parse_error, "LLM returned an empty response")

    def test_non_list_sources_are_discarded(self):
        parsed = SQLLineageOutputParser().parse('{"target": "a.b", "sources": 42}')
        self.assertEqual(parsed.target, "a.b")
        self.assertEqual(parsed.sources, [])

    def test_malformed_json_is_salvaged_and_the_json_error_is_kept(self):
        parsed = SQLLineageOutputParser().parse('{"target": "a.b", "sources": ["c.d",}')
        self.assertEqual(parsed.provenance, "regex")
        self.assertEqual(parsed.confidence, 0.3)
        self.assertIn("not valid JSON", parsed.parse_error)

    def test_unrecoverable_response_explains_why(self):
        parsed = SQLLineageOutputParser().parse("no lineage data here")
        self.assertEqual(parsed.provenance, "none")
        self.assertIn("no target/sources could be recovered", parsed.parse_error)

    def test_reasoning_and_confidence_are_carried_through(self):
        parsed = SQLLineageOutputParser().parse(
            '{"target": "a.b", "sources": [], "reasoning": "single table", "confidence": 0.5}'
        )
        self.assertEqual(parsed.reasoning, "single table")
        self.assertEqual(parsed.confidence, 0.5)

    def test_parser_type_label(self):
        self.assertEqual(SQLLineageOutputParser()._type, "sql_lineage_parser")


class TestViewModels(unittest.TestCase):
    def test_output_column_name_is_trimmed(self):
        self.assertEqual(ViewOutputColumn(name="  total  ").name, "total")

    def test_blank_output_column_name_is_rejected(self):
        with pytest.raises(ValidationError):
            ViewOutputColumn(name="   ")

    def test_view_name_is_trimmed(self):
        self.assertEqual(ViewStructure(view_name="  v_sales ").view_name, "v_sales")

    def test_blank_view_name_is_rejected(self):
        with pytest.raises(ValidationError):
            ViewStructure(view_name="")

    def test_source_table_structure_accepts_the_schema_alias(self):
        source = SourceTableStructure(full_name="raw.orders", schema="raw", table="orders")
        self.assertEqual(source.schema_name, "raw")

    def test_source_table_structure_accepts_the_field_name(self):
        source = SourceTableStructure(full_name="raw.orders", schema_name="raw")
        self.assertEqual(source.schema_name, "raw")


if __name__ == "__main__":
    unittest.main()
