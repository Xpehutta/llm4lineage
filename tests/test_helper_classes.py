import json
import unittest
from unittest.mock import patch

from Classes.helper_classes import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PROVIDER,
    HuggingFaceLLMAdapter,
    SQLDependencies,
    SQLLineageResult,
    resolve_model_name,
    resolve_provider,
)


class DummyWrapper:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        return f"ok:{prompt}"


class DummyInvokeWrapper:
    def __init__(self):
        self.calls = []

    class _Resp:
        def __init__(self, content: str):
            self.content = content

    def invoke(self, payload):
        self.calls.append(payload)
        return self._Resp(f"invoked:{payload}")


class TestSQLLineageResult(unittest.TestCase):
    def test_result_helpers(self):
        result = SQLLineageResult(target="analytics.sales", sources=["raw.orders"])
        result.add_source("raw.customers")
        result.add_source("raw.orders")  # duplicate should be ignored
        result.add_source("")  # empty should be ignored

        self.assertEqual(result.source_count, 2)
        self.assertEqual(
            result.to_dict(),
            {"target": "analytics.sales", "sources": ["raw.orders", "raw.customers"]},
        )
        self.assertIn("analytics.sales", str(result))
        self.assertEqual(json.loads(result.to_json())["target"], "analytics.sales")


class TestSQLDependencies(unittest.TestCase):
    def test_normalization_and_conversion(self):
        deps = SQLDependencies(
            target='"Analytics"."Sales"',
            sources=["'Raw.Orders'", '"Raw.Customers"'],
        )
        self.assertEqual(deps.target, "analytics.sales")
        self.assertEqual(deps.sources, ["raw.orders", "raw.customers"])

        lineage_result = deps.to_lineage_result()
        self.assertIsInstance(lineage_result, SQLLineageResult)
        self.assertEqual(lineage_result.target, "analytics.sales")


class TestResolveModelName(unittest.TestCase):
    def test_explicit_argument_wins(self):
        with patch.dict("os.environ", {"MODEL_NAME": "env/model"}):
            self.assertEqual(resolve_model_name("explicit/model"), "explicit/model")

    def test_env_var_used_when_no_argument(self):
        with patch.dict("os.environ", {"MODEL_NAME": "env/model"}):
            self.assertEqual(resolve_model_name(), "env/model")
            self.assertEqual(resolve_model_name(None), "env/model")

    def test_default_when_nothing_set(self):
        with patch.dict("os.environ", {}, clear=False):
            import os
            env = {k: v for k, v in os.environ.items() if k != "MODEL_NAME"}
            with patch.dict("os.environ", env, clear=True):
                self.assertEqual(resolve_model_name(), DEFAULT_MODEL_NAME)
                self.assertEqual(resolve_model_name(default="other/model"), "other/model")


class TestResolveProvider(unittest.TestCase):
    def test_explicit_argument_wins(self):
        with patch.dict("os.environ", {"PROVIDER": "env-provider"}):
            self.assertEqual(resolve_provider("explicit-provider"), "explicit-provider")

    def test_env_var_used_when_no_argument(self):
        with patch.dict("os.environ", {"PROVIDER": "env-provider"}):
            self.assertEqual(resolve_provider(), "env-provider")
            self.assertEqual(resolve_provider(None), "env-provider")

    def test_default_when_nothing_set(self):
        with patch.dict("os.environ", {}, clear=False):
            import os
            env = {k: v for k, v in os.environ.items() if k != "PROVIDER"}
            with patch.dict("os.environ", env, clear=True):
                self.assertEqual(resolve_provider(), DEFAULT_PROVIDER)
                self.assertEqual(resolve_provider(default="other-provider"), "other-provider")


class TestHuggingFaceLLMAdapter(unittest.TestCase):
    def test_adapter_delegates_to_wrapper(self):
        wrapper = DummyWrapper()
        adapter = HuggingFaceLLMAdapter(wrapper)

        response = adapter.invoke("hello")
        self.assertEqual(response, "ok:hello")
        self.assertEqual(adapter("world"), "ok:world")
        self.assertEqual(wrapper.calls, ["hello", "world"])

    def test_adapter_supports_message_invocation(self):
        wrapper = DummyInvokeWrapper()
        adapter = HuggingFaceLLMAdapter(wrapper)
        out = adapter.invoke_messages(["m1", "m2"])
        self.assertEqual(out, "invoked:['m1', 'm2']")
        self.assertEqual(wrapper.calls, [["m1", "m2"]])


if __name__ == "__main__":
    unittest.main()
