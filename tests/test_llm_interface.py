"""Tests for the provider-neutral LLM boundary (Phase C5)."""

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from Classes.pipeline.core.llm_interface import (
    ChatMessage,
    LangChainChatAdapter,
    LLMInterface,
    MockLLM,
    adapt_llm,
    message_text,
    render_template,
)
from Classes.pipeline.core.orchestrator import PipelineOrchestrator
from Classes.pipeline.models.config import Config

ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = ROOT / "Classes" / "pipeline" / "core"
#: The only core modules allowed to mention LangChain.
LANGCHAIN_ALLOWED = {"llm_factory.py", "llm_interface.py"}


class FakeChatModel:
    """Stands in for a LangChain chat model without importing LangChain."""

    model_name = "fake-model"

    def __init__(self, response="ok"):
        self.response = response
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return type("Response", (), {"content": self.response})()


class TestLangchainIsolation(unittest.TestCase):
    def test_core_modules_do_not_import_langchain(self):
        offenders = []
        for path in CORE_DIR.glob("*.py"):
            if path.name in LANGCHAIN_ALLOWED:
                continue
            if "langchain" in path.read_text(encoding="utf-8"):
                offenders.append(path.name)
        self.assertEqual(
            offenders,
            [],
            f"LangChain must stay behind LLMInterface; found it in {offenders}",
        )

    def test_langchain_imports_are_lazy(self):
        """Importing the pipeline core must not pull LangChain into sys.modules."""
        script = textwrap.dedent(
            """
            import sys
            import Classes.pipeline.core.chain
            import Classes.pipeline.core.orchestrator
            import Classes.pipeline.core.llm_factory
            leaked = [m for m in sys.modules if m.startswith("langchain")]
            print(",".join(sorted(leaked)))
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(completed.stdout.strip(), "", "LangChain was imported eagerly")

    def test_mock_provider_runs_without_langchain(self):
        """A core-only install must still complete a pipeline run."""
        script = textwrap.dedent(
            """
            import builtins
            real_import = builtins.__import__

            def blocked(name, *args, **kwargs):
                if name.startswith("langchain"):
                    raise ImportError(f"{name} is not installed")
                return real_import(name, *args, **kwargs)

            builtins.__import__ = blocked

            from Classes.pipeline.core.orchestrator import PipelineOrchestrator
            from Classes.pipeline.models.config import Config

            result = PipelineOrchestrator(Config(llm_provider="mock")).run("SELECT a FROM t")
            assert result.success, result.error
            assert result.llm_response
            print("OK")
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode, 0, f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
        self.assertIn("OK", completed.stdout)


class TestAdaptLlm(unittest.TestCase):
    def test_passes_through_an_llm_interface(self):
        mock = MockLLM()
        self.assertIs(adapt_llm(mock), mock)

    def test_wraps_a_chat_model(self):
        adapted = adapt_llm(FakeChatModel())
        self.assertIsInstance(adapted, LangChainChatAdapter)
        self.assertEqual(adapted.invoke("hi"), "ok")

    def test_rejects_none(self):
        with self.assertRaises(TypeError):
            adapt_llm(None)

    def test_rejects_an_object_without_invoke(self):
        with self.assertRaises(TypeError):
            adapt_llm(object())

    def test_mock_satisfies_the_protocol(self):
        self.assertIsInstance(MockLLM(), LLMInterface)


class TestMockLlm(unittest.TestCase):
    def test_default_response(self):
        self.assertTrue(MockLLM().invoke("anything"))

    def test_scripted_responses_in_order(self):
        llm = MockLLM(["first", "second"])
        self.assertEqual(llm.invoke("a"), "first")
        self.assertEqual(llm.invoke("b"), "second")

    def test_last_response_repeats(self):
        llm = MockLLM(["only"])
        llm.invoke("a")
        self.assertEqual(llm.invoke("b"), "only")

    def test_empty_response_list(self):
        self.assertEqual(MockLLM([]).invoke("a"), "")


class TestModelLabel(unittest.TestCase):
    def test_adapter_reports_the_underlying_model(self):
        self.assertEqual(LangChainChatAdapter(FakeChatModel()).model_label, "fake-model")

    def test_adapter_falls_back_to_a_nested_model(self):
        inner = type("Inner", (), {"repo_id": "org/model"})()
        outer = type("Outer", (), {"llm": inner})()
        self.assertEqual(LangChainChatAdapter(outer).model_label, "org/model")

    def test_orchestrator_records_the_label(self):
        orchestrator = PipelineOrchestrator(Config(llm_provider="mock"))
        self.assertEqual(orchestrator.run("SELECT a FROM t").model_used, "mock")


class TestMessageText(unittest.TestCase):
    def test_plain_string(self):
        self.assertEqual(message_text("hello"), "hello")

    def test_object_with_content(self):
        self.assertEqual(message_text(type("R", (), {"content": "hi"})()), "hi")

    def test_list_of_parts(self):
        response = type("R", (), {"content": [{"text": "a"}, {"text": "b"}]})()
        self.assertEqual(message_text(response), "ab")


class TestRenderTemplate(unittest.TestCase):
    def test_substitutes_placeholders(self):
        self.assertEqual(render_template("a={a} b={b}", {"a": 1, "b": 2}), "a=1 b=2")

    def test_leaves_json_braces_untouched(self):
        template = 'Return {"kind": "object"} for {name}'
        self.assertEqual(
            render_template(template, {"name": "x"}),
            'Return {"kind": "object"} for x',
        )

    def test_unknown_placeholder_is_left_alone(self):
        self.assertEqual(render_template("{a} {b}", {"a": "1"}), "1 {b}")


class TestChatMessage(unittest.TestCase):
    def test_adapter_maps_roles(self):
        model = FakeChatModel()
        LangChainChatAdapter(model).invoke_messages(
            [ChatMessage("system", "sys"), ChatMessage("user", "usr")]
        )
        sent = model.calls[0]
        self.assertEqual([type(m).__name__ for m in sent], ["SystemMessage", "HumanMessage"])
        self.assertEqual([m.content for m in sent], ["sys", "usr"])


if __name__ == "__main__":
    unittest.main()
