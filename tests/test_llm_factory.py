"""Tests for LLMFactory."""

import sys
import types
import unittest
from contextlib import contextmanager

from Classes.pipeline.core.llm_factory import LLMFactory
from Classes.pipeline.models.config import Config


class Recorder:
    """Captures the keyword arguments a provider client was constructed with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@contextmanager
def fake_provider_module(name: str, **attributes):
    """Install a stand-in for a LangChain provider package for the duration of a test."""
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    original = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield
    finally:
        if original is None:
            del sys.modules[name]
        else:
            sys.modules[name] = original


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


class TestEffectiveTemperature(unittest.TestCase):
    def test_json_mode_caps_the_temperature(self):
        config = Config(llm_provider="mock", llm_temperature=0.9, llm_json_mode=True)
        self.assertEqual(LLMFactory.effective_temperature(config), 0.1)

    def test_json_mode_leaves_a_lower_temperature_alone(self):
        config = Config(llm_provider="mock", llm_temperature=0.0, llm_json_mode=True)
        self.assertEqual(LLMFactory.effective_temperature(config), 0.0)

    def test_without_json_mode_the_configured_temperature_is_used(self):
        config = Config(llm_provider="mock", llm_temperature=0.9, llm_json_mode=False)
        self.assertEqual(LLMFactory.effective_temperature(config), 0.9)


class TestProviderWiring(unittest.TestCase):
    """Each provider branch must map Config onto its client's arguments."""

    def test_openai_requests_a_json_object_response_format(self):
        config = Config(
            llm_provider="openai",
            openai_api_key="sk-test",
            openai_model="gpt-4o-mini",
            llm_max_tokens=512,
            llm_temperature=0.9,
        )
        with fake_provider_module("langchain_openai", ChatOpenAI=Recorder):
            client = LLMFactory.create_chat_model(config)

        self.assertEqual(client.kwargs["api_key"], "sk-test")
        self.assertEqual(client.kwargs["model"], "gpt-4o-mini")
        self.assertEqual(client.kwargs["max_tokens"], 512)
        self.assertEqual(client.kwargs["temperature"], 0.1)
        self.assertEqual(
            client.kwargs["model_kwargs"],
            {"response_format": {"type": "json_object"}},
        )

    def test_openai_without_json_mode_sends_no_response_format(self):
        config = Config(
            llm_provider="openai",
            openai_api_key="",
            llm_json_mode=False,
            llm_temperature=0.7,
        )
        with fake_provider_module("langchain_openai", ChatOpenAI=Recorder):
            client = LLMFactory.create_chat_model(config)

        self.assertNotIn("model_kwargs", client.kwargs)
        self.assertEqual(client.kwargs["temperature"], 0.7)
        self.assertIsNone(client.kwargs["api_key"])

    def test_anthropic_gets_a_capped_temperature_and_no_response_format(self):
        config = Config(
            llm_provider="anthropic",
            anthropic_api_key="sk-ant",
            anthropic_model="claude-3-haiku-20240307",
            llm_temperature=0.9,
        )
        with fake_provider_module("langchain_anthropic", ChatAnthropic=Recorder):
            client = LLMFactory.create_chat_model(config)

        self.assertEqual(client.kwargs["api_key"], "sk-ant")
        self.assertEqual(client.kwargs["model"], "claude-3-haiku-20240307")
        self.assertEqual(client.kwargs["temperature"], 0.1)
        self.assertNotIn("model_kwargs", client.kwargs)

    def test_ollama_switches_on_json_format(self):
        config = Config(llm_provider="ollama", ollama_model="llama3.2", llm_json_mode=True)
        with fake_provider_module("langchain_ollama", ChatOllama=Recorder):
            client = LLMFactory.create_chat_model(config)

        self.assertEqual(client.kwargs["format"], "json")
        self.assertEqual(client.kwargs["model"], "llama3.2")
        self.assertEqual(client.kwargs["base_url"], "http://localhost:11434")

    def test_ollama_without_json_mode_sets_no_format(self):
        config = Config(llm_provider="ollama", llm_json_mode=False)
        with fake_provider_module("langchain_ollama", ChatOllama=Recorder):
            client = LLMFactory.create_chat_model(config)

        self.assertIsNone(client.kwargs["format"])

    def test_huggingface_inference_requires_a_token(self):
        config = Config(llm_provider="huggingface_inference", hf_api_token="")
        with fake_provider_module(
            "langchain_huggingface", ChatHuggingFace=Recorder, HuggingFaceEndpoint=Recorder
        ):
            with self.assertRaises(ValueError) as ctx:
                LLMFactory.create_chat_model(config)

        self.assertIn("HF_TOKEN", str(ctx.exception))

    def test_huggingface_inference_wraps_an_endpoint(self):
        config = Config(
            llm_provider="huggingface_inference",
            hf_api_token="hf-test",
            model_name="org/model",
            inference_provider="scaleway",
            hf_max_new_tokens=256,
            llm_temperature=0.0,
        )
        with fake_provider_module(
            "langchain_huggingface", ChatHuggingFace=Recorder, HuggingFaceEndpoint=Recorder
        ):
            client = LLMFactory.create_chat_model(config)

        endpoint = client.kwargs["llm"]
        self.assertEqual(endpoint.kwargs["repo_id"], "org/model")
        self.assertEqual(endpoint.kwargs["provider"], "scaleway")
        self.assertEqual(endpoint.kwargs["huggingfacehub_api_token"], "hf-test")
        self.assertEqual(endpoint.kwargs["max_new_tokens"], 256)
        self.assertFalse(endpoint.kwargs["do_sample"])

    def test_huggingface_inference_enables_sampling_for_a_warm_temperature(self):
        config = Config(
            llm_provider="hf",
            hf_api_token="hf-test",
            llm_json_mode=False,
            llm_temperature=0.7,
        )
        with fake_provider_module(
            "langchain_huggingface", ChatHuggingFace=Recorder, HuggingFaceEndpoint=Recorder
        ):
            client = LLMFactory.create_chat_model(config)

        self.assertTrue(client.kwargs["llm"].kwargs["do_sample"])

    def test_huggingface_endpoint_provider(self):
        config = Config(
            llm_provider="huggingface_endpoint",
            hf_endpoint_url="https://endpoint.local",
            hf_api_token="hf-test",
            hf_max_new_tokens=128,
        )
        with fake_provider_module("langchain_huggingface", HuggingFaceEndpoint=Recorder):
            client = LLMFactory.create_chat_model(config)

        self.assertEqual(client.kwargs["endpoint_url"], "https://endpoint.local")
        self.assertEqual(client.kwargs["task"], "text-generation")
        self.assertEqual(
            client.kwargs["model_kwargs"], {"max_new_tokens": 128, "temperature": 0.1}
        )

    def test_huggingface_local_provider(self):
        config = Config(
            llm_provider="huggingface_local",
            hf_api_token="",
            hf_model_name="mistralai/Mistral-7B-Instruct-v0.3",
        )
        with fake_provider_module("langchain_huggingface", ChatHuggingFace=Recorder):
            client = LLMFactory.create_chat_model(config)

        self.assertEqual(client.kwargs["model_name"], "mistralai/Mistral-7B-Instruct-v0.3")
        self.assertIsNone(client.kwargs["huggingfacehub_api_token"])


if __name__ == "__main__":
    unittest.main()
