"""Provider-agnostic chat model creation.

This is the only pipeline module that reaches for LangChain, and it does so
lazily: every provider imports its backend inside the branch that needs it, so
a core-only install can still build the ``mock`` provider.
"""

import logging

from Classes.pipeline.core.llm_interface import LLMInterface, MockLLM, adapt_llm
from Classes.pipeline.models.config import Config

logger = logging.getLogger(__name__)


#: Providers whose API can guarantee a JSON response. Everywhere else the
#: prompt's schema plus a low temperature is the only lever available.
JSON_MODE_PROVIDERS = frozenset({"openai", "ollama"})


class LLMFactory:
    """Create an :class:`LLMInterface` from pipeline configuration."""

    @staticmethod
    def create(config: Config) -> LLMInterface:
        return adapt_llm(LLMFactory.create_chat_model(config))

    @staticmethod
    def effective_temperature(config: Config) -> float:
        """Cap sampling while JSON mode is on, so the schema is actually followed."""
        if config.llm_json_mode:
            return min(config.llm_temperature, config.llm_json_mode_max_temperature)
        return config.llm_temperature

    @staticmethod
    def create_chat_model(config: Config):
        """Build the raw provider client (a LangChain model for most providers)."""
        provider = config.llm_provider.lower().strip()
        logger.info("Creating LLM instance for provider: %s", provider)
        temperature = LLMFactory.effective_temperature(config)

        if provider == "mock":
            return MockLLM()

        if provider in {"huggingface_inference", "huggingface", "hf"}:
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

            token = config.hf_api_token.get_secret_value() or None
            if not token:
                raise ValueError(
                    "HF_TOKEN (or HF_API_TOKEN) is required for huggingface_inference provider."
                )
            return ChatHuggingFace(
                llm=HuggingFaceEndpoint(
                    repo_id=config.model_name,
                    task="text-generation",
                    provider=config.inference_provider,
                    huggingfacehub_api_token=token,
                    max_new_tokens=config.hf_max_new_tokens,
                    do_sample=config.hf_do_sample or temperature > 0,
                    temperature=temperature,
                )
            )

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            kwargs = {}
            if config.llm_json_mode:
                kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
            return ChatOpenAI(
                api_key=config.openai_api_key.get_secret_value() or None,
                model=config.openai_model,
                temperature=temperature,
                max_tokens=config.llm_max_tokens,
                **kwargs,
            )

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            # Anthropic has no response_format switch; the prompt schema plus a
            # capped temperature is the only enforcement available here.
            return ChatAnthropic(
                api_key=config.anthropic_api_key.get_secret_value() or None,
                model=config.anthropic_model,
                temperature=temperature,
                max_tokens=config.llm_max_tokens,
            )

        if provider == "huggingface_endpoint":
            from langchain_huggingface import HuggingFaceEndpoint

            return HuggingFaceEndpoint(
                endpoint_url=config.hf_endpoint_url,
                huggingfacehub_api_token=(
                    config.hf_api_token.get_secret_value() or None
                ),
                task="text-generation",
                model_kwargs={
                    "max_new_tokens": config.hf_max_new_tokens,
                    "temperature": temperature,
                },
            )

        if provider == "huggingface_local":
            from langchain_huggingface import ChatHuggingFace

            return ChatHuggingFace(
                model_name=config.hf_model_name,
                huggingfacehub_api_token=(
                    config.hf_api_token.get_secret_value() or None
                ),
                model_kwargs={
                    "max_new_tokens": config.hf_max_new_tokens,
                    "temperature": temperature,
                },
            )

        if provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                base_url=config.ollama_base_url,
                model=config.ollama_model,
                temperature=temperature,
                format="json" if config.llm_json_mode else None,
            )

        raise ValueError(f"Unsupported LLM provider: {provider}")
