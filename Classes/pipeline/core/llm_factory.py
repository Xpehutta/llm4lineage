"""Provider-agnostic LangChain chat model creation."""

import logging

from langchain_core.language_models.chat_models import BaseChatModel

from Classes.pipeline.models.config import Config

logger = logging.getLogger(__name__)


class LLMFactory:
    """Create a LangChain chat model from pipeline configuration."""

    @staticmethod
    def create(config: Config) -> BaseChatModel:
        provider = config.llm_provider.lower().strip()
        logger.info("Creating LLM instance for provider: %s", provider)

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
                    do_sample=config.hf_do_sample or config.llm_temperature > 0,
                    temperature=config.llm_temperature,
                )
            )

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                api_key=config.openai_api_key.get_secret_value() or None,
                model=config.openai_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
            )

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                api_key=config.anthropic_api_key.get_secret_value() or None,
                model=config.anthropic_model,
                temperature=config.llm_temperature,
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
                    "temperature": config.llm_temperature,
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
                    "temperature": config.llm_temperature,
                },
            )

        if provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                base_url=config.ollama_base_url,
                model=config.ollama_model,
                temperature=config.llm_temperature,
            )

        if provider == "mock":
            from langchain_core.language_models.fake_chat_models import (
                GenericFakeChatModel,
            )

            return GenericFakeChatModel(
                messages=iter(["Mock LLM response."])
            )

        raise ValueError(f"Unsupported LLM provider: {provider}")
