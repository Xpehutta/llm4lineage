"""Shared LLM construction helpers for all Classes modules."""

from __future__ import annotations

import os
from typing import Any, Optional

from Classes.helper_classes import HuggingFaceLLMAdapter
from Classes.pipeline.core.llm_factory import LLMFactory
from Classes.pipeline.models.config import Config

DEFAULT_MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DEFAULT_PROVIDER = "scaleway"


def resolve_model_name(
    model: Optional[str] = None,
    default: str = DEFAULT_MODEL_NAME,
) -> str:
    """Explicit argument > MODEL_NAME env var > default."""
    return model or os.environ.get("MODEL_NAME") or default


def resolve_provider(
    provider: Optional[str] = None,
    default: str = DEFAULT_PROVIDER,
) -> str:
    """Explicit argument > PROVIDER env var > default."""
    return provider or os.environ.get("PROVIDER") or default


def resolve_hf_token(hf_token: Optional[str] = None) -> Optional[str]:
    """Explicit argument > HF_TOKEN or HF_API_TOKEN env var."""
    return hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HF_API_TOKEN")


def build_config(
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    hf_token: Optional[str] = None,
    llm_provider: Optional[str] = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    do_sample: Optional[bool] = None,
    sql_dialect: Optional[str] = None,
    **overrides: Any,
) -> Config:
    """Build a pipeline Config from explicit args and environment."""
    token = resolve_hf_token(hf_token)
    payload: dict[str, Any] = {
        "model_name": resolve_model_name(model),
        "inference_provider": resolve_provider(provider),
        "hf_max_new_tokens": max_new_tokens,
        "llm_temperature": temperature,
        "llm_provider": llm_provider or os.environ.get("LLM_PROVIDER") or "huggingface_inference",
    }
    if token:
        payload["hf_api_token"] = token
    if do_sample is not None:
        payload["hf_do_sample"] = do_sample
    if sql_dialect is not None:
        payload["sql_dialect"] = sql_dialect
    payload.update(overrides)
    return Config(**payload)


def create_chat_model(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    hf_token: Optional[str] = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    do_sample: Optional[bool] = None,
    llm_provider: Optional[str] = None,
    **config_overrides: Any,
) -> Any:
    """Create a provider chat model using the unified LLMFactory.

    Returns the raw provider client (a LangChain model for real providers) so
    existing callers that poke at model attributes keep working.
    """
    config = build_config(
        model=model,
        provider=provider,
        hf_token=hf_token,
        llm_provider=llm_provider,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        **config_overrides,
    )
    return LLMFactory.create_chat_model(config)


def create_chat_adapter(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    hf_token: Optional[str] = None,
    **kwargs: Any,
) -> HuggingFaceLLMAdapter:
    """Create chat model + HuggingFaceLLMAdapter (legacy Classes API)."""
    return HuggingFaceLLMAdapter(
        create_chat_model(
            model=model,
            provider=provider,
            hf_token=hf_token,
            **kwargs,
        )
    )
