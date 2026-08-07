"""LLM cache helpers and connectivity checks for the web UI."""

from __future__ import annotations

import hashlib
from typing import Any

from Classes.llm_cache import LLMCache
from Classes.pipeline.llm_helpers import create_chat_model


def llm_config_key(model: str, provider: str, token: str) -> str:
    token_fp = hashlib.sha256(token.encode()).hexdigest()[:16] if token else ""
    return f"{model.strip()}|{provider.strip()}|{token_fp}"


def make_llm_cache(enabled: bool) -> LLMCache | None:
    """Return an LLMCache instance when caching is enabled, else None."""
    return LLMCache() if enabled else None


def run_llm_health_check(model: str, provider: str, token: str) -> dict[str, Any]:
    """Minimal HF inference ping — one short completion to verify access."""
    from langchain_core.messages import HumanMessage

    chat = create_chat_model(
        model=model,
        provider=provider,
        hf_token=token,
        max_new_tokens=16,
        temperature=0.0,
    )
    response = chat.invoke([HumanMessage(content="Reply with exactly one word: OK")])
    content = getattr(response, "content", None)
    if content is None:
        content = str(response)
    if isinstance(content, list):
        content = "".join(
            part.get("text", str(part)) if isinstance(part, dict) else str(part) for part in content
        )
    preview = str(content).strip()
    if not preview:
        raise RuntimeError("Empty response from model")
    return {"ok": True, "preview": preview[:200], "model": model, "provider": provider}


def execute_llm_health_check(model: str, provider: str, token: str) -> dict[str, Any]:
    try:
        return run_llm_health_check(model, provider, token)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "model": model, "provider": provider}


def cache_status_captions(cache_info: dict[str, Any]) -> list[str]:
    """Human-readable cache status lines for the results panel."""
    if not cache_info:
        return []
    captions: list[str] = []
    if cache_info.get("hit"):
        captions.append(f"Cache hit · quality `{cache_info.get('quality_score', '—')}`")
    elif cache_info.get("updated"):
        prev = cache_info.get("previous_quality_score")
        prev_text = f" (prev `{prev:.2f}`)" if prev is not None else ""
        captions.append(
            f"Cache updated · quality `{cache_info.get('quality_score', '—')}`{prev_text}"
        )
    elif cache_info.get("updated") is False and cache_info.get("previous_quality_score") is not None:
        captions.append(
            "Cache kept — fresh result did not beat cached quality "
            f"(`{cache_info.get('quality_score', '—')}` vs `{cache_info.get('previous_quality_score', '—')}`)."
        )
    elif not cache_info.get("read_enabled"):
        captions.append("Cache bypassed for this run.")
    return captions
