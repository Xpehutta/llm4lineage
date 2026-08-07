"""JSON extraction helpers shared by agents (no langchain)."""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["estimate_tokens", "extract_json", "truncate_to_token_budget"]

#: Approximate token budget per object for Resolver prompts.
DEFAULT_TOKEN_BUDGET = 30_000


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ``len(text) // 4`` (never below 0)."""
    return max(0, len(text) // 4)


def truncate_to_token_budget(text: str, budget: int = DEFAULT_TOKEN_BUDGET) -> str:
    """Keep at most ``budget`` approximate tokens (``budget * 4`` characters)."""
    if budget <= 0:
        return ""
    max_chars = budget * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def extract_json(text: str) -> Any:
    """Parse a JSON value from an LLM response, tolerating surrounding prose."""
    if text is None:
        raise ValueError("empty LLM response")
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty LLM response")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Fenced ```json ... ``` blocks
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped, re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    # First object or array span
    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("Response contained no valid JSON")
