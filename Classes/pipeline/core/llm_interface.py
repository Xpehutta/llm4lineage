"""Provider-neutral LLM boundary for the pipeline core.

Core modules talk to models through :class:`LLMInterface` so that installing
the package without the ``[llm]`` extra still yields a working pipeline.
LangChain is confined to :class:`LangChainChatAdapter` (imported lazily) and to
:mod:`Classes.pipeline.core.llm_factory`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Protocol, Sequence, runtime_checkable

__all__ = [
    "ChatMessage",
    "LLMInterface",
    "LangChainChatAdapter",
    "MockLLM",
    "adapt_llm",
    "message_text",
    "render_template",
]


@dataclass(frozen=True)
class ChatMessage:
    """A single chat turn. ``role`` is one of ``system``/``user``/``assistant``."""

    role: str
    content: str


@runtime_checkable
class LLMInterface(Protocol):
    """The only contract the pipeline core requires of a language model."""

    def invoke(self, prompt: str) -> str:
        """Send a single user prompt and return the text response."""
        ...

    def invoke_messages(self, messages: Sequence[ChatMessage]) -> str:
        """Send a chat transcript and return the text response."""
        ...


def message_text(response: Any) -> str:
    """Normalise a model response into plain text.

    Handles raw strings, objects exposing ``.content``, and the list-of-parts
    shape that multimodal chat models return.
    """
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def render_template(template: str, values: dict) -> str:
    """Substitute ``{name}`` placeholders without touching other braces.

    ``str.format`` cannot be used here: prompts embed JSON schemas whose braces
    would be misread as placeholders.
    """
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


class MockLLM:
    """Deterministic model used by tests and langchain-free installations."""

    model_label = "mock"

    def __init__(self, responses: Optional[Iterable[str]] = None):
        self._responses = list(responses) if responses is not None else ["Mock LLM response."]
        self._index = 0

    def _next(self) -> str:
        if not self._responses:
            return ""
        response = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return response

    def invoke(self, prompt: str) -> str:
        return self._next()

    def invoke_messages(self, messages: Sequence[ChatMessage]) -> str:
        return self._next()


class LangChainChatAdapter:
    """Adapt a LangChain ``BaseChatModel`` to :class:`LLMInterface`."""

    def __init__(self, chat_model: Any):
        self.chat_model = chat_model

    @property
    def model_label(self) -> str:
        for attr in ("model_name", "model_id", "model"):
            value = getattr(self.chat_model, attr, None)
            if isinstance(value, str) and value:
                return value
        inner = getattr(self.chat_model, "llm", None)
        if inner is not None:
            for attr in ("repo_id", "model", "model_id"):
                value = getattr(inner, attr, None)
                if isinstance(value, str) and value:
                    return value
        return type(self.chat_model).__name__

    def invoke(self, prompt: str) -> str:
        return self.invoke_messages([ChatMessage("user", prompt)])

    def invoke_messages(self, messages: Sequence[ChatMessage]) -> str:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        role_map = {"system": SystemMessage, "assistant": AIMessage, "user": HumanMessage}
        payload = [role_map.get(m.role, HumanMessage)(content=m.content) for m in messages]
        return message_text(self.chat_model.invoke(payload))


def adapt_llm(llm: Any) -> LLMInterface:
    """Return ``llm`` as an :class:`LLMInterface`, wrapping LangChain models.

    Raises:
        TypeError: when ``llm`` is neither an ``LLMInterface`` nor something
            with an ``invoke`` method to adapt.
    """
    if llm is None:
        raise TypeError("An LLM instance is required")
    if hasattr(llm, "invoke_messages"):
        return llm
    if hasattr(llm, "invoke"):
        return LangChainChatAdapter(llm)
    raise TypeError(
        f"{type(llm).__name__} does not implement LLMInterface and has no `invoke` method"
    )
