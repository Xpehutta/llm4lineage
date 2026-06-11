import json
import re
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from Classes.helper_classes import HuggingFaceLLMAdapter


class DELLMKnowledge(BaseModel):
    """Structured DELLM output with light validation."""

    knowledge: str = Field(min_length=1, description="Short expert context paragraph.")
    categories: List[str] = Field(default_factory=list)

    @field_validator("knowledge")
    def normalize_knowledge(cls, value: str) -> str:
        text = re.sub(r"\s+", " ", value or "").strip()
        if not text:
            raise ValueError("knowledge cannot be empty")
        return text

    @field_validator("categories")
    def normalize_categories(cls, value: List[str]) -> List[str]:
        cleaned: List[str] = []
        for item in value:
            normalized = str(item).strip().lower()
            if normalized and normalized not in cleaned:
                cleaned.append(normalized)
        return cleaned


class DELLMGenerator:
    """
    Data Expert LLM (DELLM) implementation for knowledge-to-SQL augmentation.

    The generator receives question + schema and returns short expert context
    focused on arithmetic logic, domain terminology, and formatting/synonyms.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        provider: str = "scaleway",
        hf_token: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        max_retries: int = 3,
        max_words: int = 140,
    ):
        if not hf_token:
            raise ValueError("HF_TOKEN is required for DELLM generation.")

        self.model = model
        self.provider = provider
        self.max_retries = max_retries
        self.max_words = max_words

        self.chat_model = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id=model,
                task="text-generation",
                provider=provider,
                huggingfacehub_api_token=hf_token,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
            )
        )
        self.chat_adapter = HuggingFaceLLMAdapter(self.chat_model)

        self.system_prompt = (
            "You are DELLM (Data Expert LLM). "
            "Given a user question and database schema, produce concise expert context "
            "that helps a downstream text-to-SQL model. "
            "Prioritize hidden arithmetic rules, domain terminology/code mappings, "
            "and formatting/synonym hints. "
            "Do not write SQL. Return JSON: "
            '{"knowledge":"...", "categories":["arithmetic_reasoning","domain_terminology","formatting_synonyms"]}.'
        )

    @staticmethod
    def _is_auth_error(error: Exception) -> bool:
        marker = str(error).lower()
        return any(s in marker for s in ["401", "unauthorized", "bad credentials", "forbidden"])

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise

    @classmethod
    def _parse_knowledge_payload(cls, response_text: str) -> Dict[str, Any]:
        """
        Parse model output into the knowledge dict.

        The paper's native DELLM output is a plain natural-language paragraph;
        JSON is only our transport format. If the model skips the JSON envelope,
        treat the raw text as the knowledge paragraph instead of failing.
        """
        try:
            parsed = cls._extract_json(response_text)
        except (json.JSONDecodeError, ValueError):
            return {"knowledge": str(response_text).strip().strip("`"), "categories": []}
        if isinstance(parsed, dict):
            return parsed
        return {"knowledge": str(parsed).strip(), "categories": []}

    @staticmethod
    def _trim_words(text: str, max_words: int) -> str:
        words = re.findall(r"\S+", text or "")
        if len(words) <= max_words:
            return text.strip()
        return " ".join(words[:max_words]).strip()

    @staticmethod
    def _normalize_schema(schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not schema:
            return {}
        if not isinstance(schema, dict):
            return {"raw_schema": str(schema)}
        return schema

    def _build_user_prompt(self, question: str, schema: Optional[Dict[str, Any]]) -> str:
        schema_payload = self._normalize_schema(schema)
        return "\n".join(
            [
                "User question:",
                question.strip(),
                "",
                "Database schema JSON:",
                json.dumps(schema_payload, indent=2),
                "",
                "Instructions:",
                "- Output one concise paragraph of expert context (roughly 50-200 tokens).",
                "- Include only details needed for SQL generation.",
                "- Cover, when available: arithmetic reasoning, domain terminology, formatting/synonyms.",
                "- Avoid fabricating schema fields that do not exist.",
            ]
        )

    def _invoke_messages_text(self, messages: List[Any]) -> str:
        if hasattr(self.chat_adapter, "invoke_messages"):
            return self.chat_adapter.invoke_messages(messages)
        if hasattr(self.chat_adapter, "invoke"):
            return str(self.chat_adapter.invoke(messages))
        return str(self.chat_model.invoke(messages))

    def generate_knowledge(self, question: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate DELLM expert context from question + schema."""
        if not question or not question.strip():
            return {"error": "question is required", "knowledge": "", "categories": []}

        prompt = self._build_user_prompt(question=question, schema=schema)
        for attempt in range(1, self.max_retries + 1):
            try:
                response_text = self._invoke_messages_text(
                    [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]
                )
                parsed = self._parse_knowledge_payload(response_text)
                parsed["knowledge"] = self._trim_words(str(parsed.get("knowledge", "")), self.max_words)
                validated = DELLMKnowledge.model_validate(parsed)
                return validated.model_dump()
            except Exception as exc:
                if self._is_auth_error(exc):
                    return {
                        "error": "Hugging Face authentication failed for DELLM generator.",
                        "details": str(exc),
                        "knowledge": "",
                        "categories": [],
                    }
                if attempt == self.max_retries:
                    return {
                        "error": "DELLM generation failed",
                        "details": str(exc),
                        "knowledge": "",
                        "categories": [],
                    }
                time.sleep(min(10, 2**attempt))

        return {"error": "DELLM generation failed", "knowledge": "", "categories": []}

    def build_augmented_prompt(
        self,
        question: str,
        schema: Optional[Dict[str, Any]] = None,
        knowledge: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build final prompt for the main SQL model:
        [question] + [schema] + [dellm_knowledge].
        """
        generated: Dict[str, Any] = {"knowledge": knowledge or "", "categories": []}
        if not generated["knowledge"]:
            generated = self.generate_knowledge(question=question, schema=schema)

        schema_payload = self._normalize_schema(schema)
        final_prompt = "\n".join(
            [
                "User Question:",
                question.strip(),
                "",
                "Database Schema JSON:",
                json.dumps(schema_payload, indent=2),
                "",
                "DELLM Expert Knowledge:",
                generated.get("knowledge", ""),
            ]
        ).strip()

        return {
            "final_prompt": final_prompt,
            "knowledge": generated.get("knowledge", ""),
            "categories": generated.get("categories", []),
            "error": generated.get("error"),
            "details": generated.get("details"),
        }
