"""Optional LLM verify / enhance / parse-fallback for SQL2Graph."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from pydantic import ValidationError

from Classes.helper_classes import HuggingFaceLLMAdapter, resolve_model_name, resolve_provider
from Classes.pipeline.llm_helpers import create_chat_model, resolve_hf_token
from Classes.sql2graph.models import SQL2GraphExtraction
from Classes.sql2graph.parser import SQL2GraphParser

logger = logging.getLogger(__name__)

def _chat_messages(system_prompt: str, human_prompt: str) -> list[Any]:
    """Build LangChain chat messages.

    Imported lazily so that `import Classes` works without the `[llm]` extra;
    only the LLM code paths below need LangChain.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

class SQL2GraphLLMExtractor:
    """LLM-backed extractor for column-level lineage JSON."""

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        hf_token: str | None = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        max_retries: int = 5,
        enable_refinement: bool = True,
        cache: Any | None = None,
        prompt_version: str = "v2.1",
        use_llm_cache: bool = True,
    ):
        if not resolve_hf_token(hf_token):
            raise ValueError("HF_TOKEN is required for SQL2Graph extraction.")

        model = resolve_model_name(model)
        provider = resolve_provider(provider)
        self.model = model
        self.provider = provider
        self.max_retries = max_retries
        self.enable_refinement = enable_refinement
        self.use_llm_cache = use_llm_cache
        if cache is not None:
            self.cache = cache
        elif use_llm_cache:
            from Classes.llm_cache import LLMCache

            self.cache = LLMCache()
        else:
            self.cache = None
        self.prompt_version = prompt_version
        self.chat_model = create_chat_model(
            model=model,
            provider=provider,
            hf_token=hf_token,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        self.chat_adapter = HuggingFaceLLMAdapter(self.chat_model)
        self.structured_llm = self._try_create_structured_llm(self.chat_model)

        self.verification_system_prompt = (
            "You are a strict SQL lineage verifier. You receive deterministic column-level lineage "
            "produced by sqlglot plus the original SQL. Verify the draft against the SQL and return "
            "corrected JSON only when you find concrete issues. "
            "Return keys: ctes, output_columns, filters, joins, group_by_columns. "
            "Preserve correct sqlglot-derived fields; do not invent columns absent from the SQL."
        )
        self.enhancement_system_prompt = (
            "You are a SQL lineage enhancer. You receive a verified column-level lineage draft. "
            "Apply targeted enhancements inferable from the SQL: complete missing dependencies, "
            "filters, join keys, and CTE scopes. Preserve fields that are already correct."
        )
        # Backward-compatible aliases
        self.system_prompt = self.verification_system_prompt
        self.refinement_system_prompt = self.enhancement_system_prompt

    @property
    def _llm_system_prompt(self) -> str:
        return getattr(self, "verification_system_prompt", None) or getattr(self, "system_prompt", "")

    @staticmethod
    def _try_create_structured_llm(chat_model: Any) -> Any:
        """Return a structured-output runnable when the model supports it."""
        try:
            return chat_model.with_structured_output(SQL2GraphExtraction)
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_structured_result(result: Any) -> SQL2GraphExtraction:
        if isinstance(result, SQL2GraphExtraction):
            return result
        if isinstance(result, dict):
            return SQL2GraphExtraction.model_validate(result)
        raise TypeError(f"Unexpected structured LLM result type: {type(result)!r}")

    def _invoke_structured_extraction(self, messages: list[Any]) -> SQL2GraphExtraction:
        """Invoke the LLM and return validated SQL2GraphExtraction."""
        structured_llm = getattr(self, "structured_llm", None)
        if structured_llm is not None:
            return self._coerce_structured_result(structured_llm.invoke(messages))

        response_text = self._invoke_messages_text(messages)
        payload = self._normalize_scope_payload(self._extract_json(response_text))
        return SQL2GraphExtraction.model_validate(payload)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise

    @staticmethod
    def _extract_column_refs_from_text(text: str) -> list[dict[str, str]]:
        """Best-effort extraction of alias.column pairs from condition text."""
        pattern = r'([A-Za-z_][\w\$]*)\.(?:"([^"]+)"|([A-Za-z_][\w\$]*))'
        refs: list[dict[str, str]] = []
        seen = set()
        for match in re.findall(pattern, text or ""):
            alias = match[0]
            column = match[1] or match[2]
            key = (alias, column)
            if key in seen:
                continue
            seen.add(key)
            refs.append({"table_alias": alias, "column": column})
        return refs

    @staticmethod
    def _coerce_column_ref(value: Any) -> dict[str, str | None] | None:
        """
        Coerce common LLM column-reference variants into {"table_alias", "column"}.

        Accepts "alias.column" / "column" strings and dicts using alternative keys
        (table/alias instead of table_alias, name instead of column).
        """
        if isinstance(value, str):
            text = value.strip().strip('"')
            if not text:
                return None
            alias, _, column = text.rpartition(".")
            return {"table_alias": alias or None, "column": column.strip('"')}
        if isinstance(value, dict):
            column = value.get("column") or value.get("name")
            if not column or not str(column).strip():
                return None
            alias = value.get("table_alias") or value.get("table") or value.get("alias")
            return {"table_alias": str(alias) if alias else None, "column": str(column).strip()}
        return None

    @classmethod
    def _coerce_join_columns(cls, join: dict[str, Any]) -> list[dict[str, str | None]]:
        """Coerce join_columns variants, including {"left_column", "right_column"} pairs."""
        raw = join.get("join_columns")
        items = raw if isinstance(raw, list) else ([raw] if raw else [])

        refs: list[dict[str, str | None]] = []
        for item in items:
            if isinstance(item, dict) and ("left_column" in item or "right_column" in item):
                for side, alias_key in (("left_column", "left_alias"), ("right_column", "right_alias")):
                    ref = cls._coerce_column_ref(item.get(side))
                    if ref:
                        if not ref.get("table_alias"):
                            ref["table_alias"] = join.get(alias_key) or None
                        refs.append(ref)
            else:
                ref = cls._coerce_column_ref(item)
                if ref:
                    refs.append(ref)
        return refs

    @classmethod
    def _normalize_scope_payload(cls, scope: dict[str, Any]) -> dict[str, Any]:
        """Normalize partially structured LLM payload into schema-compatible shape."""
        normalized = dict(scope or {})
        normalized.setdefault("output_columns", [])
        normalized.setdefault("filters", [])
        normalized.setdefault("joins", [])
        normalized.setdefault("ctes", [])
        normalized.setdefault("group_by_columns", [])

        fixed_outputs = []
        for item in normalized.get("output_columns", []):
            out = dict(item or {})
            deps = out.get("dependencies") or []
            out["dependencies"] = [ref for ref in (cls._coerce_column_ref(dep) for dep in deps) if ref]
            fixed_outputs.append(out)
        normalized["output_columns"] = fixed_outputs

        fixed_filters = []
        for item in normalized.get("filters", []):
            filt = dict(item or {})
            columns_used = filt.get("columns_used") or []
            filt["columns_used"] = [ref for ref in (cls._coerce_column_ref(col) for col in columns_used) if ref]
            clause = filt.get("clause")

            # Some model outputs place the full predicate in `clause` and omit `condition`.
            if not filt.get("condition"):
                if isinstance(clause, str) and any(token in clause for token in ["=", ">", "<", "(", ")", " and ", " or "]):
                    filt["condition"] = clause
                    filt["clause"] = "WHERE"
                else:
                    filt["condition"] = ""

            if not filt.get("clause"):
                filt["clause"] = "WHERE"
            fixed_filters.append(filt)
        normalized["filters"] = fixed_filters

        fixed_joins = []
        for item in normalized.get("joins", []):
            join = dict(item or {})
            join.setdefault("type", "INNER")
            join.setdefault("left_alias", "")
            join.setdefault("right_alias", "")
            join.setdefault("condition", "")

            join_columns = cls._coerce_join_columns(join)
            if len(join_columns) >= 2:
                join["join_columns"] = join_columns[:2]
            else:
                extracted = cls._extract_column_refs_from_text(join.get("condition", ""))
                if len(extracted) >= 2:
                    join["join_columns"] = extracted[:2]
                else:
                    join["join_columns"] = [
                        {"table_alias": join.get("left_alias") or "unknown_left", "column": "unknown"},
                        {"table_alias": join.get("right_alias") or "unknown_right", "column": "unknown"},
                    ]
            fixed_joins.append(join)
        normalized["joins"] = fixed_joins

        normalized["group_by_columns"] = [
            ref
            for ref in (cls._coerce_column_ref(col) for col in normalized.get("group_by_columns", []) or [])
            if ref
        ]

        fixed_ctes = []
        for cte in normalized.get("ctes", []):
            cte_copy = dict(cte or {})
            cte_copy = cls._normalize_scope_payload(cte_copy)
            cte_copy.setdefault("alias", "cte")
            fixed_ctes.append(cte_copy)
        normalized["ctes"] = fixed_ctes

        return normalized

    @staticmethod
    def _is_auth_error(error: Exception) -> bool:
        marker = str(error).lower()
        return any(s in marker for s in ["401", "unauthorized", "bad credentials", "forbidden"])

    @staticmethod
    def _is_transient_error(error: Exception) -> bool:
        marker = str(error).lower()
        return any(
            s in marker
            for s in [
                "busy",
                "try again",
                "rate limit",
                "too many requests",
                "429",
                "502",
                "503",
                "504",
                "timeout",
                "server_error",
                "completion_error",
                "overloaded",
                "temporarily unavailable",
            ]
        )

    @staticmethod
    def _format_llm_error(error: Exception | str) -> str:
        text = str(error)
        text = re.sub(r"\(Request ID:[^)]+\)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"Root=[^;\)]+[;\)]?", "", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip()

    def _build_verification_prompt(
        self,
        sql: str,
        schema: dict[str, Any] | None,
        simplified_query: dict[str, Any] | None,
        deterministic_draft: dict[str, Any],
        validation_error: str | None = None,
    ) -> str:
        return "\n".join(
            [
                "Original SQL:",
                sql,
                "",
                "Schema JSON (optional):",
                json.dumps(schema or {}, indent=2),
                "",
                "Sqlglot simplify summary (optional):",
                json.dumps(simplified_query or {}, indent=2),
                "",
                "Deterministic sqlglot lineage draft to verify:",
                json.dumps(deterministic_draft, indent=2),
                "",
                "Tasks:",
                "1. Verify each output column dependency against the SQL.",
                "2. Confirm filters, join keys, and CTE scopes match the SQL.",
                "3. Preserve correct sqlglot fields; do not replace valid dependencies.",
                "4. Return ONLY JSON: ctes, output_columns, filters, joins, group_by_columns.",
                "5. If the draft is already correct, return it with minimal or no changes.",
            ]
            + (
                ["", "Previous output failed validation:", validation_error, "Fix and return corrected JSON only."]
                if validation_error
                else []
            )
        )

    def _build_refinement_prompt(
        self,
        sql: str,
        schema: dict[str, Any] | None,
        simplified_query: dict[str, Any] | None,
        draft_payload: dict[str, Any],
    ) -> str:
        return "\n".join(
            [
                "Original SQL:",
                sql,
                "",
                "Schema JSON (optional):",
                json.dumps(schema or {}, indent=2),
                "",
                "Simplified query structure (optional):",
                json.dumps(simplified_query or {}, indent=2),
                "",
                "Draft lineage JSON to repair:",
                json.dumps(draft_payload, indent=2),
                "",
                "Repair requirements:",
                "- Ensure every filter has non-empty clause and condition.",
                "- Ensure joins have 2 concrete join_columns when inferable from join condition.",
                "- Ensure output dependencies are complete for expressions/functions/CASE.",
                "- Preserve existing good fields and avoid inventing unrelated aliases.",
                "- Keep strict JSON shape compatible with: ctes, output_columns, filters, joins, group_by_columns.",
            ]
        )

    @staticmethod
    def _response_to_text(response: Any) -> str:
        """Normalize chat response object to text."""
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)

    def _invoke_messages_text(self, messages: list[Any]) -> str:
        """Invoke chat using adapter with compatibility for older adapter versions."""
        if hasattr(self.chat_adapter, "invoke_messages"):
            return self.chat_adapter.invoke_messages(messages)

        # Backward compatibility: older adapter may only expose invoke(prompt).
        if hasattr(self.chat_adapter, "invoke"):
            raw = self.chat_adapter.invoke(messages)
            return self._response_to_text(raw)

        # Last resort fallback to direct chat model invocation.
        raw = self.chat_model.invoke(messages)
        return self._response_to_text(raw)

    def _refine_payload_with_llm(
        self,
        sql: str,
        schema: dict[str, Any] | None,
        simplified_query: dict[str, Any] | None,
        draft_payload: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = self._build_refinement_prompt(
            sql=sql,
            schema=schema,
            simplified_query=simplified_query,
            draft_payload=draft_payload,
        )
        messages = _chat_messages(self.enhancement_system_prompt, prompt)
        structured_llm = getattr(self, "structured_llm", None)
        if structured_llm is not None:
            refined = self._coerce_structured_result(structured_llm.invoke(messages))
            return self._normalize_scope_payload(refined.model_dump())

        response_text = self._invoke_messages_text(messages)
        refined = self._extract_json(response_text)
        return self._normalize_scope_payload(refined)

    def _build_user_prompt(
        self,
        sql: str,
        schema: dict[str, Any] | None,
        simplified_query: dict[str, Any] | None,
        validation_error: str | None = None,
    ) -> str:
        parts = [
            "SQL:",
            sql,
            "",
            "Schema JSON (optional):",
            json.dumps(schema or {}, indent=2),
            "",
            "Simplified query structure (optional):",
            json.dumps(simplified_query or {}, indent=2),
            "",
            "Output format constraints:",
            "- output_columns[].dependencies[] must include table_alias and column",
            "- filters[].columns_used[] must include all columns in condition",
            "- joins[].join_columns must contain exactly 2 entries",
            "- include ctes recursively when present",
            "- include group_by_columns when GROUP BY is present",
        ]
        if validation_error:
            parts.extend(["", "Previous output failed validation:", validation_error, "Fix and return corrected JSON only."])
        return "\n".join(parts)

    def _invoke_verification_payload(
        self,
        sql: str,
        draft: dict[str, Any],
        *,
        schema: dict[str, Any] | None = None,
        simplified_query: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        last_validation_error: str | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                user_prompt = self._build_verification_prompt(
                    sql=sql,
                    schema=schema,
                    simplified_query=simplified_query,
                    deterministic_draft=draft,
                    validation_error=last_validation_error,
                )
                validated = self._invoke_structured_extraction(
                    _chat_messages(self.verification_system_prompt, user_prompt)
                )
                return self._normalize_scope_payload(validated.model_dump())
            except ValidationError as exc:
                last_validation_error = str(exc)
                if attempt == self.max_retries:
                    break
            except Exception as exc:
                if self._is_auth_error(exc):
                    return {
                        "error": "Hugging Face authentication failed for SQL2Graph extractor.",
                        "details": str(exc),
                    }
                if attempt == self.max_retries:
                    break
                time.sleep(min(10, 2**attempt))

        try:
            return SQL2GraphExtraction.model_validate(draft).model_dump()
        except ValidationError:
            return {
                "error": "LLM verification failed and sqlglot draft is invalid",
                "details": last_validation_error,
            }

    def verify(
        self,
        sql: str,
        deterministic_draft: dict[str, Any],
        schema: dict[str, Any] | None = None,
        simplified_query: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Step 3: verify sqlglot draft with LLM."""
        draft = self._normalize_scope_payload(dict(deterministic_draft or {}))
        return self._invoke_verification_payload(
            sql,
            draft,
            schema=schema,
            simplified_query=simplified_query,
        )

    def enhance(
        self,
        sql: str,
        verified_payload: dict[str, Any],
        schema: dict[str, Any] | None = None,
        simplified_query: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Step 4: enhance a verified draft with targeted LLM repairs."""
        if not self.enable_refinement:
            return self._normalize_scope_payload(dict(verified_payload or {}))

        draft = self._normalize_scope_payload(dict(verified_payload or {}))
        last_error: str | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return self._refine_payload_with_llm(
                    sql=sql,
                    schema=schema,
                    simplified_query=simplified_query,
                    draft_payload=draft,
                )
            except ValidationError as exc:
                last_error = self._format_llm_error(exc)
                if attempt == self.max_retries:
                    return {
                        "error": "LLM enhancement validation failed",
                        "details": last_error,
                    }
            except Exception as exc:
                if self._is_auth_error(exc):
                    return {
                        "error": "Hugging Face authentication failed for SQL2Graph extractor.",
                        "details": self._format_llm_error(exc),
                    }
                last_error = self._format_llm_error(exc)
                if not self._is_transient_error(exc) or attempt == self.max_retries:
                    break
                time.sleep(min(30, 2**attempt * 2))

        return {
            "error": "LLM enhancement failed",
            "details": last_error or "unknown error",
            "transient": bool(last_error and any(
                s in last_error.lower()
                for s in ("busy", "try again", "completion_error", "server_error", "overloaded")
            )),
        }

    def verify_and_enhance(
        self,
        sql: str,
        deterministic_draft: dict[str, Any],
        schema: dict[str, Any] | None = None,
        simplified_query: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run verification then optional enhancement."""
        verified = self.verify(
            sql=sql,
            deterministic_draft=deterministic_draft,
            schema=schema,
            simplified_query=simplified_query,
        )
        if "error" in verified:
            return verified
        return self.enhance(
            sql=sql,
            verified_payload=verified,
            schema=schema,
            simplified_query=simplified_query,
        )

    def extract(
        self,
        sql: str,
        schema: dict[str, Any] | None = None,
        simplified_query: dict[str, Any] | None = None,
        deterministic_draft: dict[str, Any] | None = None,
        *,
        use_cache: bool | None = None,
    ) -> dict[str, Any]:
        """Verify/enhance a sqlglot draft when available; otherwise cold-start extraction."""
        read_cache = getattr(self, "use_llm_cache", True) if use_cache is None else use_cache
        draft = deterministic_draft
        if draft is None and simplified_query and simplified_query.get("parser_used"):
            draft = SQL2GraphParser().build_deterministic_extraction(simplified_query)

        if draft:
            return self.verify_and_enhance(
                sql=sql,
                deterministic_draft=draft,
                schema=schema,
                simplified_query=simplified_query,
            )

        if read_cache:
            cached = self._cache_get(sql)
            if cached is not None:
                return cached

        last_validation_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                user_prompt = self._build_user_prompt(
                    sql=sql,
                    schema=schema,
                    simplified_query=simplified_query,
                    validation_error=last_validation_error,
                )
                validated = self._invoke_structured_extraction(
                    _chat_messages(self._llm_system_prompt, user_prompt)
                )
                payload = self._normalize_scope_payload(validated.model_dump())
                self._cache_set(sql, payload)
                return payload
            except ValidationError as exc:
                last_validation_error = str(exc)
                if attempt == self.max_retries:
                    return {
                        "error": "LLM output validation failed",
                        "details": last_validation_error,
                    }
            except Exception as exc:
                if self._is_auth_error(exc):
                    return {
                        "error": "Hugging Face authentication failed for SQL2Graph extractor.",
                        "details": str(exc),
                    }
                if attempt == self.max_retries:
                    return {"error": "SQL2Graph extraction failed", "details": str(exc)}
                time.sleep(min(10, 2**attempt))

        return {"error": "SQL2Graph extraction failed", "details": "unknown error"}

    def _cache_key(self, sql: str) -> str:
        from Classes.llm_cache import LLMCache

        return LLMCache.make_key(sql, prompt_version=self.prompt_version, model=self.model)

    def _cache_get(self, sql: str) -> dict[str, Any] | None:
        if self.cache is None:
            return None
        return self.cache.get(self._cache_key(sql))

    def _cache_set(
        self,
        sql: str,
        payload: dict[str, Any],
        *,
        quality_score: float = 0.0,
        replace_if_better: bool = False,
    ) -> dict[str, Any]:
        if self.cache is None or "error" in payload:
            return {"updated": False}
        cache_key = self._cache_key(sql)
        if replace_if_better:
            return self.cache.set_if_better(
                cache_key,
                payload,
                quality_score=quality_score,
                entry_type="extraction",
            )
        self.cache.set(cache_key, payload, quality_score=quality_score, entry_type="extraction")
        return {"updated": True, "quality_score": quality_score, "previous_quality_score": None}
