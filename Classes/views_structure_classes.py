import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from Classes.helper_classes import HuggingFaceLLMAdapter


class ViewOutputColumn(BaseModel):
    name: str
    expression: str = ""
    source_columns: List[str] = Field(default_factory=list)

    @field_validator("name")
    def validate_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("name cannot be empty")
        return cleaned


class ViewJoin(BaseModel):
    join_type: str = "INNER"
    left: str = ""
    right: str = ""
    condition: str = ""


class SourceTableStructure(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    full_name: str
    schema_name: str = Field(default="", alias="schema")
    table: str = ""
    columns_used: List[str] = Field(default_factory=list)
    join_conditions: List[str] = Field(default_factory=list)
    filter_references: List[str] = Field(default_factory=list)


class ViewStructure(BaseModel):
    view_name: str
    source_tables: List[str] = Field(default_factory=list)
    source_tables_structure: List[SourceTableStructure] = Field(default_factory=list)
    output_columns: List[ViewOutputColumn] = Field(default_factory=list)
    joins: List[ViewJoin] = Field(default_factory=list)
    filters: List[str] = Field(default_factory=list)
    ctes: List[str] = Field(default_factory=list)

    @field_validator("view_name")
    def validate_view_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("view_name cannot be empty")
        return cleaned


class ViewsStructureExtractor:
    """
    LLM-powered extractor of database structure from a CSV with columns:
    - table_name
    - view_def
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        provider: str = "scaleway",
        hf_token: Optional[str] = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        max_retries: int = 3,
        llm_pause_seconds: float = 0.0,
    ):
        if not hf_token:
            raise ValueError("HF_TOKEN is required for view-structure extraction.")

        self.model = model
        self.provider = provider
        self.max_retries = max_retries
        self.llm_pause_seconds = llm_pause_seconds

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
            "You are a database reverse-engineering assistant. "
            "Given a view name and SQL definition, return only JSON with keys: "
            "view_name, source_tables, output_columns, joins, filters, ctes. "
            "output_columns items must contain: name, expression, source_columns. "
            "source_columns must use fully qualified format schema.table.column whenever inferable; "
            "do not return alias.column in final output. "
            "joins items must contain: join_type, left, right, condition. "
            "Do not include markdown."
        )

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

    @staticmethod
    def _is_auth_error(error: Exception) -> bool:
        marker = str(error).lower()
        return any(s in marker for s in ["401", "unauthorized", "bad credentials", "forbidden"])

    @staticmethod
    def _normalize_text_list(values: Iterable[Any]) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for value in values:
            text = str(value).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(text)
        return normalized

    @staticmethod
    def _split_table_name(full_name: str) -> Dict[str, str]:
        parts = [p for p in str(full_name).split(".") if p]
        if len(parts) >= 2:
            return {"schema": ".".join(parts[:-1]), "table": parts[-1]}
        return {"schema": "", "table": str(full_name)}

    @staticmethod
    def _normalize_identifier(token: str) -> str:
        cleaned = str(token or "").strip().strip('"').strip()
        cleaned = cleaned.strip("()")
        cleaned = cleaned.rstrip(",")
        return cleaned

    @classmethod
    def _build_alias_map(cls, view_sql: str, source_tables: List[str]) -> Dict[str, str]:
        """
        Best-effort alias map: alias -> fully qualified table name.
        """
        alias_map: Dict[str, str] = {}
        source_lookup = {cls._normalize_identifier(table).lower(): table for table in source_tables}
        source_by_base = {
            cls._normalize_identifier(table).split(".")[-1].lower(): table for table in source_tables
        }

        pattern = re.compile(
            r"\b(?:from|join)\s+([A-Za-z_][\w\$\.\"']*)(?:\s+(?:as\s+)?([A-Za-z_][\w\$]*))?",
            flags=re.IGNORECASE,
        )
        reserved = {"where", "group", "order", "limit", "on", "left", "right", "inner", "full", "join"}

        for match in pattern.finditer(view_sql or ""):
            raw_table = cls._normalize_identifier(match.group(1))
            alias = cls._normalize_identifier(match.group(2))
            table_key = raw_table.lower()
            resolved = source_lookup.get(table_key) or source_by_base.get(raw_table.split(".")[-1].lower()) or raw_table

            if alias and alias.lower() not in reserved:
                alias_map[alias.lower()] = resolved
            else:
                base = raw_table.split(".")[-1]
                if base:
                    alias_map[base.lower()] = resolved
        return alias_map

    @classmethod
    def _qualify_source_column(cls, ref: str, alias_map: Dict[str, str]) -> str:
        text = str(ref).strip()
        # alias.column OR alias."column"
        match = re.match(r'^([A-Za-z_][\w\$]*)\.(?:"([^"]+)"|([A-Za-z_][\w\$]*))$', text)
        if not match:
            return text
        alias = match.group(1)
        column = match.group(2) or match.group(3)
        table_name = alias_map.get(alias.lower())
        if not table_name:
            return text
        return f"{table_name}.{column}"

    @classmethod
    def _build_source_tables_structure(cls, scope: Dict[str, Any]) -> List[Dict[str, Any]]:
        source_tables = scope.get("source_tables", []) or []
        joins = scope.get("joins", []) or []
        filters = scope.get("filters", []) or []
        output_columns = scope.get("output_columns", []) or []

        enriched: List[Dict[str, Any]] = []
        for table_name in source_tables:
            split = cls._split_table_name(str(table_name))
            table_lower = str(table_name).lower()
            item: Dict[str, Any] = {
                "full_name": str(table_name),
                "schema": split["schema"],
                "table": split["table"],
                "columns_used": [],
                "join_conditions": [],
                "filter_references": [],
            }

            for join in joins:
                condition = str(join.get("condition", "")).strip()
                left = str(join.get("left", "")).strip()
                right = str(join.get("right", "")).strip()
                combined = f"{left} {right} {condition}".lower()
                if condition and table_lower in combined and condition not in item["join_conditions"]:
                    item["join_conditions"].append(condition)

            for filt in filters:
                filter_text = str(filt).strip()
                if filter_text and table_lower in filter_text.lower() and filter_text not in item["filter_references"]:
                    item["filter_references"].append(filter_text)

            for out in output_columns:
                refs = out.get("source_columns", []) or []
                for ref in refs:
                    ref_text = str(ref).strip()
                    if ref_text.lower().startswith(table_lower + ".") and ref_text not in item["columns_used"]:
                        item["columns_used"].append(ref_text)

            enriched.append(item)
        return enriched

    @classmethod
    def _enrich_payload(cls, payload: Dict[str, Any], view_sql: str) -> Dict[str, Any]:
        normalized = dict(payload or {})
        normalized.setdefault("source_tables", [])
        normalized.setdefault("output_columns", [])
        normalized.setdefault("joins", [])
        normalized.setdefault("filters", [])
        normalized.setdefault("ctes", [])

        alias_map = cls._build_alias_map(view_sql=view_sql, source_tables=normalized.get("source_tables", []))
        for out in normalized.get("output_columns", []):
            refs = out.get("source_columns", []) or []
            qualified = [cls._qualify_source_column(ref, alias_map) for ref in refs]
            out["source_columns"] = cls._normalize_text_list(qualified)

        normalized["source_tables_structure"] = cls._build_source_tables_structure(normalized)
        return normalized

    @staticmethod
    def _regex_fallback(view_name: str, view_sql: str) -> Dict[str, Any]:
        """
        Lightweight deterministic fallback when LLM output is invalid.
        Extracts basic source tables from FROM/JOIN patterns.
        """
        source_tables = re.findall(
            r"(?:from|join)\s+([A-Za-z_][\w\$\.]*)",
            view_sql or "",
            flags=re.IGNORECASE,
        )
        source_tables = ViewsStructureExtractor._normalize_text_list(source_tables)

        filters = []
        where_match = re.search(r"\bwhere\b(.+?)(?:\bgroup\b|\border\b|\blimit\b|$)", view_sql, flags=re.IGNORECASE | re.DOTALL)
        if where_match:
            filters.append(where_match.group(1).strip())

        fallback = {
            "view_name": view_name,
            "source_tables": source_tables,
            "output_columns": [],
            "joins": [],
            "filters": filters,
            "ctes": [],
        }
        fallback["source_tables_structure"] = ViewsStructureExtractor._build_source_tables_structure(fallback)
        return fallback

    def _build_user_prompt(self, view_name: str, view_sql: str) -> str:
        return "\n".join(
            [
                f"View name: {view_name}",
                "",
                "View SQL:",
                view_sql,
                "",
                "Instructions:",
                "- Return strictly valid JSON only.",
                "- source_tables: base tables/views referenced in FROM/JOIN.",
                "- output_columns: each projected field with expression and source column references if inferable.",
                "- output_columns[].source_columns must be schema.table.column whenever mapping is possible.",
                "- Never keep alias.column in final source_columns if table mapping can be inferred.",
                "- joins: explicit joins with condition.",
                "- filters: WHERE/HAVING conditions as text list.",
                "- ctes: CTE aliases declared in WITH clause.",
            ]
        )

    def _invoke_messages_text(self, messages: List[Any]) -> str:
        if hasattr(self.chat_adapter, "invoke_messages"):
            return self.chat_adapter.invoke_messages(messages)
        if hasattr(self.chat_adapter, "invoke"):
            raw = self.chat_adapter.invoke(messages)
            if hasattr(raw, "content"):
                return str(raw.content)
            return str(raw)
        raw = self.chat_model.invoke(messages)
        if hasattr(raw, "content"):
            return str(raw.content)
        return str(raw)

    def extract_view_structure(self, view_name: str, view_sql: str) -> Dict[str, Any]:
        """Extract structure for one view definition."""
        prompt = self._build_user_prompt(view_name=view_name, view_sql=view_sql)
        last_validation_error: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                retry_hint = ""
                if last_validation_error:
                    retry_hint = f"\nPrevious response invalid: {last_validation_error}\nReturn corrected JSON only."

                response_text = self._invoke_messages_text(
                    [
                        SystemMessage(content=self.system_prompt),
                        HumanMessage(content=prompt + retry_hint),
                    ]
                )
                payload = self._extract_json(response_text)
                payload.setdefault("view_name", view_name)
                payload["source_tables"] = self._normalize_text_list(payload.get("source_tables", []))
                payload["filters"] = self._normalize_text_list(payload.get("filters", []))
                payload["ctes"] = self._normalize_text_list(payload.get("ctes", []))

                payload = self._enrich_payload(payload, view_sql=view_sql)
                validated = ViewStructure.model_validate(payload)
                return validated.model_dump(by_alias=True)
            except ValidationError as exc:
                last_validation_error = str(exc)
                if attempt == self.max_retries:
                    fallback = self._regex_fallback(view_name=view_name, view_sql=view_sql)
                    fallback["warning"] = "LLM output validation failed; returned regex fallback."
                    fallback["details"] = last_validation_error
                    return fallback
            except Exception as exc:
                if self._is_auth_error(exc):
                    return {
                        "error": "Hugging Face authentication failed for views extractor.",
                        "details": str(exc),
                        "view_name": view_name,
                    }
                if attempt == self.max_retries:
                    fallback = self._regex_fallback(view_name=view_name, view_sql=view_sql)
                    fallback["warning"] = "LLM extraction failed; returned regex fallback."
                    fallback["details"] = str(exc)
                    return fallback
                time.sleep(min(10, 2**attempt))

        return self._regex_fallback(view_name=view_name, view_sql=view_sql)

    @staticmethod
    def _iter_csv_rows(csv_path: str) -> Iterable[Dict[str, str]]:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield row

    def extract_from_csv(
        self,
        csv_path: str,
        limit: Optional[int] = None,
        include_tables: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        include_run_stats: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract structure for multiple views from data/views.csv.

        Args:
            csv_path: Path to csv file with table_name and view_def columns.
            limit: Maximum number of views to process.
            include_tables: Optional allowlist of view names.
            progress_callback: Optional callback invoked for each processed view.
            include_run_stats: Include per-view timing/status metadata in response.
        """
        path = Path(csv_path)
        if not path.exists():
            return {"error": f"CSV file not found: {csv_path}", "views": []}

        include_set = {name.lower().strip() for name in (include_tables or []) if name and name.strip()}
        extracted_views: List[Dict[str, Any]] = []
        run_stats: List[Dict[str, Any]] = []
        processed = 0

        for row in self._iter_csv_rows(str(path)):
            view_name = (row.get("table_name") or "").strip()
            view_sql = (row.get("view_def") or "").strip()
            if not view_name or not view_sql:
                continue

            if include_set and view_name.lower() not in include_set:
                continue

            cycle_start = time.perf_counter()
            extracted = self.extract_view_structure(view_name=view_name, view_sql=view_sql)
            elapsed_ms = int((time.perf_counter() - cycle_start) * 1000)

            status = "ok"
            if "error" in extracted:
                status = "error"
            elif "warning" in extracted:
                status = "warning"

            cycle_log = {
                "index": processed + 1,
                "view_name": view_name,
                "status": status,
                "elapsed_ms": elapsed_ms,
                "source_tables_count": len(extracted.get("source_tables", []) or []),
                "output_columns_count": len(extracted.get("output_columns", []) or []),
            }

            if "warning" in extracted:
                cycle_log["warning"] = extracted.get("warning")
            if "error" in extracted:
                cycle_log["error"] = extracted.get("error")

            if progress_callback:
                progress_callback(cycle_log)

            if include_run_stats:
                run_stats.append(cycle_log)

            extracted_views.append(extracted)
            processed += 1

            if self.llm_pause_seconds > 0:
                time.sleep(self.llm_pause_seconds)

            if limit is not None and processed >= limit:
                break

        response = {
            "csv_path": str(path),
            "views_count": len(extracted_views),
            "views": extracted_views,
        }
        if include_run_stats:
            response["run_stats"] = run_stats
        return response
