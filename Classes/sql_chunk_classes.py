"""
LLM-powered SQL logical chunk parser.

Decomposes complicated SQL into a small set of logical chunks (CTEs, query bodies,
UNION branches) and links between them (JOIN conditions, UNION, INSERT).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, ValidationError, field_validator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from Classes.helper_classes import HuggingFaceLLMAdapter, resolve_model_name, resolve_provider
from Classes.sql2graph_classes import SQL2GraphParser

try:
    import sqlglot  # type: ignore[import-not-found]
    from sqlglot import exp  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    sqlglot = None
    exp = None


VALID_CHUNK_TYPES = frozenset({"cte", "query", "target"})
VALID_LINK_TYPES = frozenset(
    {"JOIN", "UNION", "UNION ALL", "UNION DISTINCT", "INSERT", "INTERSECT", "EXCEPT"}
)

WEAK_SQL_PATTERN = re.compile(
    r"(\.\.\.|AS\s+\.\.\.|\(\s*\.\.\.\s*\)|<\s*truncated\s*>|<\s*sql\s*>)",
    re.IGNORECASE,
)

ALIAS_COLUMN_PATTERN = re.compile(
    r'([A-Za-z_][\w\$]*)\.(?:"([^"]+)"|([A-Za-z_][\w\$]*))'
)


class SQLChunkLink(BaseModel):
    source: str
    target: str
    link_type: str = "JOIN"
    condition: str = ""

    @field_validator("link_type")
    def validate_link_type(cls, value: str) -> str:
        normalized = (value or "JOIN").strip().upper()
        if normalized == "UNIONALL":
            normalized = "UNION ALL"
        if normalized not in VALID_LINK_TYPES:
            raise ValueError(f"link_type must be one of {sorted(VALID_LINK_TYPES)}")
        return normalized


# Backward-compatible alias
SQLChunkEdge = SQLChunkLink


class SQLChunk(BaseModel):
    id: str
    name: str
    chunk_type: str = "query"
    sql: str = ""

    @field_validator("chunk_type")
    def validate_chunk_type(cls, value: str) -> str:
        normalized = (value or "query").strip().lower()
        if normalized not in VALID_CHUNK_TYPES:
            raise ValueError(f"chunk_type must be one of {sorted(VALID_CHUNK_TYPES)}")
        return normalized

    @field_validator("id", "name")
    def validate_non_empty(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("id and name cannot be empty")
        return cleaned


class SQLChunkGraph(BaseModel):
    chunks: List[SQLChunk]
    links: List[SQLChunkLink]
    statement_type: str = "select"
    target_table: Optional[str] = None


class SQLLogicalChunkPreParser:
    """Deterministic extraction of major SQL chunks and links between them."""

    UNION_OPERATOR_PATTERN = re.compile(
        r"\b(UNION\s+ALL|UNION\s+DISTINCT|UNION|INTERSECT|EXCEPT)\b",
        re.IGNORECASE,
    )

    def __init__(self, parser: Optional[SQL2GraphParser] = None):
        self.parser = parser or SQL2GraphParser()

    @staticmethod
    def _chunk(name: str, sql: str, chunk_type: str = "query") -> Dict[str, Any]:
        return {
            "id": name,
            "name": name,
            "chunk_type": chunk_type,
            "sql": (sql or "").strip(),
        }

    @staticmethod
    def _append_link(
        links: List[Dict[str, str]],
        seen: Set[Tuple[str, str, str]],
        source: str,
        target: str,
        link_type: str = "JOIN",
        condition: str = "",
    ) -> None:
        if not source or not target or source == target:
            return
        key = (source, target, link_type.upper())
        if key in seen:
            return
        seen.add(key)
        links.append(
            {
                "source": source,
                "target": target,
                "link_type": link_type.upper(),
                "condition": (condition or "").strip(),
            }
        )

    @staticmethod
    def _expression_sql(expression: Any, dialect: Optional[str]) -> str:
        if expression is None:
            return ""
        try:
            return expression.sql(dialect=dialect)
        except Exception:
            return str(expression)

    def _extract_main_query_sql(self, sql: str, dialect: Optional[str]) -> str:
        if not sqlglot:
            return sql.strip()
        tree = sqlglot.parse_one(sql, read=dialect)

        def _select_without_cte(select_node: Any) -> str:
            if select_node is None:
                return ""
            if isinstance(select_node, exp.Select) and select_node.args.get("with_"):
                stripped = select_node.copy()
                stripped.set("with_", None)
                return self._expression_sql(stripped, dialect)
            return self._expression_sql(select_node, dialect)

        if isinstance(tree, exp.Insert):
            return _select_without_cte(tree.find(exp.Select))
        if isinstance(tree, exp.Create):
            return _select_without_cte(tree.find(exp.Select))
        if isinstance(tree, exp.Select):
            return _select_without_cte(tree)
        with_node = tree.find(exp.With)
        if with_node is not None and tree.this is not None:
            return _select_without_cte(tree.this)
        if isinstance(tree, (exp.Select, exp.Union)):
            return self._expression_sql(tree, dialect)
        return sql.strip()

    @staticmethod
    def _build_alias_map(
        simplified: Dict[str, Any],
        cte_names: Dict[str, str],
    ) -> Dict[str, str]:
        alias_map: Dict[str, str] = {}

        for table in simplified.get("from") or []:
            alias = str(table.get("alias") or "").strip()
            table_name = str(table.get("table") or alias).strip()
            if alias:
                resolved = cte_names.get(table_name.lower()) or cte_names.get(alias.lower()) or table_name
                alias_map[alias.lower()] = resolved

        for join in simplified.get("joins") or []:
            alias = str(join.get("alias") or "").strip()
            right_table = str(join.get("right_table") or alias).strip()
            if alias:
                resolved = cte_names.get(right_table.lower()) or cte_names.get(alias.lower()) or right_table
                alias_map[alias.lower()] = resolved

        return alias_map

    @classmethod
    def _normalize_join_condition(cls, condition: str, alias_map: Dict[str, str]) -> str:
        if not condition or not alias_map:
            return (condition or "").strip()

        def repl(match: re.Match[str]) -> str:
            alias = match.group(1)
            column = match.group(2) or match.group(3)
            resolved = alias_map.get(alias.lower(), alias)
            return f"{resolved}.{column}"

        return ALIAS_COLUMN_PATTERN.sub(repl, condition.strip())

    @classmethod
    def _detect_union_operator(cls, sql: str) -> str:
        match = cls.UNION_OPERATOR_PATTERN.search(sql or "")
        if not match:
            return "UNION ALL"
        op = re.sub(r"\s+", " ", match.group(1).upper()).strip()
        return op if op in VALID_LINK_TYPES else "UNION ALL"

    @staticmethod
    def _union_branch_id(block: Dict[str, Any], index: int) -> str:
        name = str(block.get("name") or "").strip()
        if name:
            return name
        return f"branch_{index}"

    def preparse(self, sql: str, dialect: Optional[str] = None) -> Dict[str, Any]:
        """Build major logical chunks and links without calling an LLM."""
        simplified = self.parser.simplify(sql, dialect=dialect)
        chunks: List[Dict[str, Any]] = []
        links: List[Dict[str, str]] = []
        seen_links: Set[Tuple[str, str, str]] = set()
        chunk_ids: Set[str] = set()

        statement_type = simplified.get("statement_type") or "select"
        target_table = simplified.get("target_table")
        union_blocks = [
            block
            for block in (simplified.get("subgraph_blocks") or [])
            if block.get("type") == "union_block" and str(block.get("sql") or "").strip()
        ]

        cte_names: Dict[str, str] = {}
        for index, cte in enumerate(simplified.get("ctes") or []):
            name = str(cte.get("alias") or f"cte_{index}").strip()
            body = str(cte.get("query") or "").strip()
            if not body:
                continue
            cte_names[name.lower()] = name
            chunks.append(self._chunk(name, body, chunk_type="cte"))
            chunk_ids.add(name)

        if target_table:
            chunks.append(self._chunk(target_table, target_table, chunk_type="target"))
            chunk_ids.add(target_table)

        if len(union_blocks) >= 2:
            branch_ids: List[str] = []
            for index, block in enumerate(union_blocks):
                branch_id = self._union_branch_id(block, index)
                if branch_id in chunk_ids:
                    branch_id = f"{branch_id}_{index}"
                branch_ids.append(branch_id)
                chunks.append(self._chunk(branch_id, str(block.get("sql") or ""), chunk_type="query"))
                chunk_ids.add(branch_id)

            union_op = self._detect_union_operator(sql)
            for left, right in zip(branch_ids, branch_ids[1:]):
                self._append_link(links, seen_links, left, right, union_op, "")

            if target_table:
                for branch_id in branch_ids:
                    self._append_link(links, seen_links, branch_id, target_table, "INSERT", "")
        else:
            main_sql = self._extract_main_query_sql(sql, dialect).strip()
            main_id = "main"
            if main_sql:
                chunks.append(self._chunk(main_id, main_sql, chunk_type="query"))
                chunk_ids.add(main_id)

            alias_map = self._build_alias_map(simplified, cte_names)
            for join in simplified.get("deterministic_joins") or []:
                right_alias = str(join.get("right_alias") or "").strip()
                if not right_alias:
                    continue
                resolved = alias_map.get(right_alias.lower(), right_alias)
                target_chunk = cte_names.get(resolved.lower()) or (
                    resolved if resolved in chunk_ids else None
                )
                if not target_chunk or not main_id in chunk_ids:
                    continue
                condition = self._normalize_join_condition(str(join.get("condition") or ""), alias_map)
                join_type = str(join.get("type") or "INNER").upper()
                self._append_link(
                    links,
                    seen_links,
                    main_id,
                    target_chunk,
                    "JOIN",
                    condition,
                )
                if join_type != "INNER":
                    links[-1]["condition"] = f"{join_type} {links[-1]['condition']}".strip()

            if target_table and main_id in chunk_ids:
                self._append_link(links, seen_links, main_id, target_table, "INSERT", "")

        return {
            "statement_type": statement_type,
            "target_table": target_table,
            "chunks": chunks,
            "links": links,
            "seed_source": "sqlglot" if simplified.get("parser_used") else "raw",
            "simplified_query": simplified,
        }


class SQLLogicalChunkParser:
    """
    Parse complicated SQL into logical chunks and links between them.

    Primary result shape:
        {"chunks": [...], "links": [...]}
    """

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        hf_token: Optional[str] = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        max_retries: int = 3,
        pre_parser: Optional[SQLLogicalChunkPreParser] = None,
    ):
        if not hf_token:
            raise ValueError("HF_TOKEN is required for SQL logical chunk parsing.")

        model = resolve_model_name(model)
        provider = resolve_provider(provider)
        self.model = model
        self.provider = provider
        self.max_retries = max_retries
        self.pre_parser = pre_parser or SQLLogicalChunkPreParser()

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
            "You are a SQL structure analyst. Return ONLY valid JSON with keys: chunks, links. "
            "Each chunk must have: id, name, chunk_type, sql. "
            "chunk_type must be one of: cte, query, target. "
            "Each link must have: source, target, link_type, condition. "
            "link_type must be one of: JOIN, UNION, UNION ALL, UNION DISTINCT, INSERT, INTERSECT, EXCEPT. "
            "Create one chunk per CTE body, one chunk per major query/UNION branch, and optional target chunk. "
            "Do not emit granular chunks (filters, select items, from scans). "
            "Links must describe how chunks connect, e.g. JOIN with condition customers.id = recent_orders.customer_id."
        )

    @staticmethod
    def _is_weak_sql(text: Any) -> bool:
        value = str(text or "").strip()
        if not value:
            return True
        return bool(WEAK_SQL_PATTERN.search(value))

    @classmethod
    def _pick_sql(cls, *candidates: Any) -> str:
        cleaned = [str(item or "").strip() for item in candidates if str(item or "").strip()]
        if not cleaned:
            return ""
        strong = [item for item in cleaned if not cls._is_weak_sql(item)]
        pool = strong or cleaned
        return max(pool, key=len)

    @classmethod
    def _sync_chunk(cls, chunk: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(chunk or {})
        item["sql"] = cls._pick_sql(item.get("sql"), item.get("code"))
        item.pop("code", None)
        item["id"] = str(item.get("id") or item.get("name") or "chunk").strip()
        item["name"] = str(item.get("name") or item["id"]).strip()
        item["chunk_type"] = str(item.get("chunk_type") or "query").strip().lower()
        return item

    @classmethod
    def collect_chunk_sql(cls, chunks: List[Dict[str, Any]]) -> Dict[str, str]:
        return {
            str(chunk.get("id")): cls._pick_sql(chunk.get("sql"))
            for chunk in (chunks or [])
            if chunk.get("id") and cls._pick_sql(chunk.get("sql"))
        }

    @classmethod
    def _normalize_payload(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(payload or {})
        normalized["chunks"] = [cls._sync_chunk(item) for item in normalized.get("chunks") or []]
        chunk_ids = {chunk["id"] for chunk in normalized["chunks"]}

        fixed_links: List[Dict[str, Any]] = []
        raw_links = normalized.get("links")
        if raw_links is None:
            raw_links = normalized.get("edges") or []

        for item in raw_links or []:
            link = dict(item or {})
            source = str(link.get("source") or "").strip()
            target = str(link.get("target") or "").strip()
            if not source or not target:
                continue
            if source not in chunk_ids or target not in chunk_ids:
                continue
            link_type = str(link.get("link_type") or link.get("edge_type") or "JOIN").strip().upper()
            if link_type == "UNIONALL":
                link_type = "UNION ALL"
            fixed_links.append(
                {
                    "source": source,
                    "target": target,
                    "link_type": link_type,
                    "condition": str(link.get("condition") or link.get("label") or "").strip(),
                }
            )
        normalized["links"] = fixed_links
        normalized.pop("edges", None)
        return normalized

    @classmethod
    def merge_seed_with_llm(cls, seed: Dict[str, Any], llm_payload: Dict[str, Any]) -> Dict[str, Any]:
        seed = cls._normalize_payload(seed)
        llm_payload = cls._normalize_payload(llm_payload)

        by_id: Dict[str, Dict[str, Any]] = {chunk["id"]: chunk for chunk in llm_payload.get("chunks") or []}
        for chunk in seed.get("chunks") or []:
            existing = by_id.get(chunk["id"], {})
            merged = dict(existing)
            merged["id"] = chunk["id"]
            merged["name"] = chunk.get("name") or chunk["id"]
            merged["chunk_type"] = chunk.get("chunk_type") or existing.get("chunk_type") or "query"
            merged["sql"] = cls._pick_sql(
                chunk.get("sql"),
                chunk.get("code"),
                existing.get("sql"),
                existing.get("code"),
            )
            merged.pop("code", None)
            by_id[chunk["id"]] = merged

        for chunk in llm_payload.get("chunks") or []:
            if chunk["id"] not in by_id:
                by_id[chunk["id"]] = chunk

        link_key = lambda link: (link["source"], link["target"], link.get("link_type", "JOIN"))
        links: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for link in (seed.get("links") or []) + (llm_payload.get("links") or []):
            links[link_key(link)] = link

        return {
            "statement_type": llm_payload.get("statement_type") or seed.get("statement_type") or "select",
            "target_table": llm_payload.get("target_table") or seed.get("target_table"),
            "chunks": list(by_id.values()),
            "links": list(links.values()),
        }

    @staticmethod
    def _build_metadata(sql: str, seed_source: str, llm_enriched: bool = False) -> Dict[str, Any]:
        metadata = {
            "source_sql_hash": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed_source": seed_source,
        }
        if llm_enriched:
            metadata["llm_enriched"] = True
        return metadata

    @staticmethod
    def _connectivity_warnings(graph: SQLChunkGraph) -> List[str]:
        if not graph.chunks:
            return ["Graph has no chunks."]
        if len(graph.chunks) == 1:
            return []

        ids = {chunk.id for chunk in graph.chunks}
        adjacency: Dict[str, Set[str]] = {node_id: set() for node_id in ids}
        for link in graph.links:
            adjacency[link.source].add(link.target)
            adjacency[link.target].add(link.source)

        start = graph.chunks[0].id
        visited: Set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(adjacency.get(node, set()) - visited)

        warnings: List[str] = []
        unreachable = ids - visited
        if unreachable:
            warnings.append(f"Disconnected chunks: {sorted(unreachable)}")
        if not graph.links and len(graph.chunks) > 1:
            warnings.append("Multiple chunks but no links.")
        return warnings

    def _build_user_prompt(
        self,
        sql: str,
        seed: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
        validation_error: Optional[str] = None,
    ) -> str:
        parts = [
            "SQL:",
            sql,
            "",
            "Deterministic seed (preserve chunk ids and sql where valid):",
            json.dumps({"chunks": seed.get("chunks"), "links": seed.get("links")}, indent=2),
            "",
            "Schema JSON (optional):",
            json.dumps(schema or {}, indent=2),
            "",
            "Tasks:",
            "- Keep only major chunks: CTE bodies, main query / UNION branches, optional INSERT target.",
            "- Keep links as JOIN / UNION / INSERT with condition text on JOIN links.",
            "- Fill sql with verbatim SQL for each chunk.",
        ]
        if validation_error:
            parts.extend(["", "Previous output failed validation:", validation_error, "Fix and return corrected JSON only."])
        return "\n".join(parts)

    def _finalize_result(
        self,
        payload: Dict[str, Any],
        sql: str,
        seed_source: str,
        llm_enriched: bool = False,
    ) -> Dict[str, Any]:
        normalized = self._normalize_payload(payload)
        validated = SQLChunkGraph.model_validate(normalized)
        warnings = self._connectivity_warnings(validated)
        result = validated.model_dump()
        return {
            "chunks": result["chunks"],
            "links": result["links"],
            "metadata": self._build_metadata(sql, seed_source, llm_enriched=llm_enriched),
            "warnings": warnings,
        }

    def preparse(self, sql: str, dialect: Optional[str] = None) -> Dict[str, Any]:
        """Deterministic chunks + links without LLM calls."""
        seed = self.pre_parser.preparse(sql, dialect=dialect)
        return self._finalize_result(seed, sql=sql, seed_source=seed.get("seed_source", "raw"))

    def parse(
        self,
        sql: str,
        dialect: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """Parse SQL into chunks and links. Set use_llm=False for deterministic output only."""
        seed = self.pre_parser.preparse(sql, dialect=dialect)
        if not use_llm:
            return self.preparse(sql, dialect=dialect)

        last_validation_error: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                user_prompt = self._build_user_prompt(
                    sql=sql,
                    seed=seed,
                    schema=schema,
                    validation_error=last_validation_error,
                )
                response_text = self._invoke_messages_text(
                    [SystemMessage(content=self.system_prompt), HumanMessage(content=user_prompt)]
                )
                llm_payload = self._normalize_payload(self._extract_json(response_text))
                merged = self.merge_seed_with_llm(seed, llm_payload)
                return self._finalize_result(
                    merged,
                    sql=sql,
                    seed_source=seed.get("seed_source", "raw"),
                    llm_enriched=True,
                )
            except ValidationError as exc:
                last_validation_error = str(exc)
                if attempt == self.max_retries:
                    fallback = self.preparse(sql, dialect=dialect)
                    fallback["error"] = "LLM chunk graph validation failed; returned deterministic preparse."
                    fallback["details"] = last_validation_error
                    return fallback
            except Exception as exc:
                if self._is_auth_error(exc):
                    return {
                        "error": "Hugging Face authentication failed for SQL logical chunk parser.",
                        "details": str(exc),
                    }
                if attempt == self.max_retries:
                    fallback = self.preparse(sql, dialect=dialect)
                    fallback["error"] = "SQL logical chunk parsing failed; returned deterministic preparse."
                    fallback["details"] = str(exc)
                    return fallback
                time.sleep(min(10, 2**attempt))

        fallback = self.preparse(sql, dialect=dialect)
        fallback["error"] = "SQL logical chunk parsing failed; returned deterministic preparse."
        return fallback

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
    def _response_to_text(response: Any) -> str:
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)

    def _invoke_messages_text(self, messages: List[Any]) -> str:
        if hasattr(self.chat_adapter, "invoke_messages"):
            return self.chat_adapter.invoke_messages(messages)
        if hasattr(self.chat_adapter, "invoke"):
            return self._response_to_text(self.chat_adapter.invoke(messages))
        return self._response_to_text(self.chat_model.invoke(messages))

    def to_node_link(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert chunks/links result to node-link JSON for visualization."""
        chunks = result.get("chunks") or result.get("graph", {}).get("chunks", [])
        links = result.get("links") or result.get("graph", {}).get("links", [])
        nodes = [
            {
                "id": chunk["id"],
                "label": chunk.get("name") or chunk["id"],
                "chunk_type": chunk.get("chunk_type", "query"),
                "sql": chunk.get("sql") or "",
            }
            for chunk in chunks
        ]
        edges = [
            {
                "source": link["source"],
                "target": link["target"],
                "link_type": link.get("link_type", "JOIN"),
                "condition": link.get("condition", ""),
            }
            for link in links
        ]
        return {"nodes": nodes, "links": edges}
