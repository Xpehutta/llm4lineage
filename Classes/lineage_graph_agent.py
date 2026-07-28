"""
LLM agent: enrich sqlglot parse output into a connected chunk graph for visualization.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from Classes.helper_classes import HuggingFaceLLMAdapter, resolve_model_name, resolve_provider
from Classes.pipeline.llm_helpers import create_chat_model, resolve_hf_token
from Classes.sql_chunk_classes import SQLChunkGraph, SQLLogicalChunkParser


class LineageGraphAgent:
    """
    Analyze sqlglot parse JSON and produce a validated chunk/link graph for the drawer.

    Pipeline:
        1. Accept deterministic ``chunks`` / ``simplify`` / ``ast_summary`` context
        2. LLM identifies missing links and corrects chunk boundaries
        3. Merge with deterministic seed and validate via ``SQLChunkGraph``
    """

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        hf_token: Optional[str] = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        max_retries: int = 3,
        chunk_parser: Optional[SQLLogicalChunkParser] = None,
    ):
        model = resolve_model_name(model)
        provider = resolve_provider(provider)
        self.model = model
        self.provider = provider
        self.max_retries = max_retries
        self.chunk_parser = chunk_parser or SQLLogicalChunkParser(
            model=model,
            provider=provider,
            hf_token=hf_token,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        self.chat_model = None
        self.chat_adapter = None

        if resolve_hf_token(hf_token):
            self.chat_model = create_chat_model(
                model=model,
                provider=provider,
                hf_token=hf_token,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
            self.chat_adapter = HuggingFaceLLMAdapter(self.chat_model)

        self.system_prompt = (
            "You are a SQL lineage graph agent. "
            "Given sqlglot parse context and a deterministic chunk split, produce JSON for graph visualization. "
            "Return ONLY valid JSON with keys: chunks, links. "
            "Each chunk: id, name, chunk_type (cte|query|target), sql. "
            "Each link: source, target, link_type, condition. "
            "link_type must be one of: JOIN, UNION, UNION ALL, UNION DISTINCT, INSERT, INTERSECT, EXCEPT. "
            "Connect all chunks into a coherent lineage graph: "
            "JOIN links from query chunks to CTEs/tables they reference, "
            "UNION ALL links between union branches inside CTEs, "
            "INSERT links from query branches to the insert target. "
            "Use verbatim SQL substrings from the source SQL for chunk sql. "
            "Preserve deterministic chunk ids when possible. "
            "Do not invent granular nodes (single columns, filters, or bare table scans)."
        )

    @staticmethod
    def _compact_simplify(simplify: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not simplify:
            return {}
        ctes = simplify.get("ctes") or []
        blocks = simplify.get("subgraph_blocks") or []
        return {
            "parser_used": simplify.get("parser_used"),
            "statement_type": simplify.get("statement_type"),
            "target_table": simplify.get("target_table"),
            "from": simplify.get("from"),
            "joins": simplify.get("joins"),
            "cte_aliases": [item.get("alias") for item in ctes],
            "subgraph_blocks": [
                {
                    "id": block.get("id"),
                    "type": block.get("type"),
                    "name": block.get("name"),
                }
                for block in blocks
            ],
            "deterministic_filters": simplify.get("deterministic_filters"),
            "deterministic_joins": simplify.get("deterministic_joins"),
            "select_aliases": (simplify.get("select") or {}).get("aliases"),
        }

    @staticmethod
    def _compact_ast(ast_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not ast_summary:
            return {}
        return {
            "tables": ast_summary.get("tables"),
            "joins": ast_summary.get("joins"),
            "union_nodes": ast_summary.get("union_nodes"),
            "cte_names": ast_summary.get("cte_names"),
            "column_ref_count": len(ast_summary.get("column_refs") or []),
        }

    @staticmethod
    def _compact_chunks(chunks: List[Dict[str, Any]], sql_limit: int = 400) -> List[Dict[str, Any]]:
        compact: List[Dict[str, Any]] = []
        for chunk in chunks or []:
            sql_text = str(chunk.get("sql") or "")
            if len(sql_text) > sql_limit:
                sql_text = sql_text[:sql_limit] + "..."
            compact.append(
                {
                    "id": chunk.get("id"),
                    "name": chunk.get("name"),
                    "chunk_type": chunk.get("chunk_type"),
                    "sql_preview": sql_text,
                }
            )
        return compact

    def _build_prompt(
        self,
        sql: str,
        deterministic: Dict[str, Any],
        simplify: Optional[Dict[str, Any]] = None,
        ast_summary: Optional[Dict[str, Any]] = None,
        validation_error: Optional[str] = None,
    ) -> str:
        warnings = deterministic.get("warnings") or []
        parts = [
            "SQL:",
            sql,
            "",
            "Sqlglot simplify summary:",
            json.dumps(self._compact_simplify(simplify), indent=2),
            "",
            "AST summary:",
            json.dumps(self._compact_ast(ast_summary), indent=2),
            "",
            "Deterministic chunks (step 1):",
            json.dumps(
                {
                    "statement_type": deterministic.get("statement_type"),
                    "target_table": deterministic.get("target_table"),
                    "chunks": self._compact_chunks(deterministic.get("chunks") or []),
                    "links": deterministic.get("links") or [],
                    "warnings": warnings,
                },
                indent=2,
            ),
            "",
            "Tasks:",
            "- Identify how chunks connect (JOIN / UNION ALL / INSERT).",
            "- Add missing links so the graph is connected for visualization.",
            "- Fix chunk sql using verbatim substrings from the SQL when needed.",
            "- Return ONLY JSON: {chunks, links}.",
        ]
        if validation_error:
            parts.extend(["", "Validation error from previous attempt:", validation_error])
        return "\n".join(parts)

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        return SQLLogicalChunkParser._extract_json(text)

    @staticmethod
    def _is_auth_error(error: Exception) -> bool:
        return SQLLogicalChunkParser._is_auth_error(error)

    def _invoke_messages_text(self, messages: List[Any]) -> str:
        if self.chat_adapter is not None:
            if hasattr(self.chat_adapter, "invoke_messages"):
                return self.chat_adapter.invoke_messages(messages)
            if hasattr(self.chat_adapter, "invoke"):
                return SQLLogicalChunkParser._response_to_text(self.chat_adapter.invoke(messages))
        if self.chat_model is not None:
            return SQLLogicalChunkParser._response_to_text(self.chat_model.invoke(messages))
        if self.chunk_parser.chat_adapter is not None or self.chunk_parser.chat_model is not None:
            return self.chunk_parser._invoke_messages_text(messages)
        raise ValueError("No LLM client configured. Set HF_TOKEN for LineageGraphAgent.")

    def build_graph(
        self,
        sql: str,
        chunk_result: Dict[str, Any],
        simplify: Optional[Dict[str, Any]] = None,
        ast_summary: Optional[Dict[str, Any]] = None,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the LLM agent and return a drawer-ready graph payload.

        ``chunk_result`` is the deterministic preparse output (chunks + links).
        """
        if not self.chat_adapter and not self.chat_model:
            raise ValueError("HF_TOKEN is required for LineageGraphAgent.build_graph().")

        deterministic = {
            "chunks": chunk_result.get("chunks") or [],
            "links": chunk_result.get("links") or [],
            "statement_type": chunk_result.get("statement_type"),
            "target_table": chunk_result.get("target_table"),
            "warnings": chunk_result.get("warnings") or [],
        }

        last_validation_error: Optional[str] = None
        parser = self.chunk_parser

        for attempt in range(1, self.max_retries + 1):
            try:
                user_prompt = self._build_prompt(
                    sql=sql,
                    deterministic=deterministic,
                    simplify=simplify,
                    ast_summary=ast_summary,
                    validation_error=last_validation_error,
                )
                response_text = self._invoke_messages_text(
                    [SystemMessage(content=self.system_prompt), HumanMessage(content=user_prompt)]
                )
                llm_payload = parser._normalize_payload(self._extract_json(response_text))
                merged = parser.merge_deterministic_with_corrections(deterministic, llm_payload)
                SQLChunkGraph.model_validate(
                    {
                        **parser._normalize_payload(merged),
                        "statement_type": merged.get("statement_type") or "select",
                        "target_table": merged.get("target_table"),
                    }
                )
                return parser._finalize_result(
                    merged,
                    sql=sql,
                    seed_source="sqlglot+llm_graph_agent",
                    llm_enriched=True,
                    pipeline_stage="llm_graph_agent",
                )
            except ValidationError as exc:
                last_validation_error = str(exc)
                if attempt == self.max_retries:
                    raise
            except Exception as exc:
                if self._is_auth_error(exc):
                    raise
                if attempt == self.max_retries:
                    raise
                time.sleep(min(10, 2 ** attempt))

        raise RuntimeError("Lineage graph agent failed after retries.")

    def build_graph_from_snapshot(self, snapshot: Dict[str, Any], sql: Optional[str] = None) -> Dict[str, Any]:
        """Build graph JSON from a saved parse snapshot (e.g. sqlglot_ddls10_first_snapshot.json)."""
        chunk_payload = snapshot.get("chunks") or {}
        if not isinstance(chunk_payload, dict):
            raise ValueError("Snapshot must contain a 'chunks' object with preparse output.")

        source_sql = sql or snapshot.get("sql") or ""
        if not source_sql and snapshot.get("source_file"):
            raise ValueError("Provide sql=... or include 'sql' in the snapshot.")

        return self.build_graph(
            sql=source_sql,
            chunk_result=chunk_payload,
            simplify=snapshot.get("simplify"),
            ast_summary=snapshot.get("ast_summary"),
        )
