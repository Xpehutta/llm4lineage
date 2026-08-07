"""Five-step SQL2Graph pipeline coordinator."""
from __future__ import annotations

import copy
import hashlib
import logging
import re
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph
from pydantic import ValidationError

from Classes.llm_cache import LLMCache
from Classes.sql2graph.builder import SQL2GraphBuilder
from Classes.sql2graph.llm_extractor import SQL2GraphLLMExtractor
from Classes.sql2graph.models import SQL2GraphExtraction
from Classes.sql2graph.parser import SQL2GraphParser
from Classes.sql2graph.validator import SQL2GraphValidator

logger = logging.getLogger(__name__)

def pipeline_result_quality(
    *,
    pipeline_stage: str,
    extraction: dict[str, Any],
    graph: dict[str, Any] | None = None,
    golden_f1: float | None = None,
) -> float:
    """Heuristic quality score for comparing cached vs fresh pipeline runs."""
    stage_scores = {
        "llm_enhanced": 100.0,
        "llm_verified": 80.0,
        "llm_parse_fallback": 70.0,
        "deterministic": 50.0,
        "deterministic_fallback": 30.0,
    }
    score = stage_scores.get(pipeline_stage, 40.0)
    if golden_f1 is not None:
        score += golden_f1 * 1000.0
    links = len((graph or {}).get("links") or [])
    outputs = extraction.get("output_columns") or []
    deps = sum(len(col.get("dependencies") or []) for col in outputs)
    score += links * 0.01 + deps * 0.1 + len(outputs) * 0.05
    return score


class SQL2GraphPipeline:
    """End-to-end SQL-to-column-lineage graph pipeline.

    Stages:
      1. chunking  — split SQL into logical chunks (CTEs, UNION branches, target)
      2. parsing   — sqlglot simplify + deterministic column extraction
      3. verifying — optional LLM review of the sqlglot draft
      4. enhancing — optional LLM targeted repairs on the verified draft
      5. combining — merge extraction, build lineage graph, validate
    """

    PIPELINE_STEP_ORDER = ("chunking", "parsing", "verifying", "enhancing", "combining")

    def __init__(
        self,
        llm_extractor: SQL2GraphLLMExtractor | None = None,
        parser: SQL2GraphParser | None = None,
        builder: SQL2GraphBuilder | None = None,
        validator: SQL2GraphValidator | None = None,
        chunk_parser: Any | None = None,
    ):
        self.llm_extractor = llm_extractor
        self.parser = parser or SQL2GraphParser()
        self.builder = builder or SQL2GraphBuilder()
        self.validator = validator or SQL2GraphValidator()
        self.chunk_parser = chunk_parser

    def _pipeline_cache_key(
        self,
        sql: str,
        *,
        dialect: str,
        use_llm_verify: bool,
        use_llm_enhance: bool,
    ) -> str | None:
        if self.llm_extractor is None or getattr(self.llm_extractor, "cache", None) is None:
            return None

        return LLMCache.make_pipeline_key(
            sql,
            prompt_version=self.llm_extractor.prompt_version,
            model=self.llm_extractor.model,
            dialect=dialect or self.parser.dialect,
            use_llm_verify=use_llm_verify,
            use_llm_enhance=use_llm_enhance,
        )

    def _load_pipeline_cache(
        self,
        sql: str,
        *,
        dialect: str,
        use_llm_verify: bool,
        use_llm_enhance: bool,
    ) -> dict[str, Any] | None:
        cache_key = self._pipeline_cache_key(
            sql,
            dialect=dialect,
            use_llm_verify=use_llm_verify,
            use_llm_enhance=use_llm_enhance,
        )
        if cache_key is None:
            return None
        return self.llm_extractor.cache.get_entry(cache_key)

    def _save_pipeline_cache(
        self,
        sql: str,
        *,
        dialect: str,
        use_llm_verify: bool,
        use_llm_enhance: bool,
        payload: dict[str, Any],
        quality_score: float,
        replace_if_better: bool,
    ) -> dict[str, Any]:
        cache_key = self._pipeline_cache_key(
            sql,
            dialect=dialect,
            use_llm_verify=use_llm_verify,
            use_llm_enhance=use_llm_enhance,
        )
        if cache_key is None:
            return {"updated": False}
        if replace_if_better:
            return self.llm_extractor.cache.set_if_better(
                cache_key,
                payload,
                quality_score=quality_score,
                entry_type="pipeline",
            )
        self.llm_extractor.cache.set(
            cache_key,
            payload,
            quality_score=quality_score,
            entry_type="pipeline",
        )
        return {"updated": True, "quality_score": quality_score, "previous_quality_score": None}

    @staticmethod
    def _collect_alias_columns_from_sql(sql_text: str) -> list[str]:
        pairs = re.findall(r'([A-Za-z_][\w\$]*)\.(?:"([^"]+)"|([A-Za-z_][\w\$]*))', sql_text or "")
        return [f"{alias}.{quoted or plain}" for alias, quoted, plain in pairs]

    @staticmethod
    def _cte_alias_map(extracted: dict[str, Any], simplified: dict[str, Any]) -> dict[str, str]:
        """Map table aliases used in the main query to the CTE names they refer to."""
        cte_names = {
            str(cte.get("alias", "")).strip().lower(): str(cte.get("alias", "")).strip()
            for cte in extracted.get("ctes", [])
            if cte.get("alias")
        }
        if not cte_names or not simplified.get("parser_used"):
            return {}

        candidates = list(simplified.get("from", []) or [])
        for join in simplified.get("joins", []) or []:
            candidates.append({"table": join.get("right_table"), "alias": join.get("alias")})

        alias_map: dict[str, str] = {}
        for item in candidates:
            table = str(item.get("table") or "").strip().strip('"').lower()
            alias = str(item.get("alias") or "").strip()
            if alias and table in cte_names and alias.lower() != table:
                alias_map[alias] = cte_names[table]
        return alias_map

    def _build_subgraphs(
        self,
        simplified: dict[str, Any],
        graph: nx.MultiDiGraph,
    ) -> list[dict[str, Any]]:
        """Build subgraph payloads for CTEs, JOIN blocks, and UNION branches."""
        subgraphs: list[dict[str, Any]] = []
        blocks = simplified.get("subgraph_blocks", [])

        for block in blocks:
            block_type = block.get("type")
            node_candidates = set()

            if block_type == "cte":
                prefix = f"{block.get('name')}."
                node_candidates = {node for node in graph.nodes if str(node).startswith(prefix)}
            elif block_type == "subjoin":
                for ref in block.get("join_columns", []):
                    alias = ref.get("table_alias")
                    column = ref.get("column")
                    if alias and column:
                        node_candidates.add(f"{alias}.{column}")
            elif block_type == "union_block":
                for alias in block.get("select_aliases", []):
                    node_candidates.add(f"output.{alias}")
                for node_id in self._collect_alias_columns_from_sql(block.get("sql", "")):
                    node_candidates.add(node_id)

            existing_nodes = {node for node in node_candidates if node in graph.nodes}
            if existing_nodes:
                subgraph_obj = graph.subgraph(existing_nodes).copy()
                try:
                    subgraph_json = json_graph.node_link_data(subgraph_obj, edges="links")
                except TypeError:
                    subgraph_json = json_graph.node_link_data(subgraph_obj)
            else:
                subgraph_json = {"nodes": [], "links": []}

            subgraphs.append(
                {
                    "id": block.get("id"),
                    "type": block_type,
                    "name": block.get("name"),
                    "sql": block.get("sql", ""),
                    "graph": subgraph_json,
                }
            )

        return subgraphs

    @staticmethod
    def _build_metadata(sql: str) -> dict[str, Any]:
        """Attach spec section 5 metadata to graph payloads."""
        return {
            "source_sql_hash": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "spec_version": "2.1",
            "implementation_profile": "column_level_v2",
            "limitations": [
                "udf_inputs_only",
                "unnest_best_effort",
                "structs_best_effort",
                "multi_statement_sql",
            ],
        }

    @staticmethod
    def _step_status(step: str, status: str, **details: Any) -> dict[str, Any]:
        return {"step": step, "status": status, **details}

    @staticmethod
    def _emit_step(
        pipeline_steps: dict[str, dict[str, Any]],
        step_name: str,
        status: str,
        step_callback: Callable[[str, dict[str, Any], dict[str, dict[str, Any]]], None] | None = None,
        **details: Any,
    ) -> None:
        pipeline_steps[step_name] = SQL2GraphPipeline._step_status(step_name, status, **details)
        if step_callback is not None:
            step_callback(step_name, pipeline_steps[step_name], dict(pipeline_steps))

    @staticmethod
    def diff_extraction(
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Summarize structural differences between two extraction payloads."""
        before = before or {}
        after = after or {}
        changes: list[dict[str, Any]] = []

        before_cols = {
            str(col.get("alias", "")).strip(): col
            for col in before.get("output_columns", [])
            if col.get("alias")
        }
        after_cols = {
            str(col.get("alias", "")).strip(): col
            for col in after.get("output_columns", [])
            if col.get("alias")
        }

        for alias in sorted(set(before_cols) | set(after_cols)):
            left = before_cols.get(alias)
            right = after_cols.get(alias)
            if left is None:
                changes.append({"area": "output_column", "alias": alias, "change": "added"})
                continue
            if right is None:
                changes.append({"area": "output_column", "alias": alias, "change": "removed"})
                continue
            for field in ("expression", "dependencies", "derivation_kind", "literal_values"):
                if left.get(field) != right.get(field):
                    changes.append(
                        {
                            "area": "output_column",
                            "alias": alias,
                            "field": field,
                            "before": left.get(field),
                            "after": right.get(field),
                        }
                    )

        for area in ("filters", "joins", "ctes"):
            before_count = len(before.get(area) or [])
            after_count = len(after.get(area) or [])
            if before_count != after_count:
                changes.append(
                    {
                        "area": area,
                        "change": "count_changed",
                        "before": before_count,
                        "after": after_count,
                    }
                )

        return {
            "change_count": len(changes),
            "changes": changes,
        }

    def _chunk_parser(self):
        if self.chunk_parser is not None:
            return self.chunk_parser
        from Classes.sql_chunk_classes import SQLLogicalChunkParser, SQLLogicalChunkPreParser

        return SQLLogicalChunkParser(
            pre_parser=SQLLogicalChunkPreParser(parser=self.parser),
        )

    def _run_plpgsql(
        self,
        sql: str,
        *,
        dialect: str | None = None,
        step_callback: Callable[[str, dict[str, Any], dict[str, dict[str, Any]]], None] | None = None,
    ) -> dict[str, Any]:
        """Route a ``CREATE FUNCTION ... LANGUAGE plpgsql`` body to the PL/pgSQL extractor.

        The response mirrors :meth:`run` so existing consumers (Web UI, CLI)
        keep working, with the procedural specifics carried in ``unresolved``,
        ``statements`` and ``temp_tables``.
        """
        from Classes.plpgsql_lineage import PlpgsqlLineageExtractor

        effective_dialect = dialect or getattr(self.parser, "dialect", "postgres")
        pipeline_steps: dict[str, dict[str, Any]] = {}

        self._emit_step(pipeline_steps, "chunking", "running", step_callback)
        extractor = PlpgsqlLineageExtractor(
            schema_registry=getattr(self.parser, "schema_registry", None),
            dialect=effective_dialect,
            parser=self.parser,
        )
        result = extractor.extract(sql)

        if result.get("error"):
            self._emit_step(pipeline_steps, "chunking", "failed", step_callback, error=result["error"])
            return {
                "error": result["error"],
                "pipeline_steps": pipeline_steps,
                "pipeline_stage": "plpgsql",
            }

        statements = result.get("statements") or []
        unresolved = result.get("unresolved") or []
        self._emit_step(
            pipeline_steps,
            "chunking",
            "completed",
            step_callback,
            chunk_count=len(statements),
            statement_type="plpgsql",
            target_table=result.get("function"),
        )
        self._emit_step(
            pipeline_steps,
            "parsing",
            "completed",
            step_callback,
            output_column_count=sum(1 for stmt in statements if stmt.get("resolved")),
            unresolved_count=len(unresolved),
            target_table=result.get("function"),
        )
        for skipped in ("verifying", "enhancing"):
            self._emit_step(
                pipeline_steps,
                skipped,
                "skipped",
                step_callback,
                message="Not applicable to PL/pgSQL extraction.",
            )

        graph_payload = result["graph"]
        self._emit_step(
            pipeline_steps,
            "combining",
            "completed",
            step_callback,
            node_count=len(graph_payload.get("nodes") or []),
            edge_count=len(graph_payload.get("links") or []),
            is_dag=result["metadata"].get("is_dag"),
        )

        return {
            "graph": graph_payload,
            "metadata": result["metadata"],
            "warnings": result.get("warnings") or [],
            "extraction": {"ctes": [], "output_columns": [], "filters": [], "joins": [], "group_by_columns": []},
            "deterministic_extraction": {},
            "pipeline_stage": "plpgsql",
            "pipeline_steps": pipeline_steps,
            "verification_diff": None,
            "enhancement_diff": None,
            "chunks": {"chunks": [], "links": []},
            "simplified_query": {
                "parser_used": True,
                "statement_type": "plpgsql",
                "target_table": result.get("function"),
            },
            "subgraphs": {},
            "cache": {"read_enabled": False, "hit": False, "updated": False},
            # PL/pgSQL specifics
            "function": result.get("function"),
            "functions": result.get("functions") or [],
            "statements": statements,
            "unresolved": unresolved,
            "temp_tables": result.get("temp_tables") or [],
            "variables": result.get("variables") or [],
            "table_lineage_statements": result.get("table_lineage") or [],
        }

    def run(
        self,
        sql: str,
        schema: dict[str, Any] | None = None,
        dialect: str | None = None,
        include_visualization: bool = False,
        use_llm_verify: bool = True,
        use_llm_enhance: bool = True,
        use_cache: bool = True,
        replace_cache_if_better: bool = True,
        golden_f1: float | None = None,
        parse_plpgsql: bool = False,
        step_callback: Callable[[str, dict[str, Any], dict[str, dict[str, Any]]], None] | None = None,
    ) -> dict[str, Any]:
        if parse_plpgsql:
            from Classes.plpgsql_lineage import contains_plpgsql_function

            if contains_plpgsql_function(sql):
                return self._run_plpgsql(sql, dialect=dialect, step_callback=step_callback)

        pipeline_steps: dict[str, dict[str, Any]] = {}
        warnings: list[str] = []
        pipeline_stage = "deterministic"
        verification_diff: dict[str, Any] | None = None
        enhancement_diff: dict[str, Any] | None = None
        cache_info: dict[str, Any] = {
            "read_enabled": use_cache,
            "write_replace_if_better": replace_cache_if_better,
            "hit": False,
            "updated": False,
        }
        effective_dialect = dialect or getattr(self.parser, "dialect", "postgres")

        # Step 1: chunking
        self._emit_step(pipeline_steps, "chunking", "running", step_callback)
        try:
            chunk_result = self._chunk_parser().preparse(sql, dialect=dialect)
            self._emit_step(
                pipeline_steps,
                "chunking",
                "completed",
                step_callback,
                chunk_count=len(chunk_result.get("chunks") or []),
                link_count=len(chunk_result.get("links") or []),
                statement_type=chunk_result.get("statement_type"),
                target_table=chunk_result.get("target_table"),
            )
        except Exception as exc:
            self._emit_step(pipeline_steps, "chunking", "failed", step_callback, error=str(exc))
            chunk_result = {"chunks": [], "links": []}
            warnings.append(f"Chunking failed: {exc}")

        # Step 2: parsing (sqlglot deterministic extraction)
        self._emit_step(pipeline_steps, "parsing", "running", step_callback)
        simplified = self.parser.simplify(sql, dialect=dialect)
        parse_fallback = False
        deterministic: dict[str, Any] = {}
        extracted: dict[str, Any]

        if not simplified.get("parser_used"):
            if use_llm_verify and self.llm_extractor is not None:
                llm_payload = self.llm_extractor.extract(
                    sql=sql,
                    schema=schema,
                    simplified_query=simplified,
                    use_cache=use_cache,
                )
                if "error" in llm_payload:
                    self._emit_step(
                        pipeline_steps,
                        "parsing",
                        "failed",
                        step_callback,
                        error=llm_payload.get("error"),
                    )
                    return {
                        "error": llm_payload.get("error", "LLM parse fallback failed"),
                        "details": llm_payload.get("details"),
                        "pipeline_steps": pipeline_steps,
                        "chunks": chunk_result,
                        "simplified_query": simplified,
                    }
                extracted = llm_payload
                parse_fallback = True
                pipeline_stage = "llm_parse_fallback"
                simplified = {
                    "parser_used": False,
                    "parse_fallback": True,
                    "raw_sql": sql,
                }
                self._emit_step(
                    pipeline_steps,
                    "parsing",
                    "fallback",
                    step_callback,
                    message="sqlglot parse failed; used LLM cold-start extraction.",
                )
            else:
                self._emit_step(
                    pipeline_steps,
                    "parsing",
                    "failed",
                    step_callback,
                    error=simplified.get("parse_error") or "sqlglot could not parse the SQL.",
                )
                return {
                    "error": simplified.get("parse_error") or "sqlglot could not parse the SQL.",
                    "pipeline_steps": pipeline_steps,
                    "chunks": chunk_result,
                    "simplified_query": simplified,
                }
        else:
            deterministic = self.parser.build_deterministic_extraction(simplified, dialect=dialect)
            extracted = deterministic
            self._emit_step(
                pipeline_steps,
                "parsing",
                "completed",
                step_callback,
                output_column_count=len(deterministic.get("output_columns") or []),
                cte_count=len(deterministic.get("ctes") or []),
                target_table=simplified.get("target_table"),
                operator_count=len(simplified.get("operators") or []),
            )

        verify_failed = parse_fallback
        loaded_from_cache = False

        if (
            use_cache
            and not parse_fallback
            and self.llm_extractor is not None
            and (use_llm_verify or use_llm_enhance)
        ):
            cached_entry = self._load_pipeline_cache(
                sql,
                dialect=effective_dialect,
                use_llm_verify=use_llm_verify,
                use_llm_enhance=use_llm_enhance,
            )
            if cached_entry and isinstance(cached_entry.get("payload"), dict):
                cached_payload = cached_entry["payload"]
                extracted = cached_payload.get("extraction") or deterministic
                pipeline_stage = cached_payload.get("pipeline_stage") or pipeline_stage
                verification_diff = cached_payload.get("verification_diff")
                enhancement_diff = cached_payload.get("enhancement_diff")
                loaded_from_cache = True
                cache_info.update(
                    {
                        "hit": True,
                        "quality_score": cached_entry.get("quality_score"),
                        "created_at": cached_entry.get("created_at"),
                    }
                )
                self._emit_step(
                    pipeline_steps,
                    "verifying",
                    "skipped",
                    step_callback,
                    message="Loaded from LLM cache.",
                )
                self._emit_step(
                    pipeline_steps,
                    "enhancing",
                    "skipped",
                    step_callback,
                    message="Loaded from LLM cache.",
                )

        # Step 3: verifying (optional LLM)
        if not loaded_from_cache and use_llm_verify and self.llm_extractor is not None and not parse_fallback:
            self._emit_step(pipeline_steps, "verifying", "running", step_callback)
            verified_payload = self.llm_extractor.verify(
                sql=sql,
                deterministic_draft=deterministic,
                schema=schema,
                simplified_query=simplified,
            )
            if "error" in verified_payload:
                verify_failed = True
                warnings.append(str(verified_payload.get("error")))
                if verified_payload.get("details"):
                    warnings.append(str(verified_payload["details"]))
                extracted = deterministic
                pipeline_stage = "deterministic_fallback"
                self._emit_step(
                    pipeline_steps,
                    "verifying",
                    "fallback",
                    step_callback,
                    message="LLM verification failed; using sqlglot draft.",
                )
            else:
                verification_diff = self.diff_extraction(deterministic, verified_payload)
                extracted = verified_payload
                pipeline_stage = "llm_verified"
                self._emit_step(
                    pipeline_steps,
                    "verifying",
                    "completed",
                    step_callback,
                    change_count=verification_diff["change_count"],
                    diff=verification_diff,
                )
        else:
            if not loaded_from_cache:
                if parse_fallback:
                    reason = "Skipped; LLM already extracted during parse fallback."
                elif not use_llm_verify:
                    reason = "LLM verification disabled."
                elif self.llm_extractor is None:
                    reason = "No LLM extractor configured."
                else:
                    reason = "LLM verification disabled."
                self._emit_step(pipeline_steps, "verifying", "skipped", step_callback, message=reason)

        # Step 4: enhancing (optional LLM)
        if (
            not loaded_from_cache
            and use_llm_enhance
            and self.llm_extractor is not None
            and (not verify_failed or parse_fallback)
        ):
            self._emit_step(pipeline_steps, "enhancing", "running", step_callback)
            before_enhance = copy.deepcopy(extracted)
            if not self.llm_extractor.enable_refinement:
                self._emit_step(
                    pipeline_steps,
                    "enhancing",
                    "skipped",
                    step_callback,
                    message="LLM refinement disabled on extractor.",
                )
            else:
                enhanced = self.llm_extractor.enhance(
                    sql=sql,
                    verified_payload=extracted,
                    schema=schema,
                    simplified_query=simplified,
                )
                if "error" in enhanced:
                    error_text = str(enhanced.get("error") or "LLM enhancement failed")
                    details = str(enhanced.get("details") or "")
                    if enhanced.get("transient"):
                        warnings.append(
                            "LLM enhancement skipped: inference provider was busy after retries. "
                            "Using verified/sqlglot draft."
                        )
                        if details:
                            warnings.append(details)
                    else:
                        warnings.append(error_text)
                        if details:
                            warnings.append(details)
                    self._emit_step(
                        pipeline_steps,
                        "enhancing",
                        "fallback",
                        step_callback,
                        message="LLM enhancement failed; using previous draft.",
                        error=details or error_text,
                        transient=bool(enhanced.get("transient")),
                    )
                else:
                    enhancement_diff = self.diff_extraction(before_enhance, enhanced)
                    extracted = enhanced
                    pipeline_stage = "llm_enhanced"
                    self._emit_step(
                        pipeline_steps,
                        "enhancing",
                        "completed",
                        step_callback,
                        change_count=enhancement_diff["change_count"],
                        diff=enhancement_diff,
                    )
        else:
            if not loaded_from_cache:
                if verify_failed:
                    message = "Skipped because verification failed."
                elif not use_llm_enhance:
                    message = "LLM enhancement disabled."
                elif self.llm_extractor is None:
                    message = "No LLM extractor configured."
                else:
                    message = "LLM enhancement disabled."
                self._emit_step(pipeline_steps, "enhancing", "skipped", step_callback, message=message)

        if (use_llm_verify or use_llm_enhance) and self.llm_extractor is not None and pipeline_stage not in {
            "deterministic_fallback",
            "deterministic",
        }:
            extracted = copy.deepcopy(extracted)

        if deterministic:
            extracted = self.parser.overlay_deterministic_column_lineage(extracted, deterministic)
            extracted = self.parser._materialize_output_dependencies(extracted, simplified)

        try:
            SQL2GraphExtraction.model_validate(extracted)
        except ValidationError as exc:
            self._emit_step(pipeline_steps, "combining", "failed", step_callback, error=str(exc))
            return {
                "error": "Lineage extraction validation failed",
                "details": str(exc),
                "deterministic_extraction": deterministic,
                "simplified_query": simplified,
                "pipeline_steps": pipeline_steps,
                "chunks": chunk_result,
            }

        # Step 5: combining (graph build + validation)
        self._emit_step(pipeline_steps, "combining", "running", step_callback)
        graph = self.builder.build(extracted)
        if pipeline_stage == "llm_parse_fallback":
            self.builder.apply_edge_provenance("llm", 0.8)
        elif pipeline_stage == "llm_verified":
            self.builder.apply_edge_provenance("llm_verified", 0.9)
        elif pipeline_stage == "llm_enhanced":
            self.builder.apply_edge_provenance("llm_verified", 0.95)
        self.builder.link_cte_aliases(self._cte_alias_map(extracted, simplified))
        self.builder.materialize_transitive_derived_from()
        dag_warnings = self.builder.ensure_acyclic()
        validation_warnings = self.validator.validate_graph(graph, schema=schema)
        warnings.extend(validation_warnings)
        warnings.extend(dag_warnings)
        graph_payload = self.builder.to_node_link()
        graph_payload["metadata"] = self._build_metadata(sql)
        graph_payload["metadata"]["is_dag"] = nx.is_directed_acyclic_graph(self.builder.graph)
        self._emit_step(
            pipeline_steps,
            "combining",
            "completed",
            step_callback,
            node_count=len(graph_payload.get("nodes") or []),
            edge_count=len(graph_payload.get("links") or []),
            is_dag=graph_payload["metadata"]["is_dag"],
        )

        response = {
            "graph": graph_payload,
            "metadata": graph_payload["metadata"],
            "warnings": warnings,
            "extraction": extracted,
            "deterministic_extraction": deterministic,
            "pipeline_stage": pipeline_stage,
            "pipeline_steps": pipeline_steps,
            "verification_diff": verification_diff,
            "enhancement_diff": enhancement_diff,
            "chunks": chunk_result,
            "chunk_graph": self._chunk_parser().to_node_link(chunk_result),
            "simplified_query": simplified,
            "subgraphs": self._build_subgraphs(simplified, graph),
            "cache": cache_info,
        }

        if self.llm_extractor is not None and getattr(self.llm_extractor, "cache", None) is not None and not loaded_from_cache:
            quality_score = pipeline_result_quality(
                pipeline_stage=pipeline_stage,
                extraction=extracted,
                graph=graph_payload,
                golden_f1=golden_f1,
            )
            cache_info["quality_score"] = quality_score
            if use_cache:
                write_result = self._save_pipeline_cache(
                    sql,
                    dialect=effective_dialect,
                    use_llm_verify=use_llm_verify,
                    use_llm_enhance=use_llm_enhance,
                    payload={
                        "extraction": extracted,
                        "pipeline_stage": pipeline_stage,
                        "verification_diff": verification_diff,
                        "enhancement_diff": enhancement_diff,
                    },
                    quality_score=quality_score,
                    replace_if_better=False,
                )
            elif replace_cache_if_better:
                write_result = self._save_pipeline_cache(
                    sql,
                    dialect=effective_dialect,
                    use_llm_verify=use_llm_verify,
                    use_llm_enhance=use_llm_enhance,
                    payload={
                        "extraction": extracted,
                        "pipeline_stage": pipeline_stage,
                        "verification_diff": verification_diff,
                        "enhancement_diff": enhancement_diff,
                    },
                    quality_score=quality_score,
                    replace_if_better=True,
                )
            else:
                write_result = {"updated": False, "skipped": True}
            cache_info.update(write_result)

        if include_visualization:
            response["visualization"] = {
                "mermaid": self.builder.to_mermaid(),
                "dot": self.builder.to_dot(),
            }
        return response
