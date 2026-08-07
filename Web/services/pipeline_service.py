"""SQL lineage pipeline helpers used by the Streamlit UI."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import networkx as nx

from Classes.schema_registry import SchemaRegistry
from Classes.sql2graph_classes import (
    SQL2GraphLLMExtractor,
    SQL2GraphParser,
    SQL2GraphPipeline,
)
from Classes.table_lineage import extract_create_ddl, extract_table_lineage
from Classes.validation_classes import SQLLineageValidator
from Web.services.cache_service import make_llm_cache

ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_DIR = ROOT / "tests" / "golden"
# (sql_file, statement_index, golden_json) — extend when adding new golden fixtures
GOLDEN_CASES: list[tuple[Path, int, Path]] = [
    (ROOT / "data" / "DDLs_10.txt", 0, GOLDEN_DIR / "ddls10_first_graph.json"),
]

PIPELINE_STAGE_LABELS = {
    "deterministic": "sqlglot",
    "llm_verified": "verified",
    "llm_enhanced": "enhanced",
    "llm_parse_fallback": "llm parse",
    "deterministic_fallback": "fallback",
}

UNRESOLVED_LABELS = {
    "dynamic_execute": "Dynamic SQL",
    "parse_failed": "Parse failed",
    "recursive_call": "Recursive call",
    "unsupported_statement": "Unsupported statement",
    "build_failed": "Graph build failed",
    "max_depth_exceeded": "Call depth limit",
}


def shorten_text(text: Any, max_len: int = 28) -> str:
    value = str(text or "—")
    return value if len(value) <= max_len else value[: max_len - 1] + "…"


def split_sql_statements(content: str) -> list[str]:
    try:
        import sqlparse

        return [str(stmt).strip() for stmt in sqlparse.parse(content) if str(stmt).strip()]
    except ImportError:
        return [s.strip() for s in content.split(";") if s.strip()]


def statement_target_table(sql: str, index: int, dialect: str) -> str:
    """Resolve write target (INSERT/MERGE/UPDATE/CREATE) for the statement picker."""
    info = extract_table_lineage(sql, dialect=dialect)
    target = (info.get("target") or "").strip()
    statement_type = (info.get("statement_type") or "").strip()
    if target:
        type_prefix = {
            "create_table_as": "CTAS",
            "create_view": "VIEW",
            "create_materialized_view": "MATVIEW",
            "create_table": "TABLE",
        }.get(statement_type)
        return f"{type_prefix} {target}" if type_prefix else target
    preview = shorten_text(re.sub(r"\s+", " ", sql), 48)
    return preview or f"Statement {index + 1}"


def build_target_table_labels(statements: list[str], dialect: str) -> list[str]:
    """Target table names; disambiguate duplicates with statement index."""
    base_labels = [statement_target_table(stmt, idx, dialect) for idx, stmt in enumerate(statements)]
    counts: dict[str, int] = {}
    for label in base_labels:
        counts[label] = counts.get(label, 0) + 1
    labels: list[str] = []
    for idx, label in enumerate(base_labels):
        if counts[label] > 1:
            labels.append(f"{label} (#{idx + 1})")
        else:
            labels.append(label)
    return labels


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip()).lower()


def golden_fixture_for_sql(sql: str) -> tuple[str, Path] | None:
    """Return (fixture_id, path) when analyzed SQL matches a known golden case."""
    norm = normalize_sql(sql)
    for sql_path, statement_index, golden_path in GOLDEN_CASES:
        if not sql_path.exists() or not golden_path.exists():
            continue
        statements = split_sql_statements(sql_path.read_text(encoding="utf-8"))
        if statement_index < len(statements) and normalize_sql(statements[statement_index]) == norm:
            return golden_path.stem.replace("_graph", ""), golden_path
    return None


def edge_f1_vs_golden(graph: dict[str, Any], golden_path: Path) -> dict[str, Any]:
    expected = json.loads(golden_path.read_text(encoding="utf-8"))
    metrics = SQLLineageValidator.calculate_edge_f1(expected, graph)
    return {
        "fixture": golden_path.stem.replace("_graph", ""),
        "golden_path": str(golden_path.relative_to(ROOT)),
        **metrics,
    }


def compact_pipeline_summary(stage: str, target_table: Any, output_columns: int) -> str:
    stage_label = PIPELINE_STAGE_LABELS.get(stage, shorten_text(stage, 16))
    target_short = shorten_text(target_table, 32)
    return f"Stage **{stage_label}** · Target `{target_short}` · **{output_columns}** cols"


def friendly_llm_warning(text: str) -> tuple[str, str]:
    """Return (severity, message) for pipeline warnings."""
    lowered = text.lower()
    if any(
        phrase in lowered
        for phrase in (
            "provider was busy",
            "model is busy",
            "try again later",
            "completion_error",
            "server_error",
        )
    ):
        return (
            "info",
            "Inference provider was busy — enhancement was skipped; lineage uses the verified/sqlglot draft.",
        )
    return ("warning", shorten_text(text, 240))


def plpgsql_table_lineage(result: dict[str, Any]) -> dict[str, Any]:
    """Roll the per-statement table lineage of a routine into one target/sources view."""
    entries = result.get("table_lineage_statements") or []
    temp_tables = set(result.get("temp_tables") or [])
    written = [entry.get("target") for entry in entries if entry.get("target")]
    sources = {
        source
        for entry in entries
        for source in entry.get("sources") or []
        if source not in temp_tables
    }
    final_targets = [table for table in written if table not in temp_tables]
    return {
        "target": final_targets[-1] if final_targets else (result.get("function") or ""),
        "sources": sorted(sources - set(final_targets)),
        "statement_type": "plpgsql",
        "parser_used": True,
        "all_targets": sorted(set(final_targets)),
    }


def build_schema_registry(
    dialect: str,
    *,
    schema_ddl: str = "",
    sql_script: str = "",
) -> SchemaRegistry | None:
    """Merge sidebar DDL with CREATE TABLE/VIEW statements found in the SQL script."""
    registry = SchemaRegistry(dialect=dialect)
    loaded = False
    if schema_ddl.strip():
        registry.load_ddl(schema_ddl)
        loaded = True
    create_ddl = extract_create_ddl(sql_script, dialect=dialect) if sql_script.strip() else ""
    if create_ddl.strip():
        registry.load_ddl(create_ddl)
        loaded = True
    if not loaded:
        return None
    if not registry.has_tables() and not registry.views:
        return None
    return registry


def run_column_pipeline(
    sql: str,
    *,
    dialect: str,
    use_llm_verify: bool,
    use_llm_enhance: bool,
    hf_token: str | None,
    hf_model: str | None = None,
    hf_provider: str | None = None,
    use_llm_cache: bool = True,
    replace_cache_if_better: bool = True,
    golden_f1: float | None = None,
    schema_registry: SchemaRegistry | None = None,
    parse_plpgsql: bool = False,
    step_callback=None,
) -> dict[str, Any]:
    """Run chunking → parsing → verifying → enhancing → combining."""
    if schema_registry is None:
        schema_registry = build_schema_registry(dialect, sql_script=sql)
    else:
        # Always fold CREATE … from the analyzed statement into the registry.
        create_ddl = extract_create_ddl(sql, dialect=dialect)
        if create_ddl.strip():
            schema_registry.load_ddl(create_ddl)
    parser = SQL2GraphParser(dialect=dialect, schema_registry=schema_registry)
    llm_extractor = None
    if (use_llm_verify or use_llm_enhance) and hf_token:
        llm_extractor = SQL2GraphLLMExtractor(
            hf_token=hf_token,
            model=hf_model,
            provider=hf_provider,
            cache=make_llm_cache(use_llm_cache),
        )
        llm_extractor.use_llm_cache = use_llm_cache

    pipeline = SQL2GraphPipeline(llm_extractor=llm_extractor, parser=parser)
    result = pipeline.run(
        sql,
        dialect=dialect,
        use_llm_verify=bool(llm_extractor and use_llm_verify),
        use_llm_enhance=bool(llm_extractor and use_llm_enhance),
        use_cache=use_llm_cache,
        replace_cache_if_better=replace_cache_if_better,
        golden_f1=golden_f1,
        parse_plpgsql=parse_plpgsql,
        step_callback=step_callback,
    )
    if "error" in result:
        raise RuntimeError(result.get("error", "Pipeline failed"))
    if llm_extractor:
        result["llm_config"] = {
            "model": llm_extractor.model,
            "provider": llm_extractor.provider,
        }
    if result.get("pipeline_stage") == "plpgsql":
        result["table_lineage"] = plpgsql_table_lineage(result)
    else:
        result["table_lineage"] = extract_table_lineage(sql, dialect=dialect)
    return result


def target_columns_from_result(result: dict[str, Any]) -> list[str]:
    extraction = result.get("extraction") or {}
    return [col.get("alias", "") for col in extraction.get("output_columns", []) if col.get("alias")]


def column_record(extraction: dict[str, Any], alias: str) -> dict[str, Any] | None:
    for col in extraction.get("output_columns", []):
        if col.get("alias") == alias:
            return col
    return None


def cte_derivation_context(
    extraction: dict[str, Any],
    record: dict[str, Any],
    simplified: dict[str, Any],
) -> dict[str, Any] | None:
    """When an output column passes through a CTE, return that CTE column definition."""
    alias_to_cte = SQL2GraphParser._cte_alias_lookup(simplified)
    if not alias_to_cte:
        return None

    expression = record.get("expression") or ""
    for dep in record.get("dependencies") or []:
        table_alias = dep.get("table_alias")
        if table_alias and table_alias.lower() in alias_to_cte:
            cte_name = alias_to_cte[table_alias.lower()]
            cte_col = next(
                (
                    col
                    for cte in extraction.get("ctes", [])
                    if cte.get("alias") == cte_name
                    for col in cte.get("output_columns", [])
                    if col.get("alias") == dep.get("column")
                ),
                None,
            )
            if cte_col:
                return {"cte_name": cte_name, "column": cte_col}

    match = re.match(r"^(\w+)\.(\w+)", expression.split()[0] if expression else "")
    if not match:
        return None
    table_alias, column = match.group(1), match.group(2)
    cte_name = alias_to_cte.get(table_alias.lower())
    if not cte_name:
        return None
    cte_col = next(
        (
            col
            for cte in extraction.get("ctes", [])
            if cte.get("alias") == cte_name
            for col in cte.get("output_columns", [])
            if col.get("alias") == column
        ),
        None,
    )
    if not cte_col:
        return None
    return {"cte_name": cte_name, "column": cte_col}


def resolve_output_node(graph: nx.MultiDiGraph, alias: str) -> str | None:
    candidate = f"output.{alias}"
    if candidate in graph:
        return candidate
    for node, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "output_column" and attrs.get("alias") == alias:
            return node
    return None


def upstream_lineage_nodes(graph: nx.MultiDiGraph, output_node: str) -> set[str]:
    """Collect nodes upstream of an output column via DERIVED_FROM / GROUPED_BY edges."""
    visited: set[str] = set()
    stack = [output_node]
    while stack:
        node = stack.pop()
        for pred in graph.predecessors(node):
            edge_data = graph.get_edge_data(pred, node) or {}
            for data in edge_data.values():
                if data.get("edge_type") in {"DERIVED_FROM", "GROUPED_BY"}:
                    if pred not in visited:
                        visited.add(pred)
                        stack.append(pred)
                    break
    return visited


def columns_linked_to_source_table(result: dict[str, Any], source_table: str) -> list[str]:
    """Output columns that depend on a physical source table (column-level detail)."""
    source_norm = source_table.strip().lower()
    linked: list[str] = []
    extraction = result.get("extraction") or {}
    for col in extraction.get("output_columns", []):
        alias = col.get("alias")
        if not alias:
            continue
        for dep in col.get("dependencies") or []:
            physical = str(dep.get("physical_table") or "").strip().lower()
            if physical and physical == source_norm:
                linked.append(alias)
                break
    return linked
