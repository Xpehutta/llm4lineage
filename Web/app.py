"""
Column-level SQL lineage — Streamlit web interface.

Workflow:
  1. Upload SQL (file drop zone or paste)
  2. Run sqlglot-first pipeline (optional LLM verify/enhance)
  3. Click a target column to inspect its source-to-target lineage
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import graphviz
import networkx as nx
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from Classes.llm_cache import LLMCache  # noqa: E402
from Classes.pipeline.llm_helpers import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_PROVIDER,
    create_chat_model,
    resolve_model_name,
    resolve_provider,
)
from Classes.schema_registry import SchemaRegistry  # noqa: E402
from Classes.sql2graph_classes import (  # noqa: E402
    SQL2GraphLLMExtractor,
    SQL2GraphParser,
    SQL2GraphPipeline,
    SQL2GraphVisualizer,
)
from Classes.table_lineage import extract_table_lineage  # noqa: E402
from Classes.validation_classes import SQLLineageValidator  # noqa: E402

GOLDEN_DIR = ROOT / "tests" / "golden"
# (sql_file, statement_index, golden_json) — extend when adding new golden fixtures
GOLDEN_CASES: list[tuple[Path, int, Path]] = [
    (ROOT / "data" / "DDLs_10.txt", 0, GOLDEN_DIR / "ddls10_first_graph.json"),
]

HF_MODEL_PRESETS = [
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "Qwen/Qwen3-Coder-Next",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Custom…",
]
HF_PROVIDER_PRESETS = [
    "scaleway",
    "novita",
    "nebius",
    "fireworks-ai",
    "together",
    "hyperbolic",
    "groq",
    "Custom…",
]
CUSTOM_CHOICE = "Custom…"

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Column Lineage Explorer", page_icon="🔍", layout="wide")

st.title("Column Lineage Explorer")
st.caption("chunking → parsing → verifying → enhancing → combining → per-column lineage")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Configuration")

hf_token = st.sidebar.text_input(
    "Hugging Face Token",
    type="password",
    value=os.environ.get("HF_TOKEN") or os.environ.get("HF_API_TOKEN") or "",
)

default_model = resolve_model_name()
default_provider = resolve_provider()
model_options = list(HF_MODEL_PRESETS)
if default_model not in model_options:
    model_options = [default_model, *[m for m in model_options if m != CUSTOM_CHOICE], CUSTOM_CHOICE]
model_index = model_options.index(default_model) if default_model in model_options else 0
model_choice = st.sidebar.selectbox(
    "HuggingFace model",
    model_options,
    index=model_index,
    help="Repo ID for Hugging Face Inference Providers router.",
)
if model_choice == CUSTOM_CHOICE:
    hf_model = st.sidebar.text_input("Custom model ID", value=default_model)
else:
    hf_model = model_choice

provider_options = list(HF_PROVIDER_PRESETS)
if default_provider not in provider_options:
    provider_options = [default_provider, *[p for p in provider_options if p != CUSTOM_CHOICE], CUSTOM_CHOICE]
provider_index = provider_options.index(default_provider) if default_provider in provider_options else 0
provider_choice = st.sidebar.selectbox(
    "Inference provider",
    provider_options,
    index=provider_index,
    help="HF Inference Providers backend (see huggingface.co/docs/inference-providers).",
)
if provider_choice == CUSTOM_CHOICE:
    hf_provider = st.sidebar.text_input("Custom provider", value=default_provider)
else:
    hf_provider = provider_choice

preset_model = model_choice != CUSTOM_CHOICE
preset_provider = provider_choice != CUSTOM_CHOICE

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "uploaded_sql" not in st.session_state:
    st.session_state.uploaded_sql = ""
if "selected_column" not in st.session_state:
    st.session_state.selected_column = None
if "source_filename" not in st.session_state:
    st.session_state.source_filename = ""
if "selected_statement_index" not in st.session_state:
    st.session_state.selected_statement_index = 0
if "sql_content_hash" not in st.session_state:
    st.session_state.sql_content_hash = ""
if "analyzed_sql_hash" not in st.session_state:
    st.session_state.analyzed_sql_hash = ""
if "lineage_level" not in st.session_state:
    st.session_state.lineage_level = "Column"
if "selected_source_table" not in st.session_state:
    st.session_state.selected_source_table = None
if "llm_check" not in st.session_state:
    st.session_state.llm_check = None
if "llm_check_key" not in st.session_state:
    st.session_state.llm_check_key = None


def llm_config_key(model: str, provider: str, token: str) -> str:
    token_fp = hashlib.sha256(token.encode()).hexdigest()[:16] if token else ""
    return f"{model.strip()}|{provider.strip()}|{token_fp}"


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


def render_llm_health_check(
    model: str,
    provider: str,
    token: str,
    *,
    preset_model: bool,
    preset_provider: bool,
) -> None:
    if not token:
        st.sidebar.caption("Model check: add HF token to test connectivity.")
        return
    if not model.strip() or not provider.strip():
        st.sidebar.caption("Model check: enter model and provider.")
        return

    config_key = llm_config_key(model, provider, token)
    auto_check = preset_model and preset_provider
    manual_check = not auto_check

    if manual_check:
        if st.sidebar.button("Test model connection", width="stretch", key="llm_health_btn"):
            with st.sidebar.spinner("Testing model…"):
                st.session_state.llm_check = execute_llm_health_check(model, provider, token)
                st.session_state.llm_check_key = config_key
    elif st.session_state.get("llm_check_key") != config_key:
        with st.sidebar.spinner("Testing model…"):
            st.session_state.llm_check = execute_llm_health_check(model, provider, token)
            st.session_state.llm_check_key = config_key

    check = st.session_state.get("llm_check")
    if not check or st.session_state.get("llm_check_key") != config_key:
        if auto_check:
            st.sidebar.caption("Model check: running…")
        return

    if check.get("ok"):
        st.sidebar.success(
            f"Model OK — `{check.get('model')}` via `{check.get('provider')}`"
        )
        if check.get("preview"):
            st.sidebar.caption(f"Response: {check['preview']}")
    else:
        st.sidebar.error(f"Model check failed: {check.get('error', 'unknown error')}")


render_llm_health_check(
    hf_model,
    hf_provider,
    hf_token,
    preset_model=preset_model,
    preset_provider=preset_provider,
)

dialect = st.sidebar.selectbox("SQL dialect", ["postgres", "spark", "teradata", "hive"], index=0)
schema_ddl = st.sidebar.text_area(
    "Schema DDL (optional)",
    height=120,
    placeholder="CREATE TABLE schema.table (col1 int, col2 text);",
    help="Paste CREATE TABLE/VIEW DDL to resolve SELECT * and qualify columns.",
)
parse_plpgsql = st.sidebar.checkbox(
    "Parse PL/pgSQL function bodies",
    value=False,
    help=(
        "Split `CREATE FUNCTION ... LANGUAGE plpgsql` bodies into statements and "
        "build lineage across them. Dynamic SQL is reported as unresolved."
    ),
)
use_llm_verify = st.sidebar.checkbox(
    "LLM verify",
    value=bool(hf_token),
    help="When enabled, the LLM reviews the sqlglot draft for correctness.",
)
use_llm_enhance = st.sidebar.checkbox(
    "LLM enhance",
    value=bool(hf_token),
    help="When enabled, the LLM applies targeted fixes after verification (or on the sqlglot draft if verify is off).",
)
use_llm_cache = st.sidebar.checkbox(
    "Use LLM cache",
    value=True,
    help="Reuse cached verify/enhance results for identical SQL + model + dialect.",
)
replace_cache_if_better = st.sidebar.checkbox(
    "Replace cache if better",
    value=True,
    help="When cache is bypassed, store the fresh result only if it scores higher than the cached entry.",
)
if not use_llm_cache and not replace_cache_if_better:
    st.sidebar.caption("Fresh run — cache will not be read or updated.")
elif not use_llm_cache:
    st.sidebar.caption("Fresh LLM run; cache updates only when the new result is better.")
if (use_llm_verify or use_llm_enhance) and not hf_token:
    st.sidebar.warning("LLM enabled but no HF token — only sqlglot will run.")
elif (use_llm_verify or use_llm_enhance) and hf_token:
    check = st.session_state.get("llm_check")
    config_key = llm_config_key(hf_model, hf_provider, hf_token)
    if not check or st.session_state.get("llm_check_key") != config_key:
        st.sidebar.warning("Run model connection test before LLM steps.")
    elif not check.get("ok"):
        st.sidebar.warning("Last model check failed — LLM steps may error.")


def split_sql_statements(content: str) -> list[str]:
    try:
        import sqlparse

        return [str(stmt).strip() for stmt in sqlparse.parse(content) if str(stmt).strip()]
    except ImportError:
        return [s.strip() for s in content.split(";") if s.strip()]


def statement_target_table(sql: str, index: int, dialect: str) -> str:
    """Resolve INSERT/MERGE/UPDATE target table for statement picker."""
    info = extract_table_lineage(sql, dialect=dialect)
    target = (info.get("target") or "").strip()
    if target:
        return target
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


def resolve_active_sql(sql_text: str, dialect: str) -> tuple[list[str], str, int]:
    """Return (all statements, selected statement SQL, selected index)."""
    statements = split_sql_statements(sql_text) if sql_text.strip() else []
    if not statements:
        return [], "", 0

    content_hash = hashlib.sha256(sql_text.encode("utf-8")).hexdigest()[:16]
    if st.session_state.get("sql_content_hash") != content_hash:
        st.session_state.sql_content_hash = content_hash
        st.session_state.selected_statement_index = 0
        st.session_state.pop("statement_selector", None)

    selected_index = int(st.session_state.get("selected_statement_index", 0))
    selected_index = max(0, min(selected_index, len(statements) - 1))
    return statements, statements[selected_index], selected_index


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


def shorten_text(text: Any, max_len: int = 28) -> str:
    value = str(text or "—")
    return value if len(value) <= max_len else value[: max_len - 1] + "…"


PIPELINE_STAGE_LABELS = {
    "deterministic": "sqlglot",
    "llm_verified": "verified",
    "llm_enhanced": "enhanced",
    "llm_parse_fallback": "llm parse",
    "deterministic_fallback": "fallback",
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
    parser = SQL2GraphParser(dialect=dialect, schema_registry=schema_registry)
    llm_extractor = None
    if (use_llm_verify or use_llm_enhance) and hf_token:
        llm_extractor = SQL2GraphLLMExtractor(
            hf_token=hf_token,
            model=hf_model,
            provider=hf_provider,
            cache=LLMCache() if use_llm_cache else None,
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


def render_pipeline_steps(steps: dict[str, Any], running_step: str | None = None) -> None:
    labels = {
        "chunking": "1. Chunking",
        "parsing": "2. Parsing",
        "verifying": "3. Verifying",
        "enhancing": "4. Enhancing",
        "combining": "5. Combining",
    }
    cols = st.columns(5)
    for index, step_name in enumerate(SQL2GraphPipeline.PIPELINE_STEP_ORDER):
        step = steps.get(step_name) or {}
        status = step.get("status", "pending")
        if step_name == running_step and status == "running":
            status = "running"
        with cols[index]:
            if status == "completed":
                st.success(labels[step_name])
            elif status == "running":
                st.info(f"⏳ {labels[step_name]}")
            elif status == "skipped":
                st.info(labels[step_name])
            elif status == "fallback":
                st.warning(labels[step_name])
            elif status == "failed":
                st.error(labels[step_name])
            else:
                st.write(labels[step_name])


def render_extraction_diff(title: str, diff: dict[str, Any] | None) -> None:
    if not diff:
        return
    change_count = diff.get("change_count", 0)
    if change_count == 0:
        st.caption(f"{title}: no structural changes detected.")
        return
    st.markdown(f"**{title}** ({change_count} change{'s' if change_count != 1 else ''})")
    for change in diff.get("changes") or []:
        area = change.get("area", "unknown")
        if change.get("change") in {"added", "removed", "count_changed"}:
            st.write(f"- `{area}`: {change.get('change')}")
            if change.get("before") is not None or change.get("after") is not None:
                st.caption(f"  {change.get('before')} → {change.get('after')}")
        else:
            alias = change.get("alias", "—")
            field = change.get("field", "—")
            st.write(f"- `{alias}.{field}` changed")
            if field == "expression":
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("Before")
                    st.code(str(change.get("before") or ""), language="sql")
                with col2:
                    st.caption("After")
                    st.code(str(change.get("after") or ""), language="sql")
            else:
                st.caption(f"  {change.get('before')} → {change.get('after')}")


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


def build_column_lineage_dot(graph_json: dict[str, Any], column_alias: str) -> graphviz.Digraph | None:
    graph = SQL2GraphVisualizer.graph_from_node_link(graph_json)
    output_node = resolve_output_node(graph, column_alias)
    if not output_node:
        return None

    nodes_to_show = upstream_lineage_nodes(graph, output_node) | {output_node}
    dot = graphviz.Digraph(comment=f"Lineage for {column_alias}")
    dot.attr(rankdir="LR")

    out_attrs = graph.nodes[output_node]
    dot.node(
        output_node,
        out_attrs.get("alias") or column_alias,
        shape="box",
        style="filled",
        fillcolor="#ADD8E6",
    )

    for node in sorted(nodes_to_show - {output_node}):
        attrs = graph.nodes[node]
        label = attrs.get("column") or node
        if attrs.get("table_alias"):
            label = f"{attrs['table_alias']}.{label}"
        dot.node(node, label, shape="ellipse", style="filled", fillcolor="#90EE90")

    for source, target, data in graph.edges(data=True):
        if data.get("edge_type") != "DERIVED_FROM":
            continue
        if source in nodes_to_show and target in nodes_to_show:
            dot.edge(source, target)

    return dot


def build_table_lineage_dot(
    target: str,
    sources: list[str],
    *,
    highlight: str | None = None,
) -> graphviz.Digraph:
    """Simple source-table → target-table lineage graph."""
    dot = graphviz.Digraph(comment="Table lineage")
    dot.attr(rankdir="LR")

    if target:
        dot.node(
            target,
            shorten_text(target, 40),
            shape="box",
            style="filled",
            fillcolor="#ADD8E6",
        )

    for source in sources:
        is_highlight = highlight and source == highlight
        dot.node(
            source,
            shorten_text(source, 40),
            shape="ellipse",
            style="filled",
            fillcolor="#FFD700" if is_highlight else "#90EE90",
        )
        if target:
            dot.edge(source, target)

    return dot


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


UNRESOLVED_LABELS = {
    "dynamic_execute": "Dynamic SQL",
    "parse_failed": "Parse failed",
    "recursive_call": "Recursive call",
    "unsupported_statement": "Unsupported statement",
    "build_failed": "Graph build failed",
    "max_depth_exceeded": "Call depth limit",
}


def render_plpgsql_panel(result: dict[str, Any]) -> None:
    """Show routine-level detail: statements, temp tables and what stayed unresolved."""
    statements = result.get("statements") or []
    unresolved = result.get("unresolved") or []
    resolved_count = sum(1 for stmt in statements if stmt.get("resolved"))

    st.markdown(f"**PL/pgSQL routine** · `{result.get('function') or 'unknown'}`")
    summary = f"{resolved_count}/{len(statements)} statements resolved"
    if result.get("temp_tables"):
        summary += f" · temp: {', '.join(f'`{t}`' for t in result['temp_tables'])}"
    if result.get("variables"):
        summary += f" · vars: {len(result['variables'])}"
    st.caption(summary)

    if unresolved:
        st.warning(
            f"{len(unresolved)} statement(s) could not be resolved statically. "
            "Their lineage is either missing or marked with low confidence."
        )
        with st.expander(f"Unresolved ({len(unresolved)})", expanded=False):
            for item in unresolved:
                label = UNRESOLVED_LABELS.get(item.get("reason", ""), item.get("reason", "unknown"))
                st.markdown(f"**{label}** — line {item.get('line_start', '?')}")
                st.caption(item.get("detail") or "")
                if item.get("sql_fragment"):
                    st.code(shorten_text(item["sql_fragment"], 400), language="sql")
    else:
        st.caption("All statements resolved statically.")

    if statements:
        with st.expander(f"Statements ({len(statements)})", expanded=False):
            for stmt in statements:
                flags = []
                if stmt.get("is_dynamic"):
                    flags.append("dynamic")
                if stmt.get("control_flow"):
                    flags.append(" / ".join(stmt["control_flow"]))
                suffix = f" · {', '.join(flags)}" if flags else ""
                target = stmt.get("target") or "—"
                st.markdown(f"`{stmt.get('kind')}` → `{target}` (line {stmt.get('line_start')}){suffix}")
                st.code(shorten_text(stmt.get("sql") or "", 300), language="sql")


def render_table_lineage_panel(result: dict[str, Any]) -> None:
    table_lineage = result.get("table_lineage") or {}
    target = table_lineage.get("target") or "—"
    sources = table_lineage.get("sources") or []
    statement_type = (table_lineage.get("statement_type") or "unknown").upper()

    st.subheader("Table lineage")
    st.caption(f"Statement: `{statement_type}` · click a source table to highlight")

    st.markdown("**Target table**")
    st.code(target, language=None)

    st.markdown("**Source tables**")
    if not sources:
        st.warning("No physical source tables detected.")
    else:
        cols_per_row = 2
        for row_start in range(0, len(sources), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for offset, source in enumerate(sources[row_start : row_start + cols_per_row]):
                with row_cols[offset]:
                    selected = st.session_state.selected_source_table == source
                    if st.button(
                        source,
                        key=f"tbl_btn_{row_start + offset}",
                        type="primary" if selected else "secondary",
                        width="stretch",
                    ):
                        st.session_state.selected_source_table = source
                        st.rerun()

    highlight = st.session_state.selected_source_table
    dot = build_table_lineage_dot(target, sources, highlight=highlight)
    if target or sources:
        st.markdown("**Lineage graph**")
        st.graphviz_chart(dot)

    if highlight:
        linked_columns = columns_linked_to_source_table(result, highlight)
        st.markdown(f"**Columns fed by `{highlight}`**")
        if linked_columns:
            st.write(", ".join(f"`{name}`" for name in linked_columns))
        else:
            st.caption("No column-level dependencies resolved for this table (try Column level).")

    with st.expander("Raw table lineage JSON"):
        st.json(table_lineage)


def render_column_lineage_panel(result: dict[str, Any]) -> None:
    st.subheader("Column lineage")
    st.caption("Click a column to view `table.column` source-to-target lineage.")

    columns = target_columns_from_result(result)
    if not columns:
        st.warning("No output columns found.")
        return

    cols_per_row = 3
    for row_start in range(0, len(columns), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for offset, alias in enumerate(columns[row_start : row_start + cols_per_row]):
            with row_cols[offset]:
                selected = st.session_state.selected_column == alias
                if st.button(
                    alias,
                    key=f"col_btn_{alias}",
                    type="primary" if selected else "secondary",
                    width="stretch",
                ):
                    st.session_state.selected_column = alias
                    st.session_state.selected_source_table = None
                    st.rerun()

    selected = st.session_state.selected_column
    if not selected:
        return

    st.divider()
    st.subheader(f"Lineage: `{selected}`")

    extraction = result.get("extraction") or {}
    record = column_record(extraction, selected)
    if record:
        st.markdown("**Expression**")
        st.code(record.get("expression") or selected, language="sql")

        derivation_kind = record.get("derivation_kind")
        if derivation_kind:
            st.caption(f"Derivation: `{derivation_kind}`")

        literal_values = record.get("literal_values") or []
        if literal_values:
            st.markdown("**Literal source expressions**")
            st.code("\nUNION ALL\n".join(literal_values), language="sql")

        deps = record.get("dependencies") or []
        st.markdown("**Direct dependencies**")
        if deps:
            dep_rows = [
                {
                    "table_alias": dep.get("table_alias") or "—",
                    "column": dep.get("column") or "—",
                    "physical_table": dep.get("physical_table") or "—",
                }
                for dep in deps
            ]
            st.dataframe(dep_rows, width="stretch", hide_index=True)
        elif derivation_kind == "literal":
            st.write("No table-column sources — value is hardcoded in SQL.")
        else:
            st.write("No direct table-column dependencies recorded.")

        union_branches = record.get("union_branches") or []
        if union_branches:
            st.markdown("**UNION branch lineage**")
            branch_rows = []
            for branch in union_branches:
                branch_rows.append(
                    {
                        "branch": branch.get("branch_index"),
                        "kind": branch.get("kind"),
                        "table_alias": branch.get("table_alias") or "—",
                        "physical_table": branch.get("physical_table") or "—",
                        "column": branch.get("column") or "—",
                        "literal_value": branch.get("literal_value") or "—",
                    }
                )
            st.dataframe(branch_rows, width="stretch", hide_index=True)

        if not literal_values and not union_branches and not deps:
            cte_ctx = cte_derivation_context(extraction, record, result.get("simplified_query") or {})
            if cte_ctx:
                cte_col = cte_ctx["column"]
                st.info(
                    f"Column is produced in CTE `{cte_ctx['cte_name']}` from constants "
                    f"(UNION branches), not from a physical source column."
                )
                st.markdown("**CTE column expression**")
                st.code(cte_col.get("expression") or cte_ctx["cte_name"], language="sql")

    graph_json = result.get("graph") or {}
    dot = build_column_lineage_dot(graph_json, selected)
    if dot:
        st.markdown("**Lineage graph**")
        st.graphviz_chart(dot)
    else:
        st.caption("Graph node not found for this column.")

    with st.expander("Raw column JSON"):
        st.json(record or {})


def paste_widget_key() -> str:
    return f"sql_paste_{st.session_state.get('sql_paste_nonce', 0)}"


def sync_paste_widget(content: str) -> None:
    st.session_state[paste_widget_key()] = content


def set_uploaded_sql(content: str, *, filename: str = "", mode: str) -> None:
    st.session_state.uploaded_sql = content
    st.session_state.source_filename = filename
    st.session_state.input_mode = mode
    sync_paste_widget(content)


def load_sql_from_upload() -> None:
    """Sync session SQL from the file_uploader widget."""
    file = st.session_state.get("sql_file_uploader")
    if file is None:
        return
    content = file.getvalue().decode("utf-8", errors="replace")
    set_uploaded_sql(content, filename=file.name, mode="upload")


def load_sql_from_paste() -> None:
    """Sync session SQL when the user edits the paste area."""
    pasted = st.session_state.get(paste_widget_key(), "")
    if pasted.strip():
        st.session_state.uploaded_sql = pasted
        st.session_state.source_filename = ""
        st.session_state.input_mode = "paste"


def load_sample_sql(relative_path: str, *, label: str) -> None:
    sample_path = ROOT / relative_path
    if not sample_path.exists():
        return
    content = sample_path.read_text(encoding="utf-8")
    set_uploaded_sql(content, filename=label, mode="sample")
    st.session_state.pop("sql_file_uploader", None)


def load_first_ddl10_statement() -> None:
    sql_path = ROOT / "data" / "DDLs_10.txt"
    if not sql_path.exists():
        return
    first = split_sql_statements(sql_path.read_text(encoding="utf-8"))[0]
    set_uploaded_sql(first, filename="DDLs_10.txt (statement 1)", mode="sample")
    st.session_state.pop("sql_file_uploader", None)


def clear_sql_input() -> None:
    st.session_state.uploaded_sql = ""
    st.session_state.source_filename = ""
    st.session_state.input_mode = ""
    st.session_state.pipeline_result = None
    st.session_state.selected_column = None
    st.session_state.selected_source_table = None
    st.session_state.selected_statement_index = 0
    st.session_state.sql_content_hash = ""
    st.session_state.analyzed_sql_hash = ""
    st.session_state.pop("statement_selector", None)
    st.session_state.pop("sql_file_uploader", None)
    st.session_state.sql_paste_nonce = st.session_state.get("sql_paste_nonce", 0) + 1


# ---------------------------------------------------------------------------
# Upload + SQL input
# ---------------------------------------------------------------------------
if "sql_paste_nonce" not in st.session_state:
    st.session_state.sql_paste_nonce = 0
if "input_mode" not in st.session_state:
    st.session_state.input_mode = ""

paste_key = paste_widget_key()
if paste_key not in st.session_state:
    st.session_state[paste_key] = st.session_state.uploaded_sql

st.subheader("SQL input")
input_tab_upload, input_tab_paste = st.tabs(["Upload file", "Paste SQL"])

with input_tab_upload:
    uploaded = st.file_uploader(
        "Choose or drop a SQL file",
        type=["sql", "txt"],
        accept_multiple_files=False,
        key="sql_file_uploader",
        on_change=load_sql_from_upload,
        help="Supports .sql and .txt files. Content appears in the editor and preview below.",
    )
    # Streamlit may not fire on_change on the same run as selection — handle inline too.
    if uploaded is not None:
        content = uploaded.getvalue().decode("utf-8", errors="replace")
        if (
            st.session_state.get("source_filename") != uploaded.name
            or st.session_state.get("uploaded_sql") != content
        ):
            set_uploaded_sql(content, filename=uploaded.name, mode="upload")

    sample_col1, sample_col2 = st.columns(2)
    with sample_col1:
        st.button(
            "Load `data/DDLs_10.txt`",
            width="stretch",
            on_click=load_sample_sql,
            kwargs={"relative_path": "data/DDLs_10.txt", "label": "data/DDLs_10.txt"},
        )
    with sample_col2:
        st.button(
            "Load first statement only",
            width="stretch",
            on_click=load_first_ddl10_statement,
        )

    if st.session_state.source_filename and st.session_state.input_mode in {"upload", "sample"}:
        chars = len(st.session_state.uploaded_sql)
        stmts = len(split_sql_statements(st.session_state.uploaded_sql))
        st.success(f"**{st.session_state.source_filename}** — {chars:,} chars, {stmts} statement(s)")

with input_tab_paste:
    st.text_area(
        "SQL query",
        height=220,
        placeholder="INSERT INTO target SELECT ...",
        key=paste_key,
        on_change=load_sql_from_paste,
        label_visibility="collapsed",
    )

run_col, clear_col = st.columns([1, 5])
with run_col:
    analyze = st.button("Analyze lineage", type="primary", width="stretch")
with clear_col:
    st.button("Clear", width="stretch", on_click=clear_sql_input)

sql_text = st.session_state.uploaded_sql.strip()
statements, sql_to_run, selected_statement_index = resolve_active_sql(sql_text, dialect)
target_table_labels: list[str] = []

if len(statements) > 1:
    target_table_labels = build_target_table_labels(statements, dialect)
    selected_statement_index = st.selectbox(
        "Target table",
        options=list(range(len(statements))),
        format_func=lambda idx: target_table_labels[idx],
        index=selected_statement_index,
        key="statement_selector",
    )
    st.session_state.selected_statement_index = selected_statement_index
    sql_to_run = statements[selected_statement_index]
    with st.expander("Preview selected statement", expanded=False):
        st.code(sql_to_run, language="sql")

active_sql_hash = hashlib.sha256(sql_to_run.encode("utf-8")).hexdigest()[:16] if sql_to_run else ""

if analyze and sql_to_run:
    steps_placeholder = st.empty()
    live_steps: dict[str, Any] = {}

    def on_pipeline_step(step_name: str, _step_data: dict[str, Any], all_steps: dict[str, Any]) -> None:
        live_steps.clear()
        live_steps.update(all_steps)
        with steps_placeholder.container():
            st.markdown("**Pipeline progress**")
            render_pipeline_steps(all_steps, running_step=step_name)

    with st.status("Running pipeline…", expanded=True) as status:
        try:
            registry = None
            if schema_ddl.strip():
                registry = SchemaRegistry(dialect=dialect).load_ddl(schema_ddl)
            st.session_state.pipeline_result = run_column_pipeline(
                sql_to_run,
                dialect=dialect,
                use_llm_verify=use_llm_verify,
                use_llm_enhance=use_llm_enhance,
                parse_plpgsql=parse_plpgsql,
                hf_token=hf_token or None,
                hf_model=hf_model,
                hf_provider=hf_provider,
                use_llm_cache=use_llm_cache,
                replace_cache_if_better=replace_cache_if_better,
                schema_registry=registry,
                step_callback=on_pipeline_step,
            )
            golden_match = golden_fixture_for_sql(sql_to_run)
            if golden_match and st.session_state.pipeline_result.get("graph"):
                _, golden_path = golden_match
                st.session_state.pipeline_result["golden_metrics"] = edge_f1_vs_golden(
                    st.session_state.pipeline_result["graph"],
                    golden_path,
                )
            else:
                st.session_state.pipeline_result["golden_metrics"] = None
            st.session_state.selected_column = None
            st.session_state.selected_source_table = None
            st.session_state.analyzed_sql_hash = active_sql_hash
            status.update(label="Pipeline complete", state="complete")
        except Exception as exc:
            st.session_state.pipeline_result = None
            status.update(label="Pipeline failed", state="error")
            st.error(f"Analysis failed: {exc}")

# ---------------------------------------------------------------------------
# Main layout: SQL viewer + column lineage
# ---------------------------------------------------------------------------
if not sql_text and not st.session_state.pipeline_result:
    st.info("Upload a SQL file or paste a query, then click **Analyze lineage**.")
    st.stop()

left, right = st.columns([1, 1])

with left:
    st.subheader("SQL")
    if st.session_state.source_filename:
        st.caption(f"Source: `{st.session_state.source_filename}`")
    if len(statements) > 1:
        st.caption(
            f"Target table: `{target_table_labels[selected_statement_index]}` "
            f"(statement {selected_statement_index + 1} of {len(statements)})"
        )
    display_sql = sql_to_run if sql_to_run else sql_text
    st.code(display_sql or "(empty)", language="sql")
    if len(statements) > 1 and sql_text:
        with st.expander("Full uploaded script"):
            st.code(sql_text, language="sql")

with right:
    result = st.session_state.pipeline_result
    if result is None:
        st.subheader("Lineage")
        st.caption("Run analysis to explore table or column lineage.")
    elif active_sql_hash and st.session_state.analyzed_sql_hash != active_sql_hash:
        st.subheader("Lineage")
        st.warning("Statement selection changed — click **Analyze lineage** to refresh results.")
    else:
        stage = result.get("pipeline_stage", "unknown")
        simplified = result.get("simplified_query") or {}
        target_table = simplified.get("target_table") or "—"
        llm_config = result.get("llm_config") or {}
        if llm_config:
            st.caption(
                f"LLM `{shorten_text(llm_config.get('model', DEFAULT_MODEL_NAME), 36)}` "
                f"· `{llm_config.get('provider', DEFAULT_PROVIDER)}`"
            )

        col_count = len(target_columns_from_result(result))
        st.caption(
            compact_pipeline_summary(stage, target_table, col_count)
        )
        cache_info = result.get("cache") or {}
        if cache_info:
            if cache_info.get("hit"):
                st.caption(
                    f"Cache hit · quality `{cache_info.get('quality_score', '—')}`"
                )
            elif cache_info.get("updated"):
                prev = cache_info.get("previous_quality_score")
                prev_text = f" (prev `{prev:.2f}`)" if prev is not None else ""
                st.caption(
                    f"Cache updated · quality `{cache_info.get('quality_score', '—')}`{prev_text}"
                )
            elif cache_info.get("updated") is False and cache_info.get("previous_quality_score") is not None:
                st.caption(
                    "Cache kept — fresh result did not beat cached quality "
                    f"(`{cache_info.get('quality_score', '—')}` vs `{cache_info.get('previous_quality_score', '—')}`)."
                )
            elif not cache_info.get("read_enabled"):
                st.caption("Cache bypassed for this run.")
        if len(str(target_table)) > 32:
            with st.expander("Full target table name"):
                st.code(str(target_table), language=None)

        golden_metrics = result.get("golden_metrics")
        if golden_metrics and st.session_state.lineage_level == "Column":
            f1 = golden_metrics.get("f1", 0.0)
            f1_delta = f1 - 0.9
            f1_help = (
                f"vs golden `{golden_metrics.get('fixture')}` "
                f"(tp={golden_metrics.get('tp', 0)}, "
                f"fp={golden_metrics.get('fp', 0)}, "
                f"fn={golden_metrics.get('fn', 0)})"
            )
            g1, g2, g3, g4 = st.columns(4)
            g1.metric(
                "Edge F1",
                f"{f1:.3f}",
                delta=f"{f1_delta:+.3f} vs 0.9",
                delta_color="normal" if f1 >= 0.9 else "inverse",
                help=f1_help,
            )
            g2.metric("Precision", f"{golden_metrics.get('precision', 0.0):.3f}")
            g3.metric("Recall", f"{golden_metrics.get('recall', 0.0):.3f}")
            g4.metric("Golden fixture", golden_metrics.get("fixture", "—"))
            if f1 < 0.9:
                st.warning(
                    "Edge F1 is below the 0.9 regression threshold. "
                    "Golden baseline was built with deterministic parsing (no LLM)."
                )
            with st.expander("Golden set comparison details"):
                st.caption(f"Reference: `{golden_metrics.get('golden_path')}`")
                st.json(
                    {
                        "precision": golden_metrics.get("precision"),
                        "recall": golden_metrics.get("recall"),
                        "f1": golden_metrics.get("f1"),
                        "tp": golden_metrics.get("tp"),
                        "fp": golden_metrics.get("fp"),
                        "fn": golden_metrics.get("fn"),
                    }
                )
        else:
            st.caption(
                "Golden F1: no matching fixture for this SQL. "
                f"Known cases: {', '.join(case[2].stem.replace('_graph', '') for case in GOLDEN_CASES)}."
            )

        if result.get("pipeline_stage") == "plpgsql":
            render_plpgsql_panel(result)

        lineage_level = st.radio(
            "Lineage level",
            options=["Table", "Column"],
            horizontal=True,
            key="lineage_level",
            help="Table — source/target tables. Column — `table.column` dependencies.",
        )

        pipeline_steps = result.get("pipeline_steps") or {}
        if pipeline_steps:
            st.markdown("**Pipeline steps**")
            render_pipeline_steps(pipeline_steps)

            with st.expander("Pipeline step details"):
                labels = {
                    "chunking": "1. Chunking",
                    "parsing": "2. Parsing",
                    "verifying": "3. Verifying",
                    "enhancing": "4. Enhancing",
                    "combining": "5. Combining",
                }
                for step_name in SQL2GraphPipeline.PIPELINE_STEP_ORDER:
                    step = pipeline_steps.get(step_name)
                    if step:
                        st.markdown(f"**{labels[step_name]}**")
                        st.json(step)

            verification_diff = result.get("verification_diff")
            enhancement_diff = result.get("enhancement_diff")
            if verification_diff or enhancement_diff:
                with st.expander("LLM changes"):
                    render_extraction_diff("Verification changes (sqlglot → LLM verify)", verification_diff)
                    render_extraction_diff("Enhancement changes (verify → LLM enhance)", enhancement_diff)

        chunks = (result.get("chunks") or {}).get("chunks") or []
        if chunks:
            with st.expander(f"SQL chunks ({len(chunks)})"):
                for chunk in chunks:
                    st.markdown(f"**{chunk.get('name')}** (`{chunk.get('chunk_type')}`)")
                    st.code(chunk.get("sql") or "", language="sql")

        if result.get("warnings"):
            shown: set[str] = set()
            with st.expander("Warnings", expanded=False):
                for warning in result["warnings"]:
                    severity, message = friendly_llm_warning(str(warning))
                    if message in shown:
                        continue
                    shown.add(message)
                    if severity == "info":
                        st.info(message)
                    else:
                        st.warning(message)

        if lineage_level == "Table":
            render_table_lineage_panel(result)
        else:
            render_column_lineage_panel(result)

st.markdown("---")
st.caption("Built with sqlglot, SQL2Graph pipeline, and Streamlit")
