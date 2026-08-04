"""
Column-level SQL lineage — Streamlit web interface.

Workflow:
  1. Upload SQL (file drop zone or paste)
  2. Run sqlglot-first pipeline (optional LLM verify/enhance)
  3. Click a target column to inspect its source-to-target lineage
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import graphviz
import networkx as nx
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from Classes.schema_registry import SchemaRegistry  # noqa: E402
from Classes.sql2graph_classes import (  # noqa: E402
    SQL2GraphLLMExtractor,
    SQL2GraphParser,
    SQL2GraphPipeline,
    SQL2GraphVisualizer,
)

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
dialect = st.sidebar.selectbox("SQL dialect", ["postgres", "spark", "teradata", "hive"], index=0)
schema_ddl = st.sidebar.text_area(
    "Schema DDL (optional)",
    height=120,
    placeholder="CREATE TABLE schema.table (col1 int, col2 text);",
    help="Paste CREATE TABLE/VIEW DDL to resolve SELECT * and qualify columns.",
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
if (use_llm_verify or use_llm_enhance) and not hf_token:
    st.sidebar.warning("LLM enabled but no HF token — only sqlglot will run.")

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


def split_sql_statements(content: str) -> List[str]:
    try:
        import sqlparse

        return [str(stmt).strip() for stmt in sqlparse.parse(content) if str(stmt).strip()]
    except ImportError:
        return [s.strip() for s in content.split(";") if s.strip()]


def run_column_pipeline(
    sql: str,
    *,
    dialect: str,
    use_llm_verify: bool,
    use_llm_enhance: bool,
    hf_token: Optional[str],
    schema_registry: Optional[SchemaRegistry] = None,
    step_callback=None,
) -> Dict[str, Any]:
    """Run chunking → parsing → verifying → enhancing → combining."""
    parser = SQL2GraphParser(dialect=dialect, schema_registry=schema_registry)
    llm_extractor = None
    if (use_llm_verify or use_llm_enhance) and hf_token:
        llm_extractor = SQL2GraphLLMExtractor(hf_token=hf_token)

    pipeline = SQL2GraphPipeline(llm_extractor=llm_extractor, parser=parser)
    result = pipeline.run(
        sql,
        dialect=dialect,
        use_llm_verify=bool(llm_extractor and use_llm_verify),
        use_llm_enhance=bool(llm_extractor and use_llm_enhance),
        step_callback=step_callback,
    )
    if "error" in result:
        raise RuntimeError(result.get("error", "Pipeline failed"))
    return result


def render_pipeline_steps(steps: Dict[str, Any], running_step: Optional[str] = None) -> None:
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


def render_extraction_diff(title: str, diff: Optional[Dict[str, Any]]) -> None:
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


def target_columns_from_result(result: Dict[str, Any]) -> List[str]:
    extraction = result.get("extraction") or {}
    return [col.get("alias", "") for col in extraction.get("output_columns", []) if col.get("alias")]


def column_record(extraction: Dict[str, Any], alias: str) -> Optional[Dict[str, Any]]:
    for col in extraction.get("output_columns", []):
        if col.get("alias") == alias:
            return col
    return None


def cte_derivation_context(
    extraction: Dict[str, Any],
    record: Dict[str, Any],
    simplified: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
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


def resolve_output_node(graph: nx.MultiDiGraph, alias: str) -> Optional[str]:
    candidate = f"output.{alias}"
    if candidate in graph:
        return candidate
    for node, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "output_column" and attrs.get("alias") == alias:
            return node
    return None


def upstream_lineage_nodes(graph: nx.MultiDiGraph, output_node: str) -> Set[str]:
    """Collect nodes upstream of an output column via DERIVED_FROM / GROUPED_BY edges."""
    visited: Set[str] = set()
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


def build_column_lineage_dot(graph_json: Dict[str, Any], column_alias: str) -> Optional[graphviz.Digraph]:
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


# ---------------------------------------------------------------------------
# Upload + SQL input
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Drop SQL file here (.sql / .txt)",
    type=["sql", "txt"],
    help="Upload a file or paste SQL below.",
)

col_upload, col_paste = st.columns([1, 1])
with col_upload:
    if uploaded is not None:
        st.session_state.uploaded_sql = uploaded.getvalue().decode("utf-8")
        st.session_state.source_filename = uploaded.name

with col_paste:
    pasted = st.text_area(
        "Or paste SQL",
        value=st.session_state.uploaded_sql,
        height=160,
        placeholder="INSERT INTO ... SELECT ...",
        key="sql_paste_area",
    )
    if pasted.strip():
        st.session_state.uploaded_sql = pasted

run_col, clear_col = st.columns([1, 5])
with run_col:
    analyze = st.button("Analyze lineage", type="primary", width="stretch")
with clear_col:
    if st.button("Clear"):
        st.session_state.pipeline_result = None
        st.session_state.selected_column = None
        st.session_state.uploaded_sql = ""
        st.session_state.source_filename = ""
        st.rerun()

sql_text = st.session_state.uploaded_sql.strip()

if analyze and sql_text:
    statements = split_sql_statements(sql_text)
    if len(statements) > 1:
        st.info(f"Multiple statements detected ({len(statements)}). Analyzing the first one.")
    sql_to_run = statements[0]

    steps_placeholder = st.empty()
    live_steps: Dict[str, Any] = {}

    def on_pipeline_step(step_name: str, _step_data: Dict[str, Any], all_steps: Dict[str, Any]) -> None:
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
                hf_token=hf_token or None,
                schema_registry=registry,
                step_callback=on_pipeline_step,
            )
            st.session_state.selected_column = None
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
    st.code(sql_text or "(empty)", language="sql")

with right:
    result = st.session_state.pipeline_result
    if result is None:
        st.subheader("Target columns")
        st.caption("Run analysis to list output columns.")
    else:
        stage = result.get("pipeline_stage", "unknown")
        simplified = result.get("simplified_query") or {}
        target_table = simplified.get("target_table") or "—"

        m1, m2, m3 = st.columns(3)
        m1.metric("Pipeline stage", stage)
        m2.metric("Target table", target_table if len(str(target_table)) < 24 else str(target_table)[:21] + "…")
        m3.metric("Output columns", len(target_columns_from_result(result)))

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
            with st.expander("Warnings"):
                for warning in result["warnings"]:
                    st.write(f"- {warning}")

        st.subheader("Target columns")
        st.caption("Click a column to view its source-to-target lineage.")

        columns = target_columns_from_result(result)
        if not columns:
            st.warning("No output columns found.")
        else:
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
                            st.rerun()

        selected = st.session_state.selected_column
        if selected:
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

st.markdown("---")
st.caption("Built with sqlglot, SQL2Graph pipeline, and Streamlit")
