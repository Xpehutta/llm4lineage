"""Results / lineage inspection panels for the Streamlit UI."""

from __future__ import annotations

from typing import Any

import streamlit as st

from Classes.pipeline.llm_helpers import DEFAULT_MODEL_NAME, DEFAULT_PROVIDER
from Classes.sql2graph_classes import SQL2GraphPipeline
from Web.components.graph_view import build_column_lineage_dot, build_table_lineage_dot
from Web.services.cache_service import cache_status_captions
from Web.services.pipeline_service import (
    GOLDEN_CASES,
    UNRESOLVED_LABELS,
    column_record,
    columns_linked_to_source_table,
    compact_pipeline_summary,
    cte_derivation_context,
    friendly_llm_warning,
    shorten_text,
    target_columns_from_result,
)


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


def render_sql_viewer(
    *,
    sql_text: str,
    sql_to_run: str,
    statements: list[str],
    selected_statement_index: int,
    target_table_labels: list[str],
) -> None:
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


def render_lineage_results(result: dict[str, Any] | None, active_sql_hash: str) -> None:
    """Render the right-hand lineage panel for the current pipeline result."""
    if result is None:
        st.subheader("Lineage")
        st.caption("Run analysis to explore table or column lineage.")
        return
    if active_sql_hash and st.session_state.analyzed_sql_hash != active_sql_hash:
        st.subheader("Lineage")
        st.warning("Statement selection changed — click **Analyze lineage** to refresh results.")
        return

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
    st.caption(compact_pipeline_summary(stage, target_table, col_count))
    for caption in cache_status_captions(result.get("cache") or {}):
        st.caption(caption)
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
