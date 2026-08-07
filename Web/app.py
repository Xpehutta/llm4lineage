"""
Column-level SQL lineage — Streamlit web interface.

Workflow:
  1. Upload SQL (file drop zone or paste)
  2. Run sqlglot-first pipeline (optional LLM verify/enhance)
  3. Click a target column to inspect its source-to-target lineage
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from Web.components.results_panel import (  # noqa: E402
    render_lineage_results,
    render_pipeline_steps,
    render_sql_viewer,
)
from Web.components.sidebar import render_sidebar  # noqa: E402
from Web.components.uploader import render_sql_input  # noqa: E402
from Web.services.pipeline_service import (  # noqa: E402
    build_schema_registry,
    edge_f1_vs_golden,
    golden_fixture_for_sql,
    plpgsql_table_lineage,
    run_column_pipeline,
)

# Re-export for tests / callers that import from Web.app
__all__ = ["plpgsql_table_lineage", "run_column_pipeline"]

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Column Lineage Explorer", page_icon="🔍", layout="wide")

st.title("Column Lineage Explorer")
st.caption("chunking → parsing → verifying → enhancing → combining → per-column lineage")

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

# ---------------------------------------------------------------------------
# Sidebar + SQL input
# ---------------------------------------------------------------------------
config = render_sidebar()
analyze, statements, sql_to_run, selected_statement_index, target_table_labels = render_sql_input(
    config.dialect
)

sql_text = st.session_state.uploaded_sql.strip()
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
            # Sidebar DDL + CREATE TABLE/VIEW from the full uploaded script
            # (so CTAS/INSERT can resolve SELECT * against earlier DDL in the file).
            registry = build_schema_registry(
                config.dialect,
                schema_ddl=config.schema_ddl,
                sql_script=sql_text,
            )
            st.session_state.pipeline_result = run_column_pipeline(
                sql_to_run,
                dialect=config.dialect,
                use_llm_verify=config.use_llm_verify,
                use_llm_enhance=config.use_llm_enhance,
                parse_plpgsql=config.parse_plpgsql,
                hf_token=config.hf_token or None,
                hf_model=config.hf_model,
                hf_provider=config.hf_provider,
                use_llm_cache=config.use_llm_cache,
                replace_cache_if_better=config.replace_cache_if_better,
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
    render_sql_viewer(
        sql_text=sql_text,
        sql_to_run=sql_to_run,
        statements=statements,
        selected_statement_index=selected_statement_index,
        target_table_labels=target_table_labels,
    )

with right:
    render_lineage_results(st.session_state.pipeline_result, active_sql_hash)

st.markdown("---")
st.caption("Built with sqlglot, SQL2Graph pipeline, and Streamlit")
