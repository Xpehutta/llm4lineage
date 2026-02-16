"""
SQL Lineage Tool – Streamlit Web Interface
------------------------------------------
- Single query lineage extraction (JSON + graph)
- Batch processing of multiple SQL statements from .sql/.txt files
- Table lookup: see where a table is target or source, with graph
- Full SQL display with built‑in copy button (no pyperclip needed)
"""

import sys
from pathlib import Path

# ----- Allow import of project modules (if not installed as package) -----
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import json
import os
import graphviz
import pandas as pd
from io import StringIO
from typing import List, Dict, Any

# ----- Import our core classes (clean import after package install) -----
try:
    from Classes import SQLLineageExtractor, SQLLineageResult
except ImportError:
    from Classes.model_classes import SQLLineageExtractor, SQLLineageResult

# ----------------------------------------------------------------------
# Page configuration
st.set_page_config(
    page_title="SQL Lineage Tool",
    page_icon="🔍",
    layout="wide"
)

# ----------------------------------------------------------------------
# Sidebar – LLM configuration
st.sidebar.title("⚙️ Configuration")
hf_token = st.sidebar.text_input(
    "Hugging Face Token",
    type="password",
    value=os.environ.get("HF_TOKEN", ""),
    help="Get your token at huggingface.co/settings/tokens"
)
model = st.sidebar.text_input(
    "Model",
    value="Qwen/Qwen3-Coder-30B-A3B-Instruct"
)
provider = st.sidebar.selectbox(
    "Provider",
    ["scaleway", "nebius", "huggingface"],
    index=0
)
max_tokens = st.sidebar.slider("Max new tokens", 256, 4096, 2048, step=256)
do_sample = st.sidebar.checkbox("Do sample", value=False)
max_retries = st.sidebar.slider("Max retries", 1, 10, 5)

# ----------------------------------------------------------------------
# Cached extractor initialisation
@st.cache_resource
def get_extractor(token, model, provider, max_tokens, do_sample, max_retries):
    return SQLLineageExtractor(
        model=model,
        provider=provider,
        hf_token=token,
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        max_retries=max_retries
    )

if not hf_token:
    st.sidebar.warning("⚠️ Please enter your Hugging Face token")
    st.stop()

try:
    extractor = get_extractor(hf_token, model, provider, max_tokens, do_sample, max_retries)
    st.sidebar.success("✅ Extractor ready")
except Exception as e:
    st.sidebar.error(f"❌ Initialization failed: {e}")
    st.stop()

# ----------------------------------------------------------------------
# Session state initialisation
if "lineage_results" not in st.session_state:
    st.session_state.lineage_results = []          # list of dicts for every processed statement
if "processed_statement_keys" not in st.session_state:
    st.session_state.processed_statement_keys = set()   # unique IDs to avoid reprocessing
if "lookup_table" not in st.session_state:
    st.session_state.lookup_table = ""             # table name for lookup (bound to text input)

# ----------------------------------------------------------------------
# Helper: split SQL statements by semicolon (with sqlparse if available)
def split_sql_statements(content: str) -> List[str]:
    """Split a string containing multiple SQL statements separated by ;"""
    try:
        import sqlparse
        return [str(stmt).strip() for stmt in sqlparse.parse(content) if str(stmt).strip()]
    except ImportError:
        # fallback: simple split (may break on strings containing ';')
        return [s.strip() for s in content.split(';') if s.strip()]

# ----------------------------------------------------------------------
# Callback: update session state when user types in the lookup input
def on_lookup_input_change():
    st.session_state.lookup_table = st.session_state.table_lookup_input

# ----------------------------------------------------------------------
# Main tabs
tab1, tab2 = st.tabs(["🔍 Single Query Lineage", "📊 Table Lineage (Batch)"])

# ######################################################################
# TAB 1 – Single Query Lineage
# ######################################################################
with tab1:
    st.header("Extract lineage from a single SQL query")

    example_sql = """INSERT INTO analytics.sales_summary
SELECT p.product_category, SUM(s.amount) as total_sales
FROM (SELECT product_id, category as product_category FROM products.raw_data) p
JOIN sales.transactions s ON p.product_id = s.product_id
GROUP BY p.product_category"""

    sql_input = st.text_area(
        "SQL Query",
        value=example_sql,
        height=250,
        placeholder="-- Your SQL here ...",
        key="single_sql_input"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("🚀 Extract Lineage", type="primary", key="run_single")
    with col2:
        if st.button("📋 Clear", key="clear_single"):
            if "result" in st.session_state:
                del st.session_state["result"]
            st.rerun()

    if run_btn and sql_input.strip():
        with st.spinner("Extracting lineage ..."):
            result = extractor.extract(sql_input)
            st.session_state["result"] = result

    if "result" in st.session_state and st.session_state["result"]:
        result = st.session_state["result"]

        if "error" in result:
            st.error(f"❌ {result['error']}")
        else:
            target = result.get("target", "")
            sources = result.get("sources", [])

            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Target Table", target if target else "—")
            with col2:
                st.metric("Number of Sources", len(sources))
            with col3:
                st.metric("Status", "✅ Success")

            # JSON Output
            st.subheader("📦 JSON Result")
            st.json(result)

            # Graph
            if target and sources:
                st.subheader("🕸️ Lineage Graph")
                dot = graphviz.Digraph(comment="SQL Lineage")
                dot.attr(rankdir="LR")
                dot.node("target", target, shape="box", style="filled", fillcolor="lightblue")
                for idx, src in enumerate(sources[:20]):   # limit to 20 for readability
                    node_id = f"source_{idx}"
                    dot.node(node_id, src, shape="ellipse", style="filled", fillcolor="lightgreen")
                    dot.edge(node_id, "target")
                st.graphviz_chart(dot)
                if len(sources) > 20:
                    st.caption(f"⚠️ Only first 20 sources are shown (total {len(sources)}).")

            # Download JSON
            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(result, indent=2),
                file_name="lineage_result.json",
                mime="application/json"
            )

# ######################################################################
# TAB 2 – Table Lineage (Batch upload, multi‑statement files)
# ######################################################################
with tab2:
    st.header("📁 Upload SQL files (multiple statements per file, delimiter ';')")

    uploaded_files = st.file_uploader(
        "Choose one or more .sql or .txt files",
        type=["sql", "txt"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    # --- Process newly uploaded files/statements ---
    if uploaded_files:
        new_entries = []   # (filename, stmt_idx, stmt_text, unique_key)

        for file in uploaded_files:
            content = file.getvalue().decode("utf-8")
            statements = split_sql_statements(content)

            for idx, stmt in enumerate(statements):
                key = f"{file.name}#{idx}"
                if key not in st.session_state.processed_statement_keys:
                    new_entries.append((file.name, idx + 1, stmt, key))

        if new_entries:
            progress_bar = st.progress(0, text="Processing SQL statements...")
            for i, (fname, stmt_no, full_stmt, key) in enumerate(new_entries):
                try:
                    result = extractor.extract(full_stmt)
                    if "error" in result:
                        st.warning(f"⚠️ {fname} (stmt {stmt_no}): {result['error']}")
                    else:
                        st.session_state.lineage_results.append({
                            "filename": fname,
                            "statement_index": stmt_no,
                            "query_preview": full_stmt[:200] + "..." if len(full_stmt) > 200 else full_stmt,
                            "full_sql": full_stmt,               # store the entire SQL for copying
                            "target": result.get("target", ""),
                            "sources": result.get("sources", []),
                            "raw_result": result
                        })
                        st.session_state.processed_statement_keys.add(key)
                except Exception as e:
                    st.error(f"❌ Error processing {fname} (stmt {stmt_no}): {e}")

                # update progress bar
                progress_bar.progress((i + 1) / len(new_entries))

            st.success(f"✅ Processed {len(new_entries)} new statement(s). "
                       f"Total in session: {len(st.session_state.lineage_results)}")

    # --- Display stored results and table lookup ---
    if st.session_state.lineage_results:
        st.subheader("📋 Extracted Lineage Overview")

        # ---- Custom table with clickable target buttons ----
        col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
        col1.markdown("**File**")
        col2.markdown("**Stmt#**")
        col3.markdown("**Target (click to lookup)**")
        col4.markdown("**Sources**")

        for idx, r in enumerate(st.session_state.lineage_results):
            col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
            with col1:
                st.text(r["filename"])
            with col2:
                st.text(str(r.get("statement_index", 1)))
            with col3:
                # Clickable button styled as a link
                if st.button(
                    r["target"] if r["target"] else "—",
                    key=f"target_btn_{idx}",
                    help="Click to set this table as lookup target",
                    use_container_width=True
                ):
                    st.session_state.lookup_table = r["target"]
                    st.rerun()
            with col4:
                st.text(str(len(r["sources"])))

        st.divider()

        # --- Table lookup ---
        st.subheader("🔎 Look up a table")

        # Text input bound to session state
        st.text_input(
            "Enter fully qualified table name (e.g., schema.table)",
            key="table_lookup_input",
            value=st.session_state.lookup_table,
            on_change=on_lookup_input_change,
            placeholder="e.g., analytics.sales_summary"
        )

        table_name = st.session_state.lookup_table.strip().lower()

        if table_name:
            # where table is target
            as_target = [r for r in st.session_state.lineage_results if r["target"] == table_name]
            # where table is source
            as_source = [r for r in st.session_state.lineage_results if table_name in r["sources"]]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("As Target", len(as_target))
            with col2:
                st.metric("As Source", len(as_source))

            # --- As Target ---
            if as_target:
                st.markdown("**🎯 Appears as TARGET in:**")
                for r in as_target:
                    label = f"📄 {r['filename']} (stmt {r.get('statement_index', 1)})"
                    with st.expander(label):
                        st.write(f"**Target:** `{r['target']}`")
                        st.write(f"**Sources ({len(r['sources'])}):**")
                        for src in r['sources'][:10]:
                            st.write(f"- `{src}`")
                        if len(r['sources']) > 10:
                            st.caption(f"... and {len(r['sources'])-10} more")

                        # Full SQL with built‑in copy button (just use st.code)
                        st.code(r['full_sql'], language="sql")

            # --- As Source ---
            if as_source:
                st.markdown("**📤 Appears as SOURCE in:**")
                for r in as_source:
                    label = f"📄 {r['filename']} (stmt {r.get('statement_index', 1)}) → `{r['target']}`"
                    with st.expander(label):
                        st.code(r['full_sql'], language="sql")

            # --- Graph for this table ---
            if as_target or as_source:
                st.subheader(f"🕸️ Lineage Graph for `{table_name}`")
                dot = graphviz.Digraph(comment=f"Lineage for {table_name}")
                dot.attr(rankdir="LR")

                # Upstream sources (tables that feed into this table)
                upstream = set()
                for r in as_target:
                    upstream.update(r["sources"])

                # Downstream targets (tables that use this table as source)
                downstream = set()
                for r in as_source:
                    downstream.add(r["target"])

                # central node
                dot.node("center", table_name, shape="box", style="filled", fillcolor="lightblue")

                # upstream (left side)
                for i, src in enumerate(sorted(upstream)[:15]):
                    node_id = f"up_{i}"
                    dot.node(node_id, src, shape="ellipse", style="filled", fillcolor="lightgreen")
                    dot.edge(node_id, "center")
                if len(upstream) > 15:
                    dot.node("up_more", "...", shape="plaintext")
                    dot.edge("up_more", "center")

                # downstream (right side)
                for i, tgt in enumerate(sorted(downstream)[:15]):
                    node_id = f"down_{i}"
                    dot.node(node_id, tgt, shape="box", style="filled", fillcolor="orange")
                    dot.edge("center", node_id)
                if len(downstream) > 15:
                    dot.node("down_more", "...", shape="plaintext")
                    dot.edge("center", "down_more")

                st.graphviz_chart(dot)

        # --- Clear all button ---
        if st.button("🧹 Clear all stored lineage results"):
            st.session_state.lineage_results = []
            st.session_state.processed_statement_keys = set()
            st.session_state.lookup_table = ""   # also clear the lookup field
            st.rerun()

    else:
        st.info("ℹ️ No lineage data yet. Upload one or more SQL files to begin.")

# ----------------------------------------------------------------------
st.markdown("---")
st.markdown("Built with ❤️ using LangChain, Hugging Face, and Streamlit")