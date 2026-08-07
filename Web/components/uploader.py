"""SQL upload / paste input widgets for the Streamlit UI."""

from __future__ import annotations

import hashlib

import streamlit as st

from Web.services.pipeline_service import (
    ROOT,
    build_target_table_labels,
    split_sql_statements,
)


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


def render_sql_input(dialect: str) -> tuple[bool, list[str], str, int, list[str]]:
    """
    Render SQL upload/paste UI and statement picker.

    Returns:
        (analyze_clicked, statements, sql_to_run, selected_index, target_table_labels)
    """
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

    return analyze, statements, sql_to_run, selected_statement_index, target_table_labels
