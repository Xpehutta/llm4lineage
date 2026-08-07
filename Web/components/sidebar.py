"""Sidebar configuration widgets for the Streamlit UI."""

from __future__ import annotations

import os
from dataclasses import dataclass

import streamlit as st

from Classes.pipeline.llm_helpers import resolve_model_name, resolve_provider
from Web.services.cache_service import execute_llm_health_check, llm_config_key

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


@dataclass(frozen=True)
class SidebarConfig:
    hf_token: str
    hf_model: str
    hf_provider: str
    dialect: str
    schema_ddl: str
    parse_plpgsql: bool
    use_llm_verify: bool
    use_llm_enhance: bool
    use_llm_cache: bool
    replace_cache_if_better: bool


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


def render_sidebar() -> SidebarConfig:
    """Render sidebar controls and return the selected configuration."""
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

    return SidebarConfig(
        hf_token=hf_token,
        hf_model=hf_model,
        hf_provider=hf_provider,
        dialect=dialect,
        schema_ddl=schema_ddl,
        parse_plpgsql=parse_plpgsql,
        use_llm_verify=use_llm_verify,
        use_llm_enhance=use_llm_enhance,
        use_llm_cache=use_llm_cache,
        replace_cache_if_better=replace_cache_if_better,
    )
