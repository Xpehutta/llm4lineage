"""Central configuration for the SQL analysis pipeline."""

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Central configuration for the SQL analysis pipeline.

    All fields can be overridden via environment variables or a .env file.
    API keys are stored as SecretStr to prevent accidental leakage in logs.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # SQL parsing
    sql_dialect: str = "postgres"
    error_on_incomplete: bool = True

    # LLM provider selection
    llm_provider: str = Field(
        default="huggingface_inference",
        validation_alias=AliasChoices("llm_provider", "LLM_PROVIDER"),
    )

    # HuggingFace inference router (Classes default stack)
    model_name: str = Field(
        default="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        validation_alias=AliasChoices("model_name", "MODEL_NAME"),
    )
    inference_provider: str = Field(
        default="scaleway",
        validation_alias=AliasChoices("inference_provider", "PROVIDER"),
    )
    hf_do_sample: bool = False

    # OpenAI
    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = "gpt-4o"

    # Anthropic
    anthropic_api_key: SecretStr = SecretStr("")
    anthropic_model: str = "claude-3-haiku-20240307"

    # HuggingFace Endpoint / token
    hf_endpoint_url: str = ""
    hf_api_token: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("hf_api_token", "HF_API_TOKEN", "HF_TOKEN"),
    )
    hf_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    hf_max_new_tokens: int = 2048

    # Ollama (local)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # Common LLM parameters
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1
    #: Ask the provider to guarantee JSON output where it supports it, and cap
    #: sampling temperature elsewhere so the prompt's schema is followed.
    llm_json_mode: bool = True
    #: Highest sampling temperature allowed while `llm_json_mode` is on.
    llm_json_mode_max_temperature: float = 0.1
    llm_retry_attempts: int = 3
    llm_retry_min_wait: float = 2.0
    llm_retry_max_wait: float = 10.0

    # Prompt template files
    prompt_system_file: str = ""
    prompt_human_template_file: str = ""

    # Lineage settings
    lineage_include_intermediate_columns: bool = False

    # AST serialisation
    ast_max_depth: int = 50

    # Logging
    log_level: str = "INFO"
