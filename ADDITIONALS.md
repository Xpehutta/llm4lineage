Specification: AI‑Driven SQL Parsing Pipeline with Flexible LLM Backends and Column‑Level Lineage

Version: 2.1
Status: Final
Date: 2026‑07‑28

---

1. Overview

This specification defines the Python classes that constitute a Target System built by an AI agent. The system:

1. Parses SQL using sqlglot.
2. Serialises the resulting abstract syntax tree (AST) to JSON.
3. Extracts column‑level, source‑to‑target lineage from the AST.
4. Feeds the AST JSON and lineage into a LangChain pipeline backed by a swappable Large Language Model (LLM) (OpenAI, Anthropic, Hugging Face, Ollama, or any LangChain‑compatible chat model).
5. Returns the LLM's response together with the parsed lineage and metadata.

The architecture guarantees provider agnosticism, meaning the LLM backend can be changed through configuration or runtime injection without altering core logic. The system is designed to be modular, testable, extensible, and production‑ready.

Design Principles

Principle Implementation
Provider agnosticism All model creation centralised in LLMFactory; no provider‑specific code in pipeline logic
Graceful degradation Pipeline failures are captured in PipelineResult.error rather than crashing the process
Secret safety API keys use pydantic.SecretStr; never logged or printed in plaintext
Resilience LLM invocations are wrapped with exponential‑backoff retry logic
Prompt isolation Prompts are loaded from external files with built‑in fallback defaults
Lineage as first‑class citizen Lineage is always computed, passed to the LLM, and returned in the result

---

1. Class Hierarchy & Dependencies

```
Config (pydantic‑settings, Pydantic V2)
  ├── fields for provider selection, credentials (SecretStr), and lineage options
  └── used by LLMFactory and orchestrator

SQLParser
  └── uses sqlglot

ASTSerializer
  └── uses SQLParser result
  └── iterative (stack‑based) traversal with configurable max_depth

ColumnLineageExtractor
  └── uses SQLParser result (AST)
  └── produces column lineage list
  └── handles SELECT *, aliased expressions, JOINs, subqueries

LLMFactory (utility)
  └── returns BaseChatModel based on Config
  └── extracts SecretStr values safely

SQLAnalysisChain (LangChain composition)
  ├── ChatPromptTemplate (system + human, with placeholders)
  ├── BaseChatModel (injected)
  ├── StrOutputParser
  └── tenacity retry wrapper

PipelineOrchestrator
  ├── owns: SQLParser, ASTSerializer, ColumnLineageExtractor, SQLAnalysisChain
  ├── coordinates: parse → serialize → lineage → LLM
  └── catches all pipeline exceptions → populates PipelineResult.error

PipelineResult (dataclass)
  ├── original_sql, ast_json, column_lineage, llm_response
  ├── latency_seconds, model_used, error

Custom Exceptions:
  - PipelineBaseError
  - ParsingError
  - SerializationError
  - LineageExtractionError
  - LLMCommunicationError
  - InvalidResponseError
```

---

1. Detailed Class Specifications

3.1 Config

Purpose: Immutable configuration loaded from environment variables or .env file. Supports all LLM providers and lineage options. Uses Pydantic V2 SettingsConfigDict and SecretStr for credential safety. Duplicate temperature fields merged into a single llm_temperature.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


class Config(BaseSettings):
    """Central configuration for the SQL analysis pipeline.

    All fields can be overridden via environment variables or a .env file.
    API keys are stored as SecretStr to prevent accidental leakage in logs.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # tolerate unexpected env vars without crashing
    )

    # ── SQL parsing ──────────────────────────────────────────────
    sql_dialect: str = "spark"
    error_on_incomplete: bool = True

    # ── LLM provider selection ───────────────────────────────────
    llm_provider: str = "openai"  # openai | anthropic | huggingface_endpoint |
                                  # huggingface_local | ollama | mock

    # ── OpenAI ───────────────────────────────────────────────────
    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = "gpt-4o"

    # ── Anthropic ────────────────────────────────────────────────
    anthropic_api_key: SecretStr = SecretStr("")
    anthropic_model: str = "claude-3-haiku-20240307"

    # ── HuggingFace Endpoint ─────────────────────────────────────
    hf_endpoint_url: str = ""
    hf_api_token: SecretStr = SecretStr("")
    hf_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    hf_max_new_tokens: int = 512

    # ── Ollama (local) ───────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # ── Common LLM parameters ────────────────────────────────────
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1        # used by all providers
    llm_retry_attempts: int = 3
    llm_retry_min_wait: float = 2.0    # seconds
    llm_retry_max_wait: float = 10.0   # seconds

    # ── Prompt template files ────────────────────────────────────
    prompt_system_file: str = "prompts/system.txt"
    prompt_human_template_file: str = "prompts/human.txt"

    # ── Lineage settings ─────────────────────────────────────────
    lineage_include_intermediate_columns: bool = False  # reserved for subclass extensions

    # ── AST serialisation ────────────────────────────────────────
    ast_max_depth: int = 50  # guard against pathological nesting

    # ── Logging ──────────────────────────────────────────────────
    log_level: str = "INFO"
```

---

3.2 SQLParser

Purpose: Wraps sqlglot.parse_one() to parse raw SQL into an AST. Raises ParsingError on failure.

```python
import logging
import sqlglot
from sqlglot.errors import ParseError

logger = logging.getLogger(__name__)


class SQLParser:
    """Parse a raw SQL string into a sqlglot AST."""

    def __init__(self, dialect: str = "spark", error_on_incomplete: bool = True):
        self.dialect = dialect
        self.error_on_incomplete = error_on_incomplete

    def parse(self, sql: str) -> sqlglot.exp.Expression:
        """Return the AST for *sql*.

        Raises:
            ParsingError: If sqlglot cannot parse the input.
        """
        if not sql or not sql.strip():
            raise ParsingError("Input SQL is empty or whitespace‑only.")

        try:
            tree = sqlglot.parse_one(
                sql,
                dialect=self.dialect,
                error_level=sqlglot.ErrorLevel.RAISE
                    if self.error_on_incomplete
                    else sqlglot.ErrorLevel.WARN,
            )
        except ParseError as exc:
            logger.error("SQL parse failure: %s", exc)
            raise ParsingError(f"Failed to parse SQL: {exc}") from exc

        if tree is None:
            raise ParsingError("Parser returned None for the given SQL.")

        logger.debug("Parsed SQL into AST root: %s", type(tree).__name__)
        return tree
```

---

3.3 ASTSerializer

Purpose: Converts a sqlglot AST into a JSON‑serializable Python dictionary using an iterative, stack‑based traversal (avoids RecursionError on deeply nested queries). Every node is represented as {"type": "...", "properties": {...}, "children": [...]}.

Properties that are primitive values (booleans, numbers, strings) are stored in their original types, never forcibly stringified. Pre‑defined convenience properties (like "distinct", "name", "alias") are set first and not overwritten by the generic loop.

```python
import logging
from typing import Any, Dict, List
import sqlglot.exp as exp

logger = logging.getLogger(__name__)


class ASTSerializer:
    """Serialise a sqlglot AST to a nested dictionary."""

    def __init__(self, max_depth: int = 50):
        self.max_depth = max_depth

    def serialize(self, tree: exp.Expression) -> Dict[str, Any]:
        """Return a JSON‑safe dict representation of *tree*.

        Raises:
            SerializationError: If traversal exceeds *max_depth*.
        """
        try:
            return self._serialize_iterative(tree)
        except Exception as exc:
            raise SerializationError(
                f"AST serialization failed: {exc}"
            ) from exc

    # ── internal ─────────────────────────────────────────────────

    def _serialize_iterative(self, root: exp.Expression) -> Dict[str, Any]:
        """Stack‑based DFS to avoid RecursionError on deep ASTs."""
        root_dict = self._make_node_dict(root)
        # stack items: (sqlglot_node, target_dict, current_depth)
        stack: List[tuple] = [(root, root_dict, 0)]

        while stack:
            node, node_dict, depth = stack.pop()

            if depth > self.max_depth:
                node_dict["properties"]["_truncated"] = True
                logger.warning(
                    "AST depth %d exceeded max_depth=%d; truncating subtree.",
                    depth, self.max_depth,
                )
                continue

            for key, value in node.args.items():
                if key in node_dict["properties"]:
                    # already handled by _make_node_dict; skip to preserve original type
                    continue

                if isinstance(value, exp.Expression):
                    child_dict = self._make_node_dict(value)
                    node_dict["children"].append(child_dict)
                    stack.append((value, child_dict, depth + 1))
                elif isinstance(value, list):
                    # Collect non‑expression items as a primitive list;
                    # expression items become children.
                    primitives = []
                    for item in value:
                        if isinstance(item, exp.Expression):
                            child_dict = self._make_node_dict(item)
                            node_dict["children"].append(child_dict)
                            stack.append((item, child_dict, depth + 1))
                        elif item is not None:
                            primitives.append(item)
                    if primitives:
                        node_dict["properties"][key] = primitives
                elif value is not None:
                    node_dict["properties"][key] = value  # keep original type (bool, int, float, str)

        return root_dict

    @staticmethod
    def _make_node_dict(node: exp.Expression) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": type(node).__name__,
            "properties": {},
            "children": [],
        }

        # Well‑known convenience properties (set before generic loop)
        if isinstance(node, exp.Table):
            result["properties"]["name"] = node.name
            result["properties"]["alias"] = node.alias_or_name
        elif isinstance(node, exp.Column):
            result["properties"]["name"] = node.name
            result["properties"]["table"] = node.table
        elif isinstance(node, exp.Select):
            # distinct is a boolean, store as bool – not string
            result["properties"]["distinct"] = node.args.get("distinct")

        return result
```



---

3.4 ColumnLineageExtractor

Purpose: Derives column‑level, source‑to‑target lineage from the outermost SELECT statement. For each output column it records:

· target_column – alias or expression string.
· source_columns – list of {table, column} references used in the expression.
· expression – full SQL text of the column expression.
· used_tables – unique source tables involved (note: for SELECT * this includes all table references found in the entire SELECT tree, including subqueries; this is a deliberate “best‑effort” behaviour, documented for transparency).

The include_intermediate constructor parameter is reserved for subclasses that wish to expand lineage through CTEs / subqueries. The base implementation does not change behaviour based on its value.

```python
import logging
from typing import Any, Dict, List, Optional
import sqlglot.exp as exp

logger = logging.getLogger(__name__)


class ColumnLineageExtractor:
    """Extract column‑level lineage from a parsed SQL AST."""

    def __init__(self, dialect: str = "spark", include_intermediate: bool = False):
        self.dialect = dialect
        self.include_intermediate = include_intermediate
        # Note: include_intermediate is reserved for subclass extensions.
        # The base class ignores it.

    def extract(self, tree: exp.Expression) -> List[Dict[str, Any]]:
        """Return a list of lineage records, one per output column.

        Raises:
            LineageExtractionError: If no SELECT is found or extraction fails.
        """
        try:
            select = self._find_outermost_select(tree)
            if select is None:
                raise LineageExtractionError("No SELECT statement found in AST.")

            lineage: List[Dict[str, Any]] = []

            for proj in select.expressions:
                # ── Handle SELECT * ──────────────────────────────
                if isinstance(proj, exp.Star):
                    lineage.append({
                        "target_column": "*",
                        "source_columns": [],
                        "expression": "*",
                        "used_tables": self._extract_tables_from_select(select),
                    })
                    continue

                # ── Normal / aliased column ──────────────────────
                target_name = proj.alias or proj.sql(dialect=self.dialect)
                source_cols = self._extract_column_refs(proj)

                lineage.append({
                    "target_column": target_name,
                    "source_columns": source_cols,
                    "expression": proj.sql(dialect=self.dialect),
                    "used_tables": sorted(
                        {c["table"] for c in source_cols if c.get("table")}
                    ),
                })

            logger.debug("Extracted lineage for %d output columns.", len(lineage))
            return lineage

        except LineageExtractionError:
            raise
        except Exception as exc:
            raise LineageExtractionError(
                f"Lineage extraction failed: {exc}"
            ) from exc

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _find_outermost_select(tree: exp.Expression) -> Optional[exp.Select]:
        """Return the outermost SELECT node.

        ``sqlglot.parse_one`` returns the root expression, which for a
        SELECT statement is the outermost Select. For other statements
        (INSERT, CREATE) that contain a SELECT, we find the first Select
        encountered during pre‑order traversal.
        """
        if isinstance(tree, exp.Select):
            return tree
        # Generic search: the first found during pre‑order is outermost.
        for node in tree.walk():
            if isinstance(node, exp.Select):
                return node
        return None

    def _extract_column_refs(
        self, expr: exp.Expression
    ) -> List[Dict[str, Optional[str]]]:
        refs: List[Dict[str, Optional[str]]] = []
        for node in expr.walk():
            if isinstance(node, exp.Column):
                refs.append({
                    "table": node.table or None,
                    "column": node.name,
                })
        return refs

    @staticmethod
    def _extract_tables_from_select(select: exp.Select) -> List[str]:
        """Best‑effort list of table names referenced in FROM / JOIN.

        Note: This searches the **entire** select tree, including
        tables inside subqueries. It is intended as a quick overview,
        not a strict source‑table list.
        """
        tables: set[str] = set()
        for tbl in select.find_all(exp.Table):
            if tbl.name:
                tables.add(tbl.name)
        return sorted(tables)
```

Extensibility: The base implementation handles simple SELECT, JOIN, functions, and subqueries in the projection. To support CTE deep‑tracing, window‑function partition columns, or PIVOT/UNPIVOT, subclass and override extract or _extract_column_refs. The include_intermediate flag can be used by such subclasses.

---

3.5 LLMFactory – Provider‑Agnostic Model Creation

Purpose: Returns a LangChain BaseChatModel instance based on the Config. Adding a new provider requires only an extra elif branch. All SecretStr values are unwrapped via .get_secret_value(). The single llm_temperature field is used for all providers.

```python
import logging
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


class LLMFactory:
    """Create a LangChain chat model from pipeline configuration."""

    @staticmethod
    def create(config: Config) -> BaseChatModel:
        provider = config.llm_provider.lower().strip()
        logger.info("Creating LLM instance for provider: %s", provider)

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                api_key=config.openai_api_key.get_secret_value() or None,
                model=config.openai_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
            )

        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                api_key=config.anthropic_api_key.get_secret_value() or None,
                model=config.anthropic_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
            )

        elif provider == "huggingface_endpoint":
            from langchain_huggingface import HuggingFaceEndpoint
            return HuggingFaceEndpoint(
                endpoint_url=config.hf_endpoint_url,
                huggingfacehub_api_token=(
                    config.hf_api_token.get_secret_value() or None
                ),
                task="text-generation",
                model_kwargs={
                    "max_new_tokens": config.hf_max_new_tokens,
                    "temperature": config.llm_temperature,   # unified temperature
                },
            )

        elif provider == "huggingface_local":
            from langchain_huggingface import ChatHuggingFace
            return ChatHuggingFace(
                model_name=config.hf_model_name,
                huggingfacehub_api_token=(
                    config.hf_api_token.get_secret_value() or None
                ),
                model_kwargs={
                    "max_new_tokens": config.hf_max_new_tokens,
                    "temperature": config.llm_temperature,
                },
            )

        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                base_url=config.ollama_base_url,
                model=config.ollama_model,
                temperature=config.llm_temperature,
            )

        elif provider == "mock":
            from langchain_core.language_models.fake_chat_models import (
                GenericFakeChatModel,
            )
            return GenericFakeChatModel(
                messages=iter(["Mock LLM response."])
            )

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
```



---

3.6 SQLAnalysisChain

Purpose: Builds a LangChain chain that combines a system prompt, a human template (with AST JSON, lineage, and instruction), the injected LLM, and a string output parser. Prompt files are loaded with built‑in fallback defaults. LLM invocation is wrapped with exponential‑backoff retry.

```python
import json
import logging
from pathlib import Path

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)

# ── Fallback prompts (used when files are missing) ───────────────
_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert SQL analyst. You receive a JSON representation of a "
    "SQL query's abstract syntax tree (AST) and a detailed column-level "
    "lineage showing how each output column is derived from source columns. "
    "Your task is to analyze the query according to the user's instruction."
)

_DEFAULT_HUMAN_TEMPLATE = (
    "Instruction: {instruction}\n\n"
    "SQL AST (JSON):\n{ast_json}\n\n"
    "Column-level lineage (source → target mapping):\n{column_lineage}"
)


class SQLAnalysisChain:
    """LangChain chain that sends AST + lineage to an LLM."""

    def __init__(self, config: Config, llm: BaseChatModel):
        self.config = config
        self.llm = llm
        self.chain = self._build_chain()

    # ── public API ───────────────────────────────────────────────

    def run(
        self,
        ast_json: dict,
        column_lineage: list,
        instruction: str = "",
    ) -> str:
        """Invoke the chain with retry logic.

        Raises:
            LLMCommunicationError: After all retry attempts are exhausted.
        """
        try:
            return self._invoke_with_retry(
                ast_json=ast_json,
                column_lineage=column_lineage,
                instruction=instruction,
            )
        except Exception as exc:
            raise LLMCommunicationError(
                f"Chain execution failed after retries: {exc}"
            ) from exc

    # ── internals ────────────────────────────────────────────────

    def _build_chain(self):
        system_text = self._load_prompt(
            self.config.prompt_system_file, _DEFAULT_SYSTEM_PROMPT
        )
        human_text = self._load_prompt(
            self.config.prompt_human_template_file, _DEFAULT_HUMAN_TEMPLATE
        )

        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_text),
            HumanMessagePromptTemplate.from_template(human_text),
        ])

        return chat_prompt | self.llm | StrOutputParser()

    @retry(
        stop=stop_after_attempt(3),  # could use config.llm_retry_attempts
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _invoke_with_retry(
        self,
        ast_json: dict,
        column_lineage: list,
        instruction: str,
    ) -> str:
        logger.debug("Invoking LLM chain (attempt may be retried).")
        return self.chain.invoke({
            "ast_json": json.dumps(ast_json, indent=2),
            "column_lineage": json.dumps(column_lineage, indent=2),
            "instruction": instruction,
        })

    @staticmethod
    def _load_prompt(path: str, default: str) -> str:
        """Read a prompt file; fall back to *default* if missing."""
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            logger.warning(
                "Prompt file '%s' not found; using built‑in default.", path
            )
            return default
```

Prompt template files (shipped with the package):

prompts/system.txt:

```
You are an expert SQL analyst. You receive a JSON representation of a SQL query's abstract syntax tree (AST) and a detailed column-level lineage showing how each output column is derived from source columns. Your task is to analyze the query according to the user's instruction.
```

prompts/human.txt:

```
Instruction: {instruction}

SQL AST (JSON):
{ast_json}

Column-level lineage (source → target mapping):
{column_lineage}
```

---

3.7 PipelineOrchestrator

Purpose: Coordinates the entire pipeline: parse → serialize → extract lineage → LLM invocation. Accepts an optional pre‑built LLM for runtime flexibility. All exceptions are caught and surfaced via PipelineResult.error so that batch processing never crashes on a single bad query.

```python
import logging
import time
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """End‑to‑end coordinator for SQL analysis."""

    def __init__(
        self,
        config: Config,
        llm: Optional[BaseChatModel] = None,
    ):
        self.config = config
        self.parser = SQLParser(
            dialect=config.sql_dialect,
            error_on_incomplete=config.error_on_incomplete,
        )
        self.serializer = ASTSerializer(max_depth=config.ast_max_depth)
        self.lineage_extractor = ColumnLineageExtractor(
            dialect=config.sql_dialect,
            include_intermediate=config.lineage_include_intermediate_columns,
        )
        self.llm = llm or LLMFactory.create(config)
        self.chain = SQLAnalysisChain(config, self.llm)

    def run(self, sql: str, instruction: str = "") -> PipelineResult:
        """Execute the full pipeline and always return a PipelineResult.

        On failure the ``error`` field is populated and the remaining
        fields carry sensible defaults (empty dict / list / string).
        """
        model_label = getattr(self.llm, "model_name", str(self.llm))

        try:
            # 1. Parse
            ast = self.parser.parse(sql)

            # 2. Serialize AST
            ast_json = self.serializer.serialize(ast)

            # 3. Extract column lineage
            column_lineage = self.lineage_extractor.extract(ast)

            # 4. Call LLM chain (with built‑in retry)
            start = time.perf_counter()
            llm_response = self.chain.run(ast_json, column_lineage, instruction)
            latency = time.perf_counter() - start

            logger.info(
                "Pipeline completed in %.3fs for query: %.60s…",
                latency, sql.replace("\n", " "),
            )

            return PipelineResult(
                original_sql=sql,
                ast_json=ast_json,
                column_lineage=column_lineage,
                llm_response=llm_response,
                latency_seconds=round(latency, 4),
                model_used=model_label,
            )

        except (
            ParsingError,
            SerializationError,
            LineageExtractionError,
            LLMCommunicationError,
            InvalidResponseError,
        ) as exc:
            logger.error("Pipeline error: %s", exc)
            return PipelineResult(
                original_sql=sql,
                ast_json={},
                column_lineage=[],
                llm_response="",
                latency_seconds=0.0,
                model_used=model_label,
                error=str(exc),
            )
        except Exception as exc:  # catch‑all safety net
            logger.exception("Unexpected pipeline failure")
            return PipelineResult(
                original_sql=sql,
                ast_json={},
                column_lineage=[],
                llm_response="",
                latency_seconds=0.0,
                model_used=model_label,
                error=f"Unexpected error: {exc}",
            )
```

---

3.8 PipelineResult (Data Class)

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PipelineResult:
    """Immutable record of a single pipeline execution."""

    original_sql: str
    ast_json: Dict[str, Any] = field(default_factory=dict)
    column_lineage: List[Dict[str, Any]] = field(default_factory=list)
    llm_response: str = ""
    latency_seconds: float = 0.0
    model_used: str = ""
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Convenience flag: True when no error was recorded."""
        return self.error is None
```

---

3.9 Custom Exceptions

```python
class PipelineBaseError(Exception):
    """Base class for all pipeline exceptions."""


class ParsingError(PipelineBaseError):
    """Raised when sqlglot cannot parse the input SQL."""


class SerializationError(PipelineBaseError):
    """Raised when AST serialization encounters an unexpected structure."""


class LineageExtractionError(PipelineBaseError):
    """Raised when column lineage cannot be derived."""


class LLMCommunicationError(PipelineBaseError):
    """Raised when the LLM API or chain invocation fails."""


class InvalidResponseError(PipelineBaseError):
    """Raised when the LLM response is malformed or cannot be validated."""
```

---

3.10 Logging Utility

```python
import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a structured console handler."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )
    # Quiet down noisy third‑party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
```

---

1. Flexible Model Usage – Explicit Demands

The system must meet the following flexibility requirements:

# Requirement Detail

1 Provider abstraction No provider‑specific code in pipeline logic. All model creation is centralised in LLMFactory.
2 Configuration‑driven Switching providers is done by changing environment variables (e.g., LLM_PROVIDER=ollama) – no source‑code modifications.
3 Runtime injection PipelineOrchestrator.**init** accepts an optional BaseChatModel. Users can supply any LangChain‑compatible model directly.
4 Mock support A mock provider is built in for unit tests and offline demonstrations.
5 Easy provider addition Adding a provider requires only a new elif in LLMFactory and optional config fields.
6 Prompt isolation Prompts are loaded from external files with built‑in fallback defaults.
7 Lineage is first‑class Lineage is always computed, passed to the LLM, and returned in PipelineResult.
8 Secret safety API keys use SecretStr; .get_secret_value() is called only inside LLMFactory.
9 Retry resilience LLM calls are retried with exponential backoff (configurable attempts / wait).
10 Graceful degradation Pipeline errors are captured in PipelineResult.error; the process never crashes on a single bad query.

---

1. Lineage Extraction Details

The ColumnLineageExtractor provides column‑oriented source‑to‑target mapping.

Supported Features

Feature Example Behaviour
Direct columns SELECT a, b FROM t a → t.a, b → t.b
Aliased expressions SELECT x AS y target y, source columns from x
Qualified columns SELECT t.a table t preserved
Functions / arithmetic UPPER(name), a + b all base columns listed in source_columns
JOINs SELECT u.name, o.total FROM … columns separated by source table
Subqueries in projection SELECT (SELECT MAX(v) …) AS m inner columns traced (if include_intermediate is overridden in a subclass)
Wildcards SELECT * single entry: target_column: "*", source_columns: [], used_tables populated from entire SELECT tree (including subqueries)

Limitations (by default)

· CTEs and nested subqueries in FROM are not fully expanded in the base implementation.
· Window functions, PIVOT/UNPIVOT may require custom handling.
· Wildcard expansion to individual columns requires external catalog metadata (not included).
· The used_tables list for wildcards includes all table references inside the SELECT tree; it is not limited to immediate FROM/JOIN tables.

Extension Points

Subclass ColumnLineageExtractor and override extract or _extract_column_refs to add CTE resolution, schema‑aware wildcard expansion, or window‑function partition tracing. The include_intermediate flag is available for subclass logic.

---

1. Usage Examples

6.1 Basic Run (OpenAI)

```python
from sql_pipeline.models.config import Config
from sql_pipeline.core.orchestrator import PipelineOrchestrator
from sql_pipeline.utils import setup_logging

setup_logging("INFO")

config = Config()  # reads .env: LLM_PROVIDER=openai, OPENAI_API_KEY=sk-…
orchestrator = PipelineOrchestrator(config)

result = orchestrator.run(
    sql=(
        "SELECT u.name, SUM(o.amount) AS total "
        "FROM users u JOIN orders o ON u.id = o.user_id "
        "GROUP BY u.name"
    ),
    instruction="Explain the query in simple terms.",
)

if result.success:
    print("Lineage:", result.column_lineage)
    print("LLM Response:", result.llm_response)
else:
    print("Pipeline error:", result.error)
```

6.2 Using Ollama Locally

```bash
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.2
python -m sql_pipeline.main --sql "SELECT 1 AS one" --instruction "What does this do?"
```

6.3 Injecting a Custom Model

```python
from langchain_openai import ChatOpenAI
from sql_pipeline.models.config import Config
from sql_pipeline.core.orchestrator import PipelineOrchestrator

custom_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
orchestrator = PipelineOrchestrator(Config(), llm=custom_llm)
result = orchestrator.run("SELECT id, email FROM customers WHERE active = 1")
```

6.4 Batch Processing (Graceful Degradation)

```python
queries = ["SELECT a FROM t", "THIS IS NOT SQL", "SELECT b FROM u"]

for sql in queries:
    result = orchestrator.run(sql, instruction="Summarise.")
    if result.success:
        print(f"✅ {sql[:40]}… → {result.llm_response[:60]}")
    else:
        print(f"❌ {sql[:40]}… → {result.error}")
# The pipeline never crashes; bad queries are reported via result.error.
```

---

1. Testing Specifications

Unit Tests

Component Test Cases
SQLParser Valid SQL (simple, JOIN, CTE, subquery); invalid SQL; empty string; dialect variations
ASTSerializer Mock sqlglot nodes; deeply nested query exceeding max_depth; preservation of boolean types; list‑valued args
ColumnLineageExtractor Simple SELECT; JOIN; aliased expression; SELECT *; aggregate functions; subquery in projection; no SELECT (e.g., INSERT)
LLMFactory Correct model class for each provider string; unknown provider raises ValueError; SecretStr unwrapping
SQLAnalysisChain Mock LLM; verify prompt placeholders are filled; missing prompt file falls back to default
PipelineResult success property is True when error is None, False otherwise

Integration Tests

· Full pipeline with mock provider: assert PipelineResult structure, lineage presence, and success == True.
· Full pipeline with intentionally broken SQL: assert success == False and error is populated.

Contract Tests

· Validate the JSON schema of column_lineage entries (keys: target_column, source_columns, expression, used_tables).
· Validate ast_json structure (keys: type, properties, children).

Test File Layout

```
tests/
├── conftest.py            # shared fixtures (mock config, sample SQL)
├── test_parser.py
├── test_serializer.py
├── test_lineage.py
├── test_llm_factory.py
├── test_chain.py
├── test_orchestrator.py
└── test_result.py
```

---

1. Package Structure (AI Agent Deliverable)

```
sql_pipeline/
├── core/
│   ├── __init__.py
│   ├── parser.py            # SQLParser
│   ├── serializer.py        # ASTSerializer
│   ├── lineage.py           # ColumnLineageExtractor
│   ├── llm_factory.py       # LLMFactory
│   ├── chain.py             # SQLAnalysisChain
│   └── orchestrator.py      # PipelineOrchestrator
├── models/
│   ├── __init__.py
│   ├── config.py            # Config
│   └── result.py            # PipelineResult
├── exceptions.py            # all custom exceptions
├── utils.py                 # setup_logging()
├── prompts/
│   ├── system.txt
│   └── human.txt
├── main.py                  # CLI entry point (argparse)
├── requirements.txt
├── pyproject.toml           # optional: modern packaging
├── .env.example             # template for environment variables
└── README.md
```

requirements.txt

```text
sqlglot>=23.0.0
langchain>=0.3.0
langchain-core>=0.3.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
tenacity>=8.2.0

# ── Provider packages (install as needed) ──
# langchain-openai>=0.2.0
# langchain-anthropic>=0.2.0
# langchain-huggingface>=0.1.0
# langchain-ollama>=0.2.0
```

.env.example

```dotenv
# LLM provider: openai | anthropic | huggingface_endpoint | huggingface_local | ollama | mock
LLM_PROVIDER=openai

# OpenAI
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o

# Anthropic
# ANTHROPIC_API_KEY=sk-ant-…
# ANTHROPIC_MODEL=claude-3-haiku-20240307

# Hugging Face Endpoint
# HF_ENDPOINT_URL=https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3
# HF_API_TOKEN=hf_your_token
# HF_MAX_NEW_TOKENS=512

# Ollama
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama3.2

# Common LLM settings
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=1024

# SQL
SQL_DIALECT=spark
LOG_LEVEL=INFO
```

---

1. AI Agent Implementation Note

When the AI agent builds this Target System, it must:

· Generate all classes with full type hints, docstrings, and proper error handling.
· Use Pydantic V2 patterns: SettingsConfigDict, SecretStr for all API keys.
· Call .get_secret_value() on SecretStr fields only inside LLMFactory.
· Use iterative (stack‑based) traversal in ASTSerializer with a configurable max_depth.
· Detect the outermost SELECT correctly: return the root if it’s a Select, otherwise find the first Select during pre‑order walk.
· In ASTSerializer, preserve original types of primitive properties (booleans, numbers, strings) and prevent overwriting pre‑defined convenience keys during the generic loop.
· Handle exp.Star (SELECT *) explicitly in ColumnLineageExtractor.
· Clearly document that include_intermediate in ColumnLineageExtractor does nothing in the base class; it is reserved for subclass use.
· Provide fallback default prompts in SQLAnalysisChain._load_prompt.
· Wrap LLM invocation with tenacity.retry (exponential backoff, configurable attempts).
· Catch all pipeline exceptions in PipelineOrchestrator.run() and return a PipelineResult with error populated (graceful degradation).
· Add a success property to PipelineResult.
· Include a setup_logging() utility called from main.py.
· Create the package structure exactly as shown in Section 8.
· Include prompt template files with placeholders {ast_json}, {column_lineage}, {instruction}.
· Provide a main.py that uses argparse with --sql, --instruction, --provider, and --dialect flags.
· Ship a .env.example file with all relevant settings (including HF_ENDPOINT_URL).
· Write a comprehensive README.md covering setup, environment variables, usage examples, extension guidelines, and lineage feature explanation.
· Generate a test suite covering all components (see Section 7).

---

1. Changelog

Version Date Changes
1.0 – Initial specification
2.0 2026‑07‑28 Pydantic V2 migration (SettingsConfigDict, SecretStr); iterative AST serialization with max_depth; outermost SELECT detection; SELECT * handling; prompt fallback defaults; tenacity retry on LLM calls; graceful degradation in orchestrator; PipelineResult.success property; PipelineBaseError exception hierarchy; setup_logging utility; .env.example; batch‑processing example; expanded test matrix; time.perf_counter for latency
2.1 2026‑07‑28 Fixed outermost SELECT detection (root Select check); fixed ASTSerializer to preserve primitive types and prevent property overwriting; merged hf_temperature into llm_temperature; clarified include_intermediate as subclass reserved; documented wildcard used_tables behaviour; added missing HF endpoint example in .env.example; minor code improvements and comments

---

This specification ensures the resulting system is modular, flexible, lineage‑aware, resilient, and ready for real‑world production deployment.