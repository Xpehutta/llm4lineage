"""Prompt chain that sends AST + lineage to an LLM."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from Classes.pipeline.core.llm_interface import (
    ChatMessage,
    LLMInterface,
    adapt_llm,
    render_template,
)
from Classes.pipeline.exceptions import LLMCommunicationError
from Classes.pipeline.models.config import Config

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_PROMPT_DIR = _PACKAGE_DIR / "prompts"

_DEFAULT_SYSTEM_PROMPT = (
    "You verify deterministic sqlglot column lineage against a SQL query AST. "
    "Check that each target column's source_columns are complete and correct. "
    "Report missing dependencies, filters, or join keys. "
    "Suggest enhancements only where the deterministic parser missed semantics."
)

_DEFAULT_HUMAN_TEMPLATE = (
    "Instruction: {instruction}\n\n"
    "SQL AST (JSON):\n{ast_json}\n\n"
    "Deterministic column lineage (sqlglot):\n{column_lineage}"
)


class _PromptChain:
    """Render the system/human prompts and forward them to the model.

    Mirrors the ``prompt | llm | StrOutputParser()`` runnable it replaces,
    including the ``invoke(payload) -> str`` shape.
    """

    def __init__(self, system_text: str, human_text: str, llm: LLMInterface):
        self.system_text = system_text
        self.human_text = human_text
        self.llm = llm

    def invoke(self, payload: Dict[str, Any]) -> str:
        messages = [
            ChatMessage("system", render_template(self.system_text, payload)),
            ChatMessage("user", render_template(self.human_text, payload)),
        ]
        return self.llm.invoke_messages(messages)


class SQLAnalysisChain:
    """Prompt chain that sends AST + lineage to an LLM."""

    def __init__(self, config: Config, llm: Any):
        self.config = config
        self.llm = adapt_llm(llm)
        self.chain = self._build_chain()

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

    def _build_chain(self):
        system_path = self._resolve_prompt_path(
            self.config.prompt_system_file,
            _DEFAULT_PROMPT_DIR / "system.txt",
        )
        human_path = self._resolve_prompt_path(
            self.config.prompt_human_template_file,
            _DEFAULT_PROMPT_DIR / "human.txt",
        )

        system_text = self._load_prompt(system_path, _DEFAULT_SYSTEM_PROMPT)
        human_text = self._load_prompt(human_path, _DEFAULT_HUMAN_TEMPLATE)

        return _PromptChain(system_text, human_text, self.llm)

    def _invoke_with_retry(
        self,
        ast_json: dict,
        column_lineage: list,
        instruction: str,
    ) -> str:
        logger.debug("Invoking LLM chain (attempt may be retried).")
        payload = {
            "ast_json": json.dumps(ast_json, indent=2),
            "column_lineage": json.dumps(column_lineage, indent=2),
            "instruction": instruction,
        }

        for attempt in Retrying(
            stop=stop_after_attempt(self.config.llm_retry_attempts),
            wait=wait_exponential(
                multiplier=1,
                min=self.config.llm_retry_min_wait,
                max=self.config.llm_retry_max_wait,
            ),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                return self.chain.invoke(payload)

        raise RuntimeError("Retry loop exited without returning a response.")

    @staticmethod
    def _resolve_prompt_path(configured: str, default: Path) -> Path:
        if configured:
            return Path(configured)
        return default

    @staticmethod
    def _load_prompt(path: Path, default: str) -> str:
        """Read a prompt file; fall back to *default* if missing."""
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            logger.warning(
                "Prompt file '%s' not found; using built-in default.", path
            )
            return default
