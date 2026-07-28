"""CLI entry point for the SQL analysis pipeline."""

from __future__ import annotations

import argparse
import json
import sys

from Classes.pipeline.core.orchestrator import PipelineOrchestrator
from Classes.pipeline.models.config import Config
from Classes.pipeline.utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse SQL, extract column lineage, and analyse via LLM.",
    )
    parser.add_argument("--sql", required=True, help="SQL query to analyse")
    parser.add_argument(
        "--instruction",
        default="Explain the query in simple terms.",
        help="Instruction passed to the LLM",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Override LLM_PROVIDER (openai, anthropic, ollama, mock, …)",
    )
    parser.add_argument(
        "--dialect",
        default=None,
        help="Override SQL_DIALECT (default: spark)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full PipelineResult as JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    overrides: dict = {}
    if args.provider:
        overrides["llm_provider"] = args.provider
    if args.dialect:
        overrides["sql_dialect"] = args.dialect

    config = Config(**overrides)
    setup_logging(config.log_level)

    orchestrator = PipelineOrchestrator(config)
    result = orchestrator.run(args.sql, instruction=args.instruction)

    if args.json:
        payload = {
            "original_sql": result.original_sql,
            "ast_json": result.ast_json,
            "column_lineage": result.column_lineage,
            "llm_response": result.llm_response,
            "latency_seconds": result.latency_seconds,
            "model_used": result.model_used,
            "error": result.error,
            "success": result.success,
        }
        print(json.dumps(payload, indent=2))
    elif result.success:
        print("Column lineage:")
        print(json.dumps(result.column_lineage, indent=2))
        print("\nLLM response:")
        print(result.llm_response)
    else:
        print(f"Pipeline error: {result.error}", file=sys.stderr)
        return 1

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
