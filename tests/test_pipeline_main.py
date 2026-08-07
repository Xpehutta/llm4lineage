"""Tests for the ``sql-pipeline`` console script (Classes/pipeline/main.py)."""

from __future__ import annotations

import io
import json
import logging
import unittest
from contextlib import redirect_stderr, redirect_stdout

from Classes.pipeline.main import build_parser, main
from Classes.pipeline.utils import setup_logging


def run_cli(*args) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(list(args))
    return code, out.getvalue(), err.getvalue()


class TestBuildParser(unittest.TestCase):
    def test_defaults(self):
        args = build_parser().parse_args(["--sql", "SELECT 1"])
        self.assertEqual(args.sql, "SELECT 1")
        self.assertEqual(args.instruction, "Explain the query in simple terms.")
        self.assertIsNone(args.provider)
        self.assertIsNone(args.dialect)
        self.assertFalse(args.json)

    def test_sql_is_required(self):
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                build_parser().parse_args([])

    def test_flags_are_parsed(self):
        args = build_parser().parse_args(
            ["--sql", "SELECT 1", "--provider", "mock", "--dialect", "spark", "--json"]
        )
        self.assertEqual(args.provider, "mock")
        self.assertEqual(args.dialect, "spark")
        self.assertTrue(args.json)


class TestMain(unittest.TestCase):
    """The CLI is exercised end-to-end against the deterministic mock provider."""

    def test_human_readable_output_contains_lineage_and_llm_response(self):
        code, stdout, _ = run_cli("--sql", "SELECT a FROM t", "--provider", "mock")

        self.assertEqual(code, 0)
        self.assertIn("Column lineage:", stdout)
        self.assertIn('"target_column": "a"', stdout)
        self.assertIn("LLM response:", stdout)

    def test_json_output_is_a_full_pipeline_result(self):
        code, stdout, _ = run_cli("--sql", "SELECT a FROM t", "--provider", "mock", "--json")

        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["original_sql"], "SELECT a FROM t")
        self.assertEqual(payload["model_used"], "mock")
        self.assertIsNone(payload["error"])
        self.assertEqual(payload["column_lineage"][0]["target_column"], "a")
        self.assertIn("ast_json", payload)

    def test_dialect_override_is_applied(self):
        code, stdout, _ = run_cli(
            "--sql", "SELECT a FROM t",
            "--provider", "mock",
            "--dialect", "spark",
            "--json",
        )
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(stdout)["success"])

    def test_failure_is_reported_on_stderr_with_exit_code_one(self):
        code, stdout, stderr = run_cli("--sql", "THIS IS NOT SQL", "--provider", "mock")

        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("Pipeline error:", stderr)

    def test_json_output_of_a_failed_run_still_exits_non_zero(self):
        code, stdout, _ = run_cli("--sql", "THIS IS NOT SQL", "--provider", "mock", "--json")

        self.assertEqual(code, 1)
        payload = json.loads(stdout)
        self.assertFalse(payload["success"])
        self.assertTrue(payload["error"])


class TestSetupLogging(unittest.TestCase):
    def setUp(self):
        root = logging.getLogger()
        self._level = root.level
        self._handlers = list(root.handlers)
        self._quiet_levels = {
            name: logging.getLogger(name).level for name in ("httpx", "langchain")
        }
        self.addCleanup(self._restore)

    def _restore(self):
        root = logging.getLogger()
        root.setLevel(self._level)
        root.handlers = self._handlers
        for name, level in self._quiet_levels.items():
            logging.getLogger(name).setLevel(level)

    def test_named_level_is_applied_to_the_root_logger(self):
        logging.getLogger().handlers = []
        setup_logging("DEBUG")
        self.assertEqual(logging.getLogger().level, logging.DEBUG)

    def test_level_name_is_case_insensitive(self):
        logging.getLogger().handlers = []
        setup_logging("warning")
        self.assertEqual(logging.getLogger().level, logging.WARNING)

    def test_unknown_level_falls_back_to_info(self):
        logging.getLogger().handlers = []
        setup_logging("NOT_A_LEVEL")
        self.assertEqual(logging.getLogger().level, logging.INFO)

    def test_noisy_third_party_loggers_are_quietened(self):
        setup_logging("DEBUG")
        self.assertEqual(logging.getLogger("httpx").level, logging.WARNING)
        self.assertEqual(logging.getLogger("langchain").level, logging.WARNING)


if __name__ == "__main__":
    unittest.main()
