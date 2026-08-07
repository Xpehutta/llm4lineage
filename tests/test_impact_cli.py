"""Tests for the ``llm4lineage-impact`` console script (Classes/impact.py)."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from Classes.impact import main

SQL = "SELECT o.amount AS total FROM orders o"


class ImpactCliTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.sql_path = Path(self._tmpdir.name) / "query.sql"
        self.sql_path.write_text(SQL, encoding="utf-8")

    def run_cli(self, *args) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["--sql", str(self.sql_path), *args])
        return code, out.getvalue(), err.getvalue()


class TestImpactCli(ImpactCliTestCase):
    def test_upstream_report_lists_the_source_column(self):
        code, stdout, _ = self.run_cli("--target", "output.total", "--direction", "up")

        self.assertEqual(code, 0)
        report = json.loads(stdout)
        self.assertEqual(report["target"], "output.total")
        self.assertEqual([hit["node"] for hit in report["upstream"]], ["orders.amount"])
        self.assertEqual(report["downstream"], [])

    def test_both_directions_is_the_default(self):
        code, stdout, _ = self.run_cli("--target", "output.total")

        self.assertEqual(code, 0)
        report = json.loads(stdout)
        self.assertIn("orders.amount", [hit["node"] for hit in report["upstream"]])
        self.assertEqual(report["downstream"], [])

    def test_qualified_target_is_rewritten_to_the_output_namespace(self):
        """``--target o.total`` is a convenience spelling of ``output.total``."""
        code, stdout, _ = self.run_cli("--target", "o.total", "--direction", "up")

        self.assertEqual(code, 0)
        report = json.loads(stdout)
        self.assertEqual(report["target"], "output.total")
        self.assertEqual([hit["node"] for hit in report["upstream"]], ["orders.amount"])

    def test_unqualified_target_is_used_verbatim_and_finds_nothing(self):
        code, stdout, _ = self.run_cli("--target", "total")

        self.assertEqual(code, 0)
        report = json.loads(stdout)
        self.assertEqual(report, {"target": "total", "upstream": [], "downstream": []})

    def test_unparseable_sql_reports_the_error_and_exits_non_zero(self):
        self.sql_path.write_text("SELECT FROM FROM", encoding="utf-8")

        code, stdout, stderr = self.run_cli("--target", "output.total")

        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("Failed to parse SQL", stderr)

    def test_target_is_required(self):
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                main(["--sql", str(self.sql_path)])


if __name__ == "__main__":
    unittest.main()
