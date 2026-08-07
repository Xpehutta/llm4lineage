"""Golden graph regression tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from Classes.sql2graph_classes import SQL2GraphParser, SQL2GraphPipeline
from Classes.validation_classes import SQLLineageValidator
from tests.golden.update_golden import build_payload, diff_against_golden

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).resolve().parent / "ddls10_first_graph.json"
SQL_SOURCE = ROOT / "data" / "DDLs_10.txt"


class TestGoldenGraph(unittest.TestCase):
    def test_ddls10_first_graph_edge_f1(self):
        if not GOLDEN.exists():
            self.skipTest("golden file missing; run tests/golden/update_golden.py")

        sql = ROOT.joinpath("data/DDLs_10.txt").read_text(encoding="utf-8").split(";")[0].strip()
        pipeline = SQL2GraphPipeline(parser=SQL2GraphParser(dialect="postgres"))
        result = pipeline.run(sql, dialect="postgres", use_llm_verify=False, use_llm_enhance=False)
        self.assertNotIn("error", result)

        expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
        metrics = SQLLineageValidator.calculate_edge_f1(expected, result["graph"])
        self.assertGreaterEqual(metrics["f1"], 0.9, metrics)


class TestGoldenDrift(unittest.TestCase):
    """The fixture must match a fresh run exactly, not just score well against it.

    The F1 check above tolerates a 10% deviation, which lets small regressions
    accumulate unnoticed. This is the dry-run of ``update_golden.py``: any
    change to the generated graph has to be reviewed and committed deliberately.
    """

    def test_golden_fixture_has_not_drifted(self):
        if not GOLDEN.exists() or not SQL_SOURCE.exists():
            self.skipTest("golden fixture or source SQL missing")

        differences = diff_against_golden(SQL_SOURCE, GOLDEN)
        self.assertEqual(
            differences,
            [],
            "Golden graph drifted. Review the changes, then refresh with:\n"
            "  python tests/golden/update_golden.py",
        )

    def test_generation_is_deterministic(self):
        if not SQL_SOURCE.exists():
            self.skipTest("source SQL missing")
        self.assertEqual(build_payload(SQL_SOURCE), build_payload(SQL_SOURCE))

    def test_generation_survives_hash_randomisation(self):
        """Graph ids must not depend on PYTHONHASHSEED.

        sqlglot gathers a column's sources through a set of expressions whose
        hash is salted per interpreter, so anything we derive from their order
        silently changes between runs. Repeating the run in-process cannot catch
        that - the seed is fixed once at startup - hence the subprocesses.
        """
        if not SQL_SOURCE.exists():
            self.skipTest("source SQL missing")

        script = (
            "import json;"
            "from tests.golden.update_golden import build_payload;"
            f"print(json.dumps(build_payload({str(SQL_SOURCE)!r}), sort_keys=True))"
        )
        payloads = []
        for seed in ("0", "2", "13"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            completed = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                cwd=ROOT,
                env=env,
                check=True,
            )
            payloads.append((seed, completed.stdout.strip()))

        baseline_seed, baseline = payloads[0]
        for seed, payload in payloads[1:]:
            self.assertEqual(
                payload,
                baseline,
                f"graph differs between PYTHONHASHSEED={baseline_seed} and {seed}",
            )


if __name__ == "__main__":
    unittest.main()
