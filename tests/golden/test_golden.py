"""Golden graph regression tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from Classes.sql2graph_classes import SQL2GraphParser, SQL2GraphPipeline
from Classes.validation_classes import SQLLineageValidator

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).resolve().parent / "ddls10_first_graph.json"


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


if __name__ == "__main__":
    unittest.main()
