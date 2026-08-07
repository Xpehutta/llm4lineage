"""Airflow DAG module must import without Airflow installed."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class TestLineageDailyDag(unittest.TestCase):
    def test_module_imports(self) -> None:
        import dags.lineage_daily as mod

        self.assertTrue(callable(mod.extract_catalog))
        self.assertTrue(callable(mod.parse_and_build))
        self.assertTrue(callable(mod.publish_openlineage))

    def test_parse_build_and_publish(self) -> None:
        import dags.lineage_daily as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sql_path = root / "query.sql"
            sql_path.write_text(
                "INSERT INTO analytics.sales SELECT amount AS total FROM orders",
                encoding="utf-8",
            )
            out_dir = root / "out"
            built = mod.parse_and_build(sql_path=sql_path, out_dir=out_dir, parse_plpgsql=False)
            self.assertIsNone(built.get("error"))
            self.assertTrue(Path(built["graph_path"]).exists())

            published = mod.publish_openlineage(
                graph_path=built["graph_path"],
                sql_path=sql_path,
                out_dir=out_dir,
            )
            events = json.loads(Path(published["events_path"]).read_text(encoding="utf-8"))
            self.assertEqual([e["eventType"] for e in events], ["START", "COMPLETE"])
            complete = events[1]
            output_names = [ds["name"] for ds in complete["outputs"]]
            self.assertIn("analytics.sales", output_names)

    def test_repository_changed_when_state_missing(self) -> None:
        import dags.lineage_daily as mod

        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(mod.repository_changed(Path(tmp) / "missing.json"))


if __name__ == "__main__":
    unittest.main()
