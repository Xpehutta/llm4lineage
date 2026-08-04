"""Tests for OpenLineage export."""

from __future__ import annotations

import unittest

from Classes.openlineage_exporter import to_openlineage_job_event, to_openlineage_run_event
from Classes.sql2graph_classes import SQL2GraphBuilder


class TestOpenLineageExporter(unittest.TestCase):
    def test_run_and_job_events_have_required_fields(self):
        extraction = {
            "ctes": [],
            "output_columns": [
                {
                    "alias": "total",
                    "expression": "SUM(amount)",
                    "dependencies": [{"table_alias": "orders", "column": "amount"}],
                    "aggregate": True,
                    "window_function": False,
                }
            ],
            "filters": [],
            "joins": [],
            "group_by_columns": [],
        }
        builder = SQL2GraphBuilder()
        builder.build(extraction)
        graph_json = builder.to_node_link()
        sql = "SELECT SUM(amount) AS total FROM orders"

        run_event = to_openlineage_run_event(graph_json, sql)
        self.assertEqual(run_event["eventType"], "START")
        self.assertIn("job", run_event)
        self.assertIn("inputs", run_event)
        self.assertIn("outputs", run_event)

        job_event = to_openlineage_job_event(graph_json, sql)
        self.assertEqual(job_event["eventType"], "COMPLETE")


if __name__ == "__main__":
    unittest.main()
