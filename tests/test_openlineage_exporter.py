"""Tests for OpenLineage export."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from Classes.openlineage_exporter import (
    _emit,
    build_run_event,
    main,
    run_lifecycle,
    sql_hash,
    to_openlineage_job_event,
    to_openlineage_run_event,
)
from Classes.sql2graph_classes import SQL2GraphBuilder

CLI_SQL = "SELECT o.amount AS total FROM orders o"


def _payload(nodes: list[dict], links: list[dict]) -> dict:
    return {
        "directed": True,
        "multigraph": True,
        "graph": {},
        "nodes": nodes,
        "links": [{"key": index, **link} for index, link in enumerate(links)],
    }


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


class TestEventContents(unittest.TestCase):
    """The event body must name the real datasets and per-field inputs."""

    GRAPH = _payload(
        nodes=[
            {"id": "orders.amount", "node_type": "source_column",
             "physical_table": "Raw.Orders", "column": "amount"},
            {"id": "output.total", "node_type": "output_column", "alias": "total"},
            {"id": "output.grand", "node_type": "output_column", "alias": "grand"},
        ],
        links=[
            {"source": "orders.amount", "target": "output.total", "edge_type": "DERIVED_FROM"},
            {"source": "output.total", "target": "output.grand", "edge_type": "DERIVED_FROM"},
        ],
    )

    def test_inputs_are_lowercased_physical_tables(self):
        event = to_openlineage_run_event(self.GRAPH, "SELECT 1")
        self.assertEqual(event["inputs"][0]["namespace"], "greenplum")
        self.assertEqual(event["inputs"][0]["name"], "raw.orders")

    def test_outputs_use_the_output_alias(self):
        event = to_openlineage_run_event(self.GRAPH, "SELECT 1")
        self.assertEqual(
            sorted(dataset["name"] for dataset in event["outputs"]),
            ["output.grand", "output.total"],
        )

    def test_column_lineage_facet_lists_only_source_column_inputs(self):
        """``output.grand`` derives from another output column, so it has no inputs."""
        event = to_openlineage_run_event(self.GRAPH, "SELECT 1")

        fields = event["inputs"][0]["facets"]["columnLineage"]["fields"]
        self.assertEqual(
            fields["total"]["inputFields"],
            [{"namespace": "greenplum", "name": "Raw.Orders", "field": "amount"}],
        )
        self.assertEqual(fields["grand"]["inputFields"], [])

    def test_job_name_is_the_sql_hash_and_the_query_is_carried_in_a_facet(self):
        sql = "SELECT 1"
        event = to_openlineage_run_event(self.GRAPH, sql, job_namespace="custom")

        self.assertEqual(event["job"], {"namespace": "custom", "name": sql_hash(sql)})
        self.assertEqual(event["run"]["facets"]["sql"]["query"], sql)

    def test_namespace_override_is_applied_everywhere(self):
        event = to_openlineage_run_event(self.GRAPH, "SELECT 1", namespace="warehouse")

        self.assertEqual(event["inputs"][0]["namespace"], "warehouse")
        self.assertEqual(event["outputs"][0]["namespace"], "warehouse")

    def test_sql_hash_is_stable_and_content_dependent(self):
        self.assertEqual(sql_hash("SELECT 1"), sql_hash("SELECT 1"))
        self.assertNotEqual(sql_hash("SELECT 1"), sql_hash("SELECT 2"))


class TestEmit(unittest.TestCase):
    def test_posts_json_to_the_configured_endpoint(self):
        response = MagicMock()
        response.__enter__.return_value = response
        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            _emit("http://collector.local/api/v1/lineage", {"eventType": "START"})

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://collector.local/api/v1/lineage")
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(json.loads(request.data), {"eventType": "START"})
        self.assertEqual(urlopen.call_args.kwargs, {"timeout": 30})
        response.read.assert_called_once()


class TestOpenLineageCli(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.tmp = Path(self._tmpdir.name)
        self.sql_path = self.tmp / "query.sql"
        self.sql_path.write_text(CLI_SQL, encoding="utf-8")

    def run_cli(self, *args) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["--sql", str(self.sql_path), *args])
        return code, out.getvalue(), err.getvalue()

    def test_default_run_event_is_printed_to_stdout(self):
        code, stdout, _ = self.run_cli()

        self.assertEqual(code, 0)
        event = json.loads(stdout)
        self.assertEqual(event["eventType"], "START")
        self.assertEqual(event["job"]["name"], sql_hash(CLI_SQL))
        self.assertEqual([item["name"] for item in event["inputs"]], ["orders"])

    def test_job_format_emits_a_complete_event(self):
        code, stdout, _ = self.run_cli("--format", "job")

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout)["eventType"], "COMPLETE")

    def test_out_writes_the_event_to_a_file_instead_of_stdout(self):
        destination = self.tmp / "event.json"

        code, stdout, _ = self.run_cli("--out", str(destination))

        self.assertEqual(code, 0)
        self.assertEqual(stdout, "")
        self.assertEqual(json.loads(destination.read_text(encoding="utf-8"))["eventType"], "START")

    def test_emit_posts_the_same_payload_that_is_printed(self):
        with patch("Classes.openlineage_exporter._emit") as emit:
            code, stdout, _ = self.run_cli("--emit", "http://collector.local")

        self.assertEqual(code, 0)
        emit.assert_called_once()
        url, payload = emit.call_args.args
        self.assertEqual(url, "http://collector.local")
        self.assertEqual(payload, json.loads(stdout))

    def test_unparseable_sql_exits_non_zero(self):
        self.sql_path.write_text("SELECT FROM FROM", encoding="utf-8")

        code, stdout, stderr = self.run_cli()

        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("Failed to parse SQL", stderr)


class TestOpenLineageLifecycle(unittest.TestCase):
    GRAPH = _payload(
        nodes=[
            {
                "id": "orders.amount",
                "node_type": "source_column",
                "physical_table": "orders",
                "column": "amount",
            },
            {"id": "output.total", "node_type": "output_column", "alias": "total"},
        ],
        links=[
            {
                "source": "orders.amount",
                "target": "output.total",
                "edge_type": "DERIVED_FROM",
            }
        ],
    )

    def test_lifecycle_emits_start_then_complete(self):
        sql = "INSERT INTO analytics.sales SELECT amount AS total FROM orders"
        events = run_lifecycle(self.GRAPH, sql, success=True)
        self.assertEqual([e["eventType"] for e in events], ["START", "COMPLETE"])
        self.assertEqual(events[0]["run"]["runId"], events[1]["run"]["runId"])
        self.assertEqual(
            [ds["name"] for ds in events[1]["outputs"]],
            ["analytics.sales"],
        )
        self.assertEqual(
            [ds["name"] for ds in events[1]["inputs"]],
            ["orders"],
        )

    def test_fail_event_carries_error_message(self):
        events = run_lifecycle(
            self.GRAPH,
            "SELECT 1",
            success=False,
            error_message="boom",
        )
        self.assertEqual(events[1]["eventType"], "FAIL")
        self.assertEqual(events[1]["run"]["facets"]["errorMessage"]["message"], "boom")

    def test_build_run_event_accepts_namespace_override(self):
        event = build_run_event(
            self.GRAPH,
            "INSERT INTO t SELECT 1 AS x FROM s",
            event_type="COMPLETE",
            namespace="warehouse",
        )
        self.assertEqual(event["outputs"][0]["namespace"], "warehouse")
        self.assertEqual(event["outputs"][0]["name"], "t")


if __name__ == "__main__":
    unittest.main()
