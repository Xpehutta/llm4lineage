"""Export SQL2Graph lineage to OpenLineage design-time / run lifecycle events."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from Classes.impact_analyzer import graph_from_payload
from Classes.sql2graph_classes import SQL2GraphParser, SQL2GraphPipeline
from Classes.table_lineage import extract_table_lineage

EventType = Literal["START", "COMPLETE", "FAIL", "RUNNING", "ABORT"]


def sql_hash(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def _dataset(namespace: str, name: str) -> dict[str, Any]:
    return {"namespace": namespace, "name": name}


def _field_lineage(graph_json: dict[str, Any], namespace: str = "greenplum") -> dict[str, dict[str, Any]]:
    graph = graph_from_payload(graph_json)
    fields: dict[str, dict[str, Any]] = {}
    output_nodes = [
        (node, attrs)
        for node, attrs in graph.nodes(data=True)
        if attrs.get("node_type") == "output_column"
    ]
    for node, attrs in output_nodes:
        alias = attrs.get("alias") or node.split(".")[-1]
        input_fields: list[dict[str, str]] = []
        for pred in graph.predecessors(node):
            edge_data = graph.get_edge_data(pred, node) or {}
            for data in edge_data.values():
                if data.get("edge_type") != "DERIVED_FROM":
                    continue
                pred_attrs = graph.nodes.get(pred, {})
                if pred_attrs.get("node_type") != "source_column":
                    continue
                table = pred_attrs.get("physical_table") or pred_attrs.get("table_alias") or "unknown"
                column = pred_attrs.get("column") or pred.split(".")[-1]
                input_fields.append(
                    {
                        "namespace": namespace,
                        "name": str(table),
                        "field": str(column),
                    }
                )
        fields[alias] = {"inputFields": input_fields}
    return fields


def _resolve_output_datasets(
    graph_json: dict[str, Any],
    sql: str,
    *,
    namespace: str,
    dialect: str = "postgres",
    table_lineage: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Map ``output.{alias}`` nodes to the real target table from table lineage."""
    lineage = table_lineage or extract_table_lineage(sql, dialect=dialect)
    target = str(lineage.get("target") or "").strip()
    if target:
        return [_dataset(namespace, target.lower())]

    # Fall back to output.* names when table lineage cannot resolve a target.
    graph = graph_from_payload(graph_json)
    outputs: set[str] = set()
    for node, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "output_column":
            outputs.add(f"output.{attrs.get('alias', node)}")
    return [_dataset(namespace, name) for name in sorted(outputs)]


def _resolve_input_datasets(
    graph_json: dict[str, Any],
    sql: str,
    *,
    namespace: str,
    dialect: str = "postgres",
    table_lineage: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    lineage = table_lineage or extract_table_lineage(sql, dialect=dialect)
    sources = [str(s).lower() for s in (lineage.get("sources") or []) if s]
    if sources:
        return [_dataset(namespace, name) for name in sorted(set(sources))]

    graph = graph_from_payload(graph_json)
    inputs: set[str] = set()
    for node, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "source_column":
            table = attrs.get("physical_table") or attrs.get("table_alias") or node.split(".")[0]
            inputs.add(str(table).lower())
    return [_dataset(namespace, name) for name in sorted(inputs)]


def build_run_event(
    graph_json: dict[str, Any],
    sql: str,
    *,
    event_type: EventType = "START",
    namespace: str = "greenplum",
    job_namespace: str = "llm4lineage",
    run_id: str | None = None,
    dialect: str = "postgres",
    table_lineage: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Build a single OpenLineage run event (START / COMPLETE / FAIL / …)."""
    digest = sql_hash(sql)
    lineage = table_lineage or extract_table_lineage(sql, dialect=dialect)
    inputs = _resolve_input_datasets(
        graph_json, sql, namespace=namespace, dialect=dialect, table_lineage=lineage
    )
    outputs = _resolve_output_datasets(
        graph_json, sql, namespace=namespace, dialect=dialect, table_lineage=lineage
    )
    field_maps = _field_lineage(graph_json, namespace=namespace)

    # Attach column lineage facet to the first input for design-time consumers.
    if inputs and field_maps:
        first = dict(inputs[0])
        first["facets"] = {"columnLineage": {"fields": field_maps}}
        inputs = [first, *inputs[1:]]

    event: dict[str, Any] = {
        "eventType": event_type,
        "eventTime": datetime.now(timezone.utc).isoformat(),
        "job": {"namespace": job_namespace, "name": digest},
        "run": {
            "runId": run_id or str(uuid.uuid4()),
            "facets": {
                "sql": {"query": sql},
                "tableLineage": {
                    "target": lineage.get("target"),
                    "sources": lineage.get("sources") or [],
                },
            },
        },
        "inputs": inputs,
        "outputs": outputs,
        "producer": "https://github.com/Xpehutta/llm4lineage",
        "schemaURL": "https://openlineage.io/spec/2-0-2/OpenLineage.json",
    }
    if event_type == "FAIL" and error_message:
        event["run"]["facets"]["errorMessage"] = {
            "message": error_message,
            "programmingLanguage": "SQL",
        }
    return event


def run_lifecycle(
    graph_json: dict[str, Any],
    sql: str,
    *,
    namespace: str = "greenplum",
    job_namespace: str = "llm4lineage",
    dialect: str = "postgres",
    success: bool = True,
    error_message: str | None = None,
) -> list[dict[str, Any]]:
    """Emit START followed by COMPLETE or FAIL for one SQL analysis run."""
    run_id = str(uuid.uuid4())
    lineage = extract_table_lineage(sql, dialect=dialect)
    start = build_run_event(
        graph_json,
        sql,
        event_type="START",
        namespace=namespace,
        job_namespace=job_namespace,
        run_id=run_id,
        dialect=dialect,
        table_lineage=lineage,
    )
    end_type: EventType = "COMPLETE" if success else "FAIL"
    end = build_run_event(
        graph_json,
        sql,
        event_type=end_type,
        namespace=namespace,
        job_namespace=job_namespace,
        run_id=run_id,
        dialect=dialect,
        table_lineage=lineage,
        error_message=error_message,
    )
    return [start, end]


def to_openlineage_run_event(
    graph_json: dict[str, Any],
    sql: str,
    *,
    namespace: str = "greenplum",
    job_namespace: str = "llm4lineage",
) -> dict[str, Any]:
    """Backward-compatible START event."""
    return build_run_event(
        graph_json,
        sql,
        event_type="START",
        namespace=namespace,
        job_namespace=job_namespace,
    )


def to_openlineage_job_event(
    graph_json: dict[str, Any],
    sql: str,
    *,
    namespace: str = "greenplum",
    job_namespace: str = "llm4lineage",
) -> dict[str, Any]:
    """Backward-compatible COMPLETE event with real output dataset names."""
    return build_run_event(
        graph_json,
        sql,
        event_type="COMPLETE",
        namespace=namespace,
        job_namespace=job_namespace,
    )


def _emit(url: str, payload: dict[str, Any]) -> None:
    import urllib.parse
    import urllib.request

    # Restrict to HTTP(S): urlopen also honours file:// and custom schemes, which
    # would turn a mistyped --emit into a local file read.
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError(f"OpenLineage endpoint must be http or https, got {scheme or 'no'} scheme")

    data = json.dumps(payload).encode("utf-8")
    # Scheme is validated above, so only http(s) reaches urlopen.
    request = urllib.request.Request(  # noqa: S310
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        response.read()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export SQL lineage to OpenLineage JSON")
    parser.add_argument("--sql", required=True, help="Path to SQL file")
    parser.add_argument(
        "--format",
        choices=["run", "job", "lifecycle"],
        default="run",
        help="run=START, job=COMPLETE, lifecycle=START+COMPLETE",
    )
    parser.add_argument("--dialect", default="postgres")
    parser.add_argument("--namespace", default="greenplum")
    parser.add_argument("--emit", default="", help="Optional HTTP endpoint URL")
    parser.add_argument("--out", default="", help="Optional output JSON path")
    args = parser.parse_args(argv)

    sql = Path(args.sql).read_text(encoding="utf-8")
    pipeline = SQL2GraphPipeline(parser=SQL2GraphParser(dialect=args.dialect))
    result = pipeline.run(sql, dialect=args.dialect, use_llm_verify=False, use_llm_enhance=False)
    if "error" in result:
        print(result["error"], file=sys.stderr)
        return 1

    graph_json = result["graph"]
    if args.format == "lifecycle":
        payload: Any = run_lifecycle(
            graph_json, sql, namespace=args.namespace, dialect=args.dialect
        )
    elif args.format == "job":
        payload = to_openlineage_job_event(
            graph_json, sql, namespace=args.namespace
        )
    else:
        payload = to_openlineage_run_event(
            graph_json, sql, namespace=args.namespace
        )

    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)

    if args.emit:
        events = payload if isinstance(payload, list) else [payload]
        for event in events:
            _emit(args.emit, event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
