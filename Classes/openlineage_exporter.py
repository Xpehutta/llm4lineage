"""Export SQL2Graph lineage to OpenLineage design-time events."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Classes.impact_analyzer import graph_from_payload
from Classes.sql2graph_classes import SQL2GraphParser, SQL2GraphPipeline


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


def to_openlineage_run_event(
    graph_json: dict[str, Any],
    sql: str,
    *,
    namespace: str = "greenplum",
    job_namespace: str = "llm4lineage",
) -> dict[str, Any]:
  digest = sql_hash(sql)
  graph = graph_from_payload(graph_json)
  inputs: set[tuple[str, str]] = set()
  outputs: set[tuple[str, str]] = set()
  for node, attrs in graph.nodes(data=True):
    if attrs.get("node_type") == "source_column":
      table = attrs.get("physical_table") or attrs.get("table_alias") or node.split(".")[0]
      inputs.add((namespace, str(table).lower()))
    if attrs.get("node_type") == "output_column":
      outputs.add((namespace, f"output.{attrs.get('alias', node)}"))

  field_maps = _field_lineage(graph_json, namespace=namespace)
  return {
    "eventType": "START",
    "eventTime": datetime.now(timezone.utc).isoformat(),
    "job": {"namespace": job_namespace, "name": digest},
    "inputs": [
      {
        **_dataset(ns, name),
        "facets": {"columnLineage": {"fields": field_maps}} if field_maps else {},
      }
      for ns, name in sorted(inputs)
    ],
    "outputs": [_dataset(namespace, name) for ns, name in sorted(outputs)],
    "run": {"facets": {"sql": {"query": sql}}},
  }


def to_openlineage_job_event(
    graph_json: dict[str, Any],
    sql: str,
    *,
    namespace: str = "greenplum",
    job_namespace: str = "llm4lineage",
) -> dict[str, Any]:
  event = to_openlineage_run_event(graph_json, sql, namespace=namespace, job_namespace=job_namespace)
  event["eventType"] = "COMPLETE"
  return event


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
  parser.add_argument("--format", choices=["run", "job"], default="run")
  parser.add_argument("--dialect", default="postgres")
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
  payload = (
    to_openlineage_job_event(graph_json, sql)
    if args.format == "job"
    else to_openlineage_run_event(graph_json, sql)
  )
  text = json.dumps(payload, indent=2)
  if args.out:
    Path(args.out).write_text(text, encoding="utf-8")
  else:
    print(text)

  if args.emit:
    _emit(args.emit, payload)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
