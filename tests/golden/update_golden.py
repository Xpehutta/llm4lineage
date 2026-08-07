#!/usr/bin/env python3
"""Regenerate golden graph fixtures (explicit opt-in).

Run with ``--check`` to compare instead of write: that is the dry-run CI uses
to catch silent golden drift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from Classes.sql2graph_classes import SQL2GraphParser, SQL2GraphPipeline

DEFAULT_SQL = "data/DDLs_10.txt"
DEFAULT_OUT = "tests/golden/ddls10_first_graph.json"


def build_payload(sql_path: str | Path = DEFAULT_SQL) -> dict[str, Any]:
    """Produce the golden payload for the first statement in ``sql_path``."""
    sql = Path(sql_path).read_text(encoding="utf-8").split(";")[0].strip()
    pipeline = SQL2GraphPipeline(parser=SQL2GraphParser(dialect="postgres"))
    result = pipeline.run(sql, dialect="postgres", use_llm_verify=False, use_llm_enhance=False)
    links = [
        {key: link[key] for key in ("source", "target", "edge_type") if key in link}
        for link in result["graph"].get("links", [])
    ]
    return {"nodes": len(result["graph"].get("nodes", [])), "links": links}


def diff_against_golden(
    sql_path: str | Path = DEFAULT_SQL,
    out_path: str | Path = DEFAULT_OUT,
) -> list[str]:
    """Return human-readable differences between a fresh run and the fixture.

    An empty list means the fixture is still accurate.
    """
    golden_file = Path(out_path)
    if not golden_file.exists():
        return [f"Golden file {golden_file} is missing; run tests/golden/update_golden.py"]

    expected = json.loads(golden_file.read_text(encoding="utf-8"))
    actual = build_payload(sql_path)

    differences: list[str] = []
    if expected.get("nodes") != actual.get("nodes"):
        differences.append(f"node count: golden {expected.get('nodes')} != current {actual['nodes']}")

    def as_set(payload: dict[str, Any]) -> set[tuple[str, str, str]]:
        return {
            (link.get("source", ""), link.get("target", ""), link.get("edge_type", ""))
            for link in payload.get("links", [])
        }

    expected_links, actual_links = as_set(expected), as_set(actual)
    for missing in sorted(expected_links - actual_links):
        differences.append(f"edge disappeared: {missing[0]} -> {missing[1]} ({missing[2]})")
    for added in sorted(actual_links - expected_links):
        differences.append(f"edge appeared: {added[0]} -> {added[1]} ({added[2]})")
    return differences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sql", default=DEFAULT_SQL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report drift without rewriting the fixture; exits non-zero when it differs.",
    )
    args = parser.parse_args()

    if args.check:
        differences = diff_against_golden(args.sql, args.out)
        if not differences:
            print(f"{args.out} is up to date")
            return 0
        print(f"{args.out} has drifted:")
        for line in differences:
            print(f"  - {line}")
        return 1

    payload = build_payload(args.sql)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({len(payload['links'])} links)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
