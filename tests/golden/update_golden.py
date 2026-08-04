#!/usr/bin/env python3
"""Regenerate golden graph fixtures (explicit opt-in)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Classes.sql2graph_classes import SQL2GraphParser, SQL2GraphPipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sql", default="data/DDLs_10.txt")
    parser.add_argument("--out", default="tests/golden/ddls10_first_graph.json")
    args = parser.parse_args()

    sql = Path(args.sql).read_text(encoding="utf-8").split(";")[0].strip()
    pipeline = SQL2GraphPipeline(parser=SQL2GraphParser(dialect="postgres"))
    result = pipeline.run(sql, dialect="postgres", use_llm_verify=False, use_llm_enhance=False)
    links = [
        {key: link[key] for key in ("source", "target", "edge_type") if key in link}
        for link in result["graph"].get("links", [])
    ]
    payload = {"nodes": len(result["graph"].get("nodes", [])), "links": links}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({len(links)} links)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
