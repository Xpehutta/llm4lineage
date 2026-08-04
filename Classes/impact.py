"""CLI entry point for impact analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from Classes.impact_analyzer import analyze_impact
from Classes.sql2graph_classes import SQL2GraphParser, SQL2GraphPipeline


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description="Analyze column lineage impact")
  parser.add_argument("--sql", required=True, help="Path to SQL file")
  parser.add_argument("--target", required=True, help="Target node, e.g. output.total or alias.column")
  parser.add_argument("--direction", choices=["up", "down", "both"], default="both")
  parser.add_argument("--dialect", default="postgres")
  args = parser.parse_args(argv)

  sql = Path(args.sql).read_text(encoding="utf-8")
  pipeline = SQL2GraphPipeline(parser=SQL2GraphParser(dialect=args.dialect))
  result = pipeline.run(sql, dialect=args.dialect, use_llm_verify=False, use_llm_enhance=False)
  if "error" in result:
    print(result["error"], file=sys.stderr)
    return 1

  target = args.target
  if not target.startswith("output.") and "." in target:
    target = f"output.{target.split('.')[-1]}"

  report = analyze_impact(result["graph"], target, direction=args.direction)
  print(json.dumps(report, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
