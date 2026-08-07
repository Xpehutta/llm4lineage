"""Daily lineage pipeline DAG: extract → parse → build → publish.

The module is importable without Airflow. When ``airflow`` is available the
``lineage_daily`` DAG object is registered; otherwise the callable steps can be
exercised directly from tests / CLI.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DUMP_DIR = Path("data/gp_dump")
DEFAULT_STATE_PATH = Path("data/gp_dump_state.json")
DEFAULT_OUT_DIR = Path("data/lineage_out")


def extract_catalog(
    *,
    dsn: str | None = None,
    out_dir: str | Path = DEFAULT_DUMP_DIR,
    state_path: str | Path = DEFAULT_STATE_PATH,
) -> dict[str, Any]:
    """Pull changed DDL from GreenPlum (no-op when DSN is absent)."""
    from Classes.gp_catalog import GPCatalogExtractor

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not dsn:
        logger.info("No GP DSN configured — skipping live catalog extract")
        ddl_path = out / "schema.sql"
        return {
            "skipped": True,
            "ddl_path": str(ddl_path) if ddl_path.exists() else "",
            "changed": 0,
        }

    extractor = GPCatalogExtractor(dsn=dsn, state_path=str(state_path))
    changed = list(extractor.iter_functions(only_changed=True))
    ddl_text = extractor.dump_ddl_text(include_functions=True)
    ddl_path = out / "schema.sql"
    ddl_path.write_text(ddl_text, encoding="utf-8")
    extractor.dump_csv(out)
    return {"skipped": False, "ddl_path": str(ddl_path), "changed": len(changed)}


def parse_and_build(
    *,
    sql_path: str | Path,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    dialect: str = "postgres",
    parse_plpgsql: bool = True,
) -> dict[str, Any]:
    """Run SQL2Graph (and optional PL/pgSQL routing) and persist graph JSON."""
    from Classes.sql2graph_classes import SQL2GraphParser, SQL2GraphPipeline

    sql_file = Path(sql_path)
    if not sql_file.exists():
        raise FileNotFoundError(f"SQL input not found: {sql_file}")
    sql = sql_file.read_text(encoding="utf-8")
    pipeline = SQL2GraphPipeline(parser=SQL2GraphParser(dialect=dialect))
    result = pipeline.run(
        sql,
        dialect=dialect,
        use_llm_verify=False,
        use_llm_enhance=False,
        parse_plpgsql=parse_plpgsql,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graph_path = out / f"{sql_file.stem}_graph.json"
    graph_path.write_text(json.dumps(result.get("graph") or {}, indent=2), encoding="utf-8")
    return {
        "graph_path": str(graph_path),
        "pipeline_stage": result.get("pipeline_stage"),
        "error": result.get("error"),
    }


def publish_openlineage(
    *,
    graph_path: str | Path,
    sql_path: str | Path,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    namespace: str = "greenplum",
) -> dict[str, Any]:
    """Write OpenLineage START+COMPLETE lifecycle events next to the graph."""
    from Classes.openlineage_exporter import run_lifecycle

    graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    sql = Path(sql_path).read_text(encoding="utf-8")
    events = run_lifecycle(graph, sql, namespace=namespace, success=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    events_path = out / "openlineage_lifecycle.json"
    events_path.write_text(json.dumps(events, indent=2), encoding="utf-8")
    return {"events_path": str(events_path), "event_types": [e["eventType"] for e in events]}


def repository_changed(state_path: str | Path = DEFAULT_STATE_PATH) -> bool:
    """Cheap sensor: True when dump state is missing or marked dirty."""
    path = Path(state_path)
    if not path.exists():
        return True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True
    return bool(payload.get("dirty")) or bool(payload.get("objects"))


# ---------------------------------------------------------------------------
# Optional Airflow DAG registration
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only when Airflow is installed
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    default_args = {
        "owner": "llm4lineage",
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    }

    with DAG(
        dag_id="lineage_daily",
        default_args=default_args,
        description="Extract GP catalog → parse lineage → publish OpenLineage",
        schedule="@daily",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["lineage", "greenplum"],
    ) as dag:
        extract = PythonOperator(
            task_id="extract_catalog",
            python_callable=extract_catalog,
        )
        parse = PythonOperator(
            task_id="parse_and_build",
            python_callable=parse_and_build,
            op_kwargs={"sql_path": str(DEFAULT_DUMP_DIR / "schema.sql")},
        )
        publish = PythonOperator(
            task_id="publish_openlineage",
            python_callable=publish_openlineage,
            op_kwargs={
                "graph_path": str(DEFAULT_OUT_DIR / "schema_graph.json"),
                "sql_path": str(DEFAULT_DUMP_DIR / "schema.sql"),
            },
        )
        extract >> parse >> publish

except ImportError:  # pragma: no cover
    dag = None  # type: ignore[assignment]
