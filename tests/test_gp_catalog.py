"""Tests for GreenPlum catalog extraction against a fake catalog (no database needed)."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch

from Classes.gp_catalog import (
    DEFAULT_STATE_PATH,
    CatalogState,
    GPCatalogError,
    GPCatalogExtractor,
    PsycopgNotInstalledError,
    main,
    redact_dsn,
)
from Classes.schema_registry import SchemaRegistry

FUNCTION_BODY = """CREATE OR REPLACE FUNCTION sales.load_orders()
 RETURNS void
 LANGUAGE plpgsql
AS $function$
BEGIN
    INSERT INTO sales.orders_daily SELECT order_id, amount FROM sales.orders;
END;
$function$"""

CATALOG_ROWS: dict[str, list[tuple[Any, ...]]] = {
    "functions": [("sales", "load_orders", FUNCTION_BODY)],
    "views": [("sales", "v_orders", "SELECT o.order_id, o.amount FROM sales.orders o")],
    "matviews": [("sales", "mv_totals", "SELECT o.order_id FROM sales.orders o")],
    "columns": [
        ("sales", "orders", "order_id", "integer", 1),
        ("sales", "orders", "order", "character varying", 2),
        ("sales", "orders", "amount", "numeric", 3),
        ("sales", "orders", "tags", "ARRAY", 4),
        ("sales", "orders", "payload", "USER-DEFINED", 5),
        ("sales", "v_orders", "order_id", "integer", 1),
        ("sales", "v_orders", "amount", "numeric", 2),
        ("staging", "ext_payments", "payment_id", "bigint", 1),
        ("staging", "ext_payments", "paid_at", "timestamp without time zone", 2),
    ],
    "external": [("staging", "ext_payments", "gpfdist://etl:8081/payments.csv", "c")],
}


class FakeUndefinedTable(Exception):
    """Stand-in for psycopg2.errors.UndefinedTable."""

    pgcode = "42P01"

    def __init__(self, relation: str):
        super().__init__(f'relation "{relation}" does not exist')


class FakeCursor:
    """Cursor that routes catalog queries to canned rows."""

    def __init__(self, connection: FakeConnection):
        self._connection = connection
        self._rows: list[tuple[Any, ...]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> None:
        self._connection.executed.append(sql)
        relation = self._relation_for(sql)
        if relation is None:
            self._rows = []
            return
        if relation in self._connection.missing:
            raise FakeUndefinedTable(relation)
        self._rows = list(self._connection.rows.get(self._key_for(relation), []))

    @staticmethod
    def _relation_for(sql: str) -> str | None:
        if "pg_get_functiondef" in sql:
            return "pg_proc"
        if "pg_matviews" in sql:
            return "pg_matviews"
        if "pg_views" in sql:
            return "pg_views"
        if "information_schema.columns" in sql:
            return "information_schema.columns"
        if "pg_exttable" in sql:
            return "pg_exttable"
        return None

    @staticmethod
    def _key_for(relation: str) -> str:
        return {
            "pg_proc": "functions",
            "pg_views": "views",
            "pg_matviews": "matviews",
            "information_schema.columns": "columns",
            "pg_exttable": "external",
        }[relation]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)

    def close(self) -> None:
        self._rows = []


class FakeConnection:
    """Minimal psycopg2-like connection backed by canned catalog rows."""

    def __init__(
        self,
        rows: dict[str, list[tuple[Any, ...]]] | None = None,
        *,
        missing: Sequence[str] = (),
    ):
        self.rows = {key: list(value) for key, value in (rows or CATALOG_ROWS).items()}
        self.missing = set(missing)
        self.executed: list[str] = []
        self.readonly: bool | None = None
        self.autocommit: bool | None = None
        self.rollbacks = 0
        self.closed = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def set_session(self, readonly: bool = False, autocommit: bool = False) -> None:
        self.readonly = readonly
        self.autocommit = autocommit

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


def build_extractor(
    connection: FakeConnection,
    state_path: str | Path = DEFAULT_STATE_PATH,
) -> GPCatalogExtractor:
    return GPCatalogExtractor(
        dsn="postgresql://user:secret@localhost:5432/dwh",
        connection=connection,
        state_path=state_path,
        load_env=False,
    )


class TestGPCatalogQueries(unittest.TestCase):
    def test_functions_query_matches_spec(self):
        extractor = build_extractor(FakeConnection())
        sql = extractor.functions_sql()
        self.assertIn("pg_get_functiondef(p.oid)", sql)
        self.assertIn("JOIN pg_namespace n ON n.oid = p.pronamespace", sql)
        self.assertIn("lanname IN ('plpgsql', 'sql')", sql)
        self.assertIn("n.nspname NOT IN ('pg_catalog', 'information_schema', 'gp_toolkit')", sql)

    def test_read_only_session_is_applied(self):
        connection = FakeConnection()
        extractor = build_extractor(connection)
        extractor.snapshot()
        self.assertTrue(connection.readonly)
        self.assertFalse(any("INSERT" in sql.upper() for sql in connection.executed))

    def test_custom_schema_exclusions_are_validated(self):
        extractor = GPCatalogExtractor(
            connection=FakeConnection(),
            exclude_schemas=("pg_catalog", "tmp_stage"),
            load_env=False,
        )
        self.assertIn("NOT IN ('pg_catalog', 'tmp_stage')", extractor.views_sql())

        hostile = GPCatalogExtractor(
            connection=FakeConnection(),
            exclude_schemas=("public'); DROP TABLE t; --",),
            load_env=False,
        )
        with self.assertRaises(GPCatalogError):
            hostile.views_sql()

    def test_iter_functions_yields_definitions(self):
        extractor = build_extractor(FakeConnection())
        functions = list(extractor.iter_functions())
        self.assertEqual(len(functions), 1)
        self.assertEqual(functions[0].qualified_name, "sales.load_orders")
        self.assertIn("LANGUAGE plpgsql", functions[0].definition)
        self.assertEqual(functions[0].key, "function:sales.load_orders")


class TestGPCatalogDDLText(unittest.TestCase):
    def test_ddl_text_is_consumable_by_schema_registry(self):
        extractor = build_extractor(FakeConnection())
        ddl_text = extractor.dump_ddl_text()

        registry = SchemaRegistry(dialect="postgres").load_ddl(ddl_text)
        orders = registry.table_columns("sales", "orders")
        self.assertEqual(
            set(orders.keys()),
            {"order_id", "order", "amount", "tags", "payload"},
        )
        self.assertIn("payment_id", registry.table_columns("staging", "ext_payments"))
        self.assertTrue(registry.is_view("sales", "v_orders"))
        self.assertTrue(registry.is_view("sales", "mv_totals"))
        self.assertIn("order_id", registry.table_columns("sales", "v_orders"))

    def test_ddl_text_marks_external_tables_and_omits_functions_by_default(self):
        extractor = build_extractor(FakeConnection())
        ddl_text = extractor.dump_ddl_text()
        self.assertIn("-- EXTERNAL TABLE (source) staging.ext_payments", ddl_text)
        self.assertIn("gpfdist://etl:8081/payments.csv", ddl_text)
        self.assertNotIn("CREATE OR REPLACE FUNCTION", ddl_text)

        with_functions = extractor.dump_ddl_text(include_functions=True)
        self.assertIn("sales.load_orders", with_functions)
        SchemaRegistry(dialect="postgres").load_ddl(with_functions)

    def test_reserved_identifiers_and_unsupported_types_are_normalized(self):
        extractor = build_extractor(FakeConnection())
        ddl_text = extractor.dump_ddl_text()
        self.assertIn('"order" CHARACTER VARYING', ddl_text)
        self.assertIn("tags TEXT", ddl_text)
        self.assertIn("payload TEXT", ddl_text)

    def test_unparseable_view_is_commented_out(self):
        rows = {key: list(value) for key, value in CATALOG_ROWS.items()}
        rows["views"] = list(rows["views"]) + [("sales", "v_broken", "NOT A SELECT AT ALL <<>>")]
        extractor = build_extractor(FakeConnection(rows))
        ddl_text = extractor.dump_ddl_text()
        self.assertIn("-- UNPARSEABLE view sales.v_broken", ddl_text)
        self.assertTrue(any("not parseable" in warning for warning in extractor.warnings))

        registry = SchemaRegistry(dialect="postgres").load_ddl(ddl_text)
        self.assertFalse(registry.is_view("sales", "v_broken"))
        self.assertIn("order_id", registry.table_columns("sales", "orders"))

    def test_functions_script_contains_definitions(self):
        extractor = build_extractor(FakeConnection())
        script = extractor.dump_functions_sql()
        self.assertIn("CREATE OR REPLACE FUNCTION sales.load_orders()", script)
        self.assertTrue(script.rstrip().endswith(";"))


class TestGPCatalogCsvDump(unittest.TestCase):
    def test_dump_csv_writes_registry_compatible_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "gp_dump"
            extractor = build_extractor(FakeConnection(), state_path=Path(tmp) / "state.json")
            written = extractor.dump_csv(out_dir)

            self.assertEqual(
                set(written),
                {"columns", "views", "functions", "external_tables"},
            )
            for path in written.values():
                self.assertTrue(Path(path).exists())

            csv_text = Path(written["columns"]).read_text(encoding="utf-8")
            registry = SchemaRegistry(dialect="postgres").load_csv(csv_text)
            self.assertIn("amount", registry.table_columns("sales", "orders"))

            external_text = Path(written["external_tables"]).read_text(encoding="utf-8")
            self.assertIn("gpfdist://etl:8081/payments.csv", external_text)


class TestGPCatalogIncrementality(unittest.TestCase):
    def test_second_run_without_changes_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "gp_dump"
            state_path = Path(tmp) / "gp_dump_state.json"

            first = build_extractor(FakeConnection(), state_path=state_path).dump(out_dir)
            self.assertFalse(first.skipped)
            self.assertTrue(first.has_changes)
            self.assertIn("function:sales.load_orders", first.changed)
            self.assertTrue(state_path.exists())
            state_after_first = state_path.read_text(encoding="utf-8")
            ddl_after_first = (out_dir / "ddl.sql").read_text(encoding="utf-8")

            second = build_extractor(FakeConnection(), state_path=state_path).dump(out_dir)
            self.assertTrue(second.skipped)
            self.assertEqual(second.changed, [])
            self.assertEqual(second.removed, [])
            self.assertGreater(second.unchanged, 0)
            self.assertEqual(state_path.read_text(encoding="utf-8"), state_after_first)
            self.assertEqual((out_dir / "ddl.sql").read_text(encoding="utf-8"), ddl_after_first)

    def test_changed_and_removed_objects_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "gp_dump"
            state_path = Path(tmp) / "gp_dump_state.json"
            build_extractor(FakeConnection(), state_path=state_path).dump(out_dir)

            rows = {key: list(value) for key, value in CATALOG_ROWS.items()}
            rows["views"] = [("sales", "v_orders", "SELECT o.order_id FROM sales.orders o")]
            rows["matviews"] = []
            extractor = build_extractor(FakeConnection(rows), state_path=state_path)

            diff = extractor.changed_objects()
            self.assertIn("view:sales.v_orders", diff.changed)
            self.assertIn("matview:sales.mv_totals", diff.removed)
            self.assertIn("function:sales.load_orders", diff.unchanged)

            report = extractor.dump(out_dir)
            self.assertFalse(report.skipped)
            payload = json.loads((out_dir / "changed_objects.json").read_text(encoding="utf-8"))
            self.assertIn("view:sales.v_orders", payload["changed"])

    def test_iter_functions_only_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "gp_dump_state.json"
            build_extractor(FakeConnection(), state_path=state_path).dump(Path(tmp) / "gp_dump")

            unchanged = build_extractor(FakeConnection(), state_path=state_path)
            self.assertEqual(list(unchanged.iter_functions(only_changed=True)), [])

            rows = {key: list(value) for key, value in CATALOG_ROWS.items()}
            rows["functions"] = [("sales", "load_orders", FUNCTION_BODY + "\n-- touched")]
            changed = build_extractor(FakeConnection(rows), state_path=state_path)
            self.assertEqual(len(list(changed.iter_functions(only_changed=True))), 1)

    def test_state_path_is_configurable(self):
        self.assertEqual(DEFAULT_STATE_PATH, "data/gp_dump_state.json")
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "nested" / "custom_state.json"
            report = build_extractor(FakeConnection(), state_path=state_path).dump(Path(tmp) / "out")
            self.assertEqual(report.state_path, str(state_path))
            self.assertTrue(state_path.exists())
            self.assertIn("objects", json.loads(state_path.read_text(encoding="utf-8")))

    def test_corrupt_state_file_raises_explicit_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            state_path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(GPCatalogError):
                CatalogState.load(state_path)


class TestGPCatalogDegradation(unittest.TestCase):
    def test_missing_pg_exttable_degrades_gracefully(self):
        connection = FakeConnection(missing=["pg_exttable"])
        extractor = build_extractor(connection)
        snapshot = extractor.snapshot()

        self.assertEqual(snapshot.external_tables, [])
        self.assertTrue(any("pg_exttable" in warning for warning in extractor.warnings))
        self.assertEqual(connection.rollbacks, 1)
        self.assertEqual(len(snapshot.functions), 1)
        registry = SchemaRegistry(dialect="postgres").load_ddl(extractor.dump_ddl_text())
        self.assertIn("payment_id", registry.table_columns("staging", "ext_payments"))

    def test_missing_pg_matviews_degrades_gracefully(self):
        extractor = build_extractor(FakeConnection(missing=["pg_matviews"]))
        self.assertEqual(extractor.fetch_materialized_views(), [])
        self.assertTrue(any("pg_matviews" in warning for warning in extractor.warnings))

    def test_unexpected_query_errors_are_not_swallowed(self):
        extractor = build_extractor(FakeConnection(missing=["pg_views"]))
        with self.assertRaises(FakeUndefinedTable):
            extractor.fetch_views()


class TestGPCatalogWithoutPsycopg2(unittest.TestCase):
    def test_module_imports_without_psycopg2(self):
        with patch.dict(sys.modules, {"psycopg2": None}):
            import importlib

            module = importlib.import_module("Classes.gp_catalog")
            self.assertTrue(hasattr(module, "GPCatalogExtractor"))

    def test_connect_without_psycopg2_raises_actionable_error(self):
        extractor = GPCatalogExtractor(dsn="postgresql://user:secret@localhost/dwh", load_env=False)
        with patch.dict(sys.modules, {"psycopg2": None}):
            with self.assertRaises(PsycopgNotInstalledError) as ctx:
                extractor.connect()
        message = str(ctx.exception)
        self.assertIn("llm4lineage[gp]", message)
        self.assertNotIn("secret", message)

    def test_missing_dsn_error_is_explicit(self):
        extractor = GPCatalogExtractor(load_env=False)
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(GPCatalogError) as ctx:
                extractor.resolve_dsn()
        self.assertIn("GP_DSN", str(ctx.exception))

    def test_dsn_from_environment_and_redaction(self):
        extractor = GPCatalogExtractor(load_env=False)
        with patch.dict("os.environ", {"GP_DSN": "postgresql://u:secret@h:5432/db"}, clear=True):
            self.assertEqual(extractor.resolve_dsn(), "postgresql://u:secret@h:5432/db")
            self.assertNotIn("secret", extractor.target)

        with patch.dict(
            "os.environ",
            {"GP_HOST": "gp", "GP_DATABASE": "dwh", "GP_USER": "ro", "GP_PASSWORD": "secret"},
            clear=True,
        ):
            self.assertIn("host=gp", extractor.resolve_dsn())
            self.assertNotIn("secret", redact_dsn(extractor.resolve_dsn()))


class TestGPCatalogCli(unittest.TestCase):
    def test_cli_reports_missing_driver_without_leaking_dsn(self):
        stderr = io.StringIO()
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / "empty.env"
            env_file.write_text("", encoding="utf-8")
            argv = [
                "--dsn",
                "postgresql://user:secret@localhost:5432/dwh",
                "--out",
                str(Path(tmp) / "out"),
                "--state",
                str(Path(tmp) / "state.json"),
                "--env-file",
                str(env_file),
            ]
            # patch.dict restores os.environ so a real .env cannot leak into other tests.
            with patch.dict("os.environ", {}), patch.dict(sys.modules, {"psycopg2": None}):
                with redirect_stderr(stderr), redirect_stdout(stdout):
                    code = main(argv)

        self.assertEqual(code, 1)
        self.assertIn("psycopg2", stderr.getvalue())
        self.assertNotIn("secret", stderr.getvalue())
        self.assertNotIn("secret", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
