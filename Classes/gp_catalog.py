"""Extract DDL from a GreenPlum/PostgreSQL catalog into SchemaRegistry-compatible dumps.

The extractor is read-only by design: it opens a read-only session and never issues DML.
``psycopg2`` is an optional dependency (extra ``gp``) and is imported lazily, so this module
can be imported — and unit-tested against a fake catalog — without a database driver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import sqlglot
    from sqlglot import exp
except Exception:  # pragma: no cover - sqlglot is a hard dependency, guard mirrors repo style
    sqlglot = None
    exp = None

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = "data/gp_dump_state.json"
DEFAULT_EXCLUDED_SCHEMAS: tuple[str, ...] = ("pg_catalog", "information_schema", "gp_toolkit")
DSN_ENV_VARS: tuple[str, ...] = ("GP_DSN", "GREENPLUM_DSN", "DATABASE_URL")

_MISSING_OBJECT_SQLSTATES = frozenset({"42P01", "42703"})
_PLAIN_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_SAFE_TYPE_RE = re.compile(r"^[a-z][a-z0-9_ ]*$")
_SCHEMA_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")

_TYPE_ALIASES: dict[str, str] = {
    "array": "TEXT",
    "user-defined": "TEXT",
    "unknown": "TEXT",
    "anyarray": "TEXT",
    '"char"': "CHAR",
}

# Identifiers that must be quoted to survive a sqlglot round-trip.
_RESERVED_IDENTIFIERS = frozenset(
    {
        "all", "analyse", "analyze", "and", "any", "array", "as", "asc", "asymmetric",
        "authorization", "between", "binary", "both", "case", "cast", "check", "collate",
        "column", "constraint", "create", "cross", "current_date", "current_role",
        "current_time", "current_timestamp", "current_user", "default", "deferrable",
        "desc", "distinct", "do", "else", "end", "except", "false", "for", "foreign",
        "freeze", "from", "full", "grant", "group", "having", "ilike", "in", "initially",
        "inner", "intersect", "into", "is", "isnull", "join", "leading", "left", "like",
        "limit", "localtime", "localtimestamp", "natural", "not", "notnull", "null",
        "offset", "on", "only", "or", "order", "outer", "overlaps", "placing", "primary",
        "references", "returning", "right", "select", "session_user", "similar", "some",
        "symmetric", "table", "then", "to", "trailing", "true", "union", "unique", "user",
        "using", "values", "verbose", "when", "where", "window", "with",
    }
)

FUNCTIONS_SQL_TEMPLATE = """SELECT n.nspname, p.proname, pg_get_functiondef(p.oid)
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE p.prolang IN (SELECT oid FROM pg_language WHERE lanname IN ('plpgsql', 'sql'))
  AND {schema_filter}
ORDER BY n.nspname, p.proname"""

VIEWS_SQL_TEMPLATE = """SELECT schemaname, viewname, definition
FROM pg_views
WHERE {schema_filter}
ORDER BY schemaname, viewname"""

MATVIEWS_SQL_TEMPLATE = """SELECT schemaname, matviewname, definition
FROM pg_matviews
WHERE {schema_filter}
ORDER BY schemaname, matviewname"""

COLUMNS_SQL_TEMPLATE = """SELECT table_schema, table_name, column_name, data_type, ordinal_position
FROM information_schema.columns
WHERE {schema_filter}
ORDER BY table_schema, table_name, ordinal_position"""

EXTERNAL_TABLES_SQL_TEMPLATE = """SELECT n.nspname, c.relname,
       array_to_string(x.urilocation, ',') AS location, x.fmttype
FROM pg_exttable x
JOIN pg_class c ON c.oid = x.reloid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE {schema_filter}
ORDER BY n.nspname, c.relname"""


class GPCatalogError(RuntimeError):
    """Base error for GreenPlum catalog extraction."""


class PsycopgNotInstalledError(GPCatalogError):
    """Raised when a database operation is attempted without ``psycopg2`` installed."""


def quote_identifier(name: str) -> str:
    """Return ``name`` quoted only when a bare identifier would be ambiguous."""
    cleaned = (name or "").strip().strip('"')
    if _PLAIN_IDENT_RE.match(cleaned) and cleaned not in _RESERVED_IDENTIFIERS:
        return cleaned
    return '"' + cleaned.replace('"', '""') + '"'


def normalize_data_type(data_type: str | None) -> str:
    """Map an ``information_schema`` type name onto a type sqlglot can parse."""
    raw = (data_type or "").strip().lower()
    if not raw:
        return "TEXT"
    if raw in _TYPE_ALIASES:
        return _TYPE_ALIASES[raw]
    if not _SAFE_TYPE_RE.match(raw):
        return "TEXT"
    return raw.upper()


def redact_dsn(dsn: str) -> str:
    """Strip credentials from a DSN so it can be shown in logs and error messages."""
    if not dsn:
        return ""
    redacted = re.sub(r"(?i)(password\s*=\s*)\S+", r"\1***", dsn)
    return re.sub(r"://[^/@\s]+@", "://***@", redacted)


def definition_hash(definition: str) -> str:
    """Return a stable hash of an object definition."""
    return hashlib.sha256((definition or "").encode("utf-8")).hexdigest()


def _schema_filter(column: str, excluded: Sequence[str]) -> str:
    """Render a ``NOT IN`` schema filter from validated schema names."""
    names = [name for name in (excluded or ()) if name]
    for name in names:
        if not _SCHEMA_NAME_RE.match(name):
            raise GPCatalogError(f"Invalid schema name for exclusion filter: {name!r}")
    if not names:
        return "TRUE"
    rendered = ", ".join(f"'{name}'" for name in names)
    return f"{column} NOT IN ({rendered})"


def _import_psycopg2() -> Any:
    """Import ``psycopg2`` lazily with an actionable error when it is missing."""
    try:
        import psycopg2
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise PsycopgNotInstalledError(
            "psycopg2 is required to read the GreenPlum catalog but is not installed. "
            "Install the optional extra: pip install 'llm4lineage[gp]' "
            "(or pip install 'psycopg2-binary>=2.9')."
        ) from exc
    return psycopg2


def _is_missing_object_error(exc: BaseException) -> bool:
    """Detect undefined table/column failures without importing ``psycopg2``."""
    if getattr(exc, "pgcode", None) in _MISSING_OBJECT_SQLSTATES:
        return True
    if type(exc).__name__ in {"UndefinedTable", "UndefinedColumn"}:
        return True
    message = str(exc).lower()
    return "does not exist" in message and ("relation" in message or "column" in message)


@dataclass
class CatalogObject:
    """A single catalog object together with the definition used for hashing."""

    kind: str
    schema: str
    name: str
    definition: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.schema}.{self.name}"

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"

    @property
    def definition_hash(self) -> str:
        return definition_hash(self.definition)


@dataclass
class ColumnRecord:
    """One row of ``information_schema.columns``."""

    schema: str
    table: str
    column: str
    data_type: str
    ordinal_position: int = 0


@dataclass
class CatalogDiff:
    """Difference between the persisted state and the current catalog."""

    changed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.changed or self.removed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed": list(self.changed),
            "unchanged": len(self.unchanged),
            "removed": list(self.removed),
        }


@dataclass
class DumpReport:
    """Outcome of a catalog dump."""

    out_dir: str
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: int = 0
    files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: bool = False
    state_path: str = ""

    @property
    def has_changes(self) -> bool:
        return bool(self.changed or self.removed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_dir": self.out_dir,
            "state_path": self.state_path,
            "skipped": self.skipped,
            "changed": list(self.changed),
            "removed": list(self.removed),
            "unchanged": self.unchanged,
            "files": list(self.files),
            "warnings": list(self.warnings),
        }


@dataclass
class CatalogSnapshot:
    """Raw catalog payload fetched from the database."""

    functions: list[CatalogObject] = field(default_factory=list)
    views: list[CatalogObject] = field(default_factory=list)
    materialized_views: list[CatalogObject] = field(default_factory=list)
    external_tables: list[CatalogObject] = field(default_factory=list)
    columns: list[ColumnRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def external_keys(self) -> set[tuple[str, str]]:
        return {(obj.schema.lower(), obj.name.lower()) for obj in self.external_tables}

    def columns_by_table(self) -> dict[tuple[str, str], list[ColumnRecord]]:
        grouped: dict[tuple[str, str], list[ColumnRecord]] = {}
        for record in self.columns:
            grouped.setdefault((record.schema, record.table), []).append(record)
        for records in grouped.values():
            records.sort(key=lambda item: (item.ordinal_position, item.column))
        return grouped

    def table_objects(self) -> list[CatalogObject]:
        """Render one ``CREATE TABLE`` object per relation seen in the column list."""
        view_keys = {
            (obj.schema.lower(), obj.name.lower())
            for obj in list(self.views) + list(self.materialized_views)
        }
        external_keys = self.external_keys()
        objects: list[CatalogObject] = []
        for (schema_name, table_name), records in sorted(self.columns_by_table().items()):
            if (schema_name.lower(), table_name.lower()) in view_keys:
                continue
            is_external = (schema_name.lower(), table_name.lower()) in external_keys
            metadata: dict[str, str] = {}
            if is_external:
                for external in self.external_tables:
                    if (external.schema.lower(), external.name.lower()) == (
                        schema_name.lower(),
                        table_name.lower(),
                    ):
                        metadata = dict(external.metadata)
                        break
            objects.append(
                CatalogObject(
                    kind="external_table" if is_external else "table",
                    schema=schema_name,
                    name=table_name,
                    definition=render_create_table(schema_name, table_name, records),
                    metadata=metadata,
                )
            )
        return objects

    def objects(self) -> list[CatalogObject]:
        """Return every tracked object in a stable order."""
        tables = self.table_objects()
        described = {(obj.schema.lower(), obj.name.lower()) for obj in tables}
        orphan_externals = [
            obj
            for obj in self.external_tables
            if (obj.schema.lower(), obj.name.lower()) not in described
        ]
        return (
            tables
            + orphan_externals
            + list(self.views)
            + list(self.materialized_views)
            + list(self.functions)
        )


def render_create_table(schema: str, table: str, columns: Sequence[ColumnRecord]) -> str:
    """Render a ``CREATE TABLE`` statement for the given column records."""
    target = f"{quote_identifier(schema)}.{quote_identifier(table)}"
    if not columns:
        return f"CREATE TABLE {target} ();"
    rendered = ",\n".join(
        f"    {quote_identifier(record.column)} {normalize_data_type(record.data_type)}"
        for record in columns
    )
    return f"CREATE TABLE {target} (\n{rendered}\n);"


def render_create_view(obj: CatalogObject, *, materialized: bool = False) -> str:
    """Render a ``CREATE [MATERIALIZED] VIEW`` statement from a catalog definition."""
    target = f"{quote_identifier(obj.schema)}.{quote_identifier(obj.name)}"
    body = (obj.definition or "").strip().rstrip(";").strip()
    keyword = "CREATE MATERIALIZED VIEW" if materialized else "CREATE VIEW"
    return f"{keyword} {target} AS\n{body};"


def _is_parseable(statement: str, dialect: str) -> bool:
    """Check that sqlglot understands a statement instead of falling back to a raw Command."""
    if sqlglot is None or exp is None:
        return True
    try:
        parsed = sqlglot.parse_one(statement, read=dialect)
    except Exception:
        return False
    if parsed is None:
        return False
    return next(parsed.find_all(exp.Command), None) is None


class CatalogState:
    """Persisted ``{object_key: definition_hash}`` map enabling incremental dumps."""

    def __init__(self, path: str | Path = DEFAULT_STATE_PATH, hashes: dict[str, str] | None = None):
        self.path = Path(path)
        self.hashes: dict[str, str] = dict(hashes or {})

    @classmethod
    def load(cls, path: str | Path = DEFAULT_STATE_PATH) -> CatalogState:
        state_path = Path(path)
        if not state_path.exists():
            return cls(state_path)
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise GPCatalogError(f"Cannot read state file {state_path}: {exc}") from exc
        objects = payload.get("objects") if isinstance(payload, dict) else None
        if not isinstance(objects, dict):
            objects = payload if isinstance(payload, dict) else {}
        hashes = {str(key): str(value) for key, value in objects.items() if isinstance(value, str)}
        return cls(state_path, hashes)

    def diff(self, objects: Sequence[CatalogObject]) -> CatalogDiff:
        """Compare the current catalog against the persisted hashes."""
        diff = CatalogDiff()
        seen: set[str] = set()
        for obj in objects:
            seen.add(obj.key)
            if self.hashes.get(obj.key) == obj.definition_hash:
                diff.unchanged.append(obj.key)
            else:
                diff.changed.append(obj.key)
        diff.removed = sorted(key for key in self.hashes if key not in seen)
        return diff

    def apply(self, objects: Sequence[CatalogObject]) -> CatalogState:
        """Replace the tracked hashes with the current catalog snapshot."""
        self.hashes = {obj.key: obj.definition_hash for obj in objects}
        return self

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "objects": dict(sorted(self.hashes.items()))}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return self.path


class GPCatalogExtractor:
    """Read-only reader of GreenPlum catalog metadata.

    Credentials come from an explicit ``dsn``, from the environment (``GP_DSN``,
    ``GREENPLUM_DSN``, ``DATABASE_URL``) or from ``GP_HOST``/``GP_PORT``/``GP_DATABASE``/
    ``GP_USER``/``GP_PASSWORD``. A ``.env`` file is loaded when ``python-dotenv`` is
    available. DSNs are never logged verbatim.
    """

    def __init__(
        self,
        dsn: str | None = None,
        *,
        connection: Any = None,
        read_only: bool = True,
        exclude_schemas: Sequence[str] | None = None,
        state_path: str | Path = DEFAULT_STATE_PATH,
        dialect: str = "postgres",
        env_file: str | Path | None = None,
        load_env: bool = True,
    ):
        self.read_only = read_only
        self.dialect = dialect
        self.exclude_schemas: tuple[str, ...] = tuple(
            exclude_schemas if exclude_schemas is not None else DEFAULT_EXCLUDED_SCHEMAS
        )
        self.state = CatalogState.load(state_path)
        self.warnings: list[str] = []
        self._dsn = dsn
        self._connection = connection
        self._owns_connection = connection is None
        self._session_configured = False
        self._snapshot: CatalogSnapshot | None = None
        if load_env:
            self._load_env_file(env_file)

    # ------------------------------------------------------------------ connection

    @staticmethod
    def _load_env_file(env_file: str | Path | None) -> None:
        try:
            from dotenv import load_dotenv
        except Exception:  # pragma: no cover - python-dotenv is a hard dependency
            return
        if env_file is not None:
            load_dotenv(env_file)
        else:
            load_dotenv()

    def resolve_dsn(self) -> str:
        """Resolve the connection string without ever returning it to a log sink."""
        if self._dsn:
            return self._dsn
        for name in DSN_ENV_VARS:
            value = os.environ.get(name)
            if value:
                return value
        host = os.environ.get("GP_HOST")
        database = os.environ.get("GP_DATABASE") or os.environ.get("GP_DBNAME")
        if not host or not database:
            raise GPCatalogError(
                "No GreenPlum DSN configured. Pass --dsn, or set one of "
                f"{', '.join(DSN_ENV_VARS)}, or set GP_HOST/GP_DATABASE/GP_USER/GP_PASSWORD "
                "in the environment or .env file."
            )
        parts = [f"host={host}", f"dbname={database}"]
        port = os.environ.get("GP_PORT")
        if port:
            parts.append(f"port={port}")
        user = os.environ.get("GP_USER")
        if user:
            parts.append(f"user={user}")
        password = os.environ.get("GP_PASSWORD")
        if password:
            parts.append(f"password={password}")
        return " ".join(parts)

    @property
    def target(self) -> str:
        """Human-readable, credential-free description of the connection target."""
        try:
            return redact_dsn(self.resolve_dsn())
        except GPCatalogError:
            return "<unconfigured>"

    def connect(self) -> Any:
        """Return the (lazily created) read-only connection."""
        if self._connection is None:
            psycopg2 = _import_psycopg2()
            self._connection = psycopg2.connect(self.resolve_dsn())
            self._owns_connection = True
        if self.read_only and not self._session_configured:
            self._apply_read_only(self._connection)
            self._session_configured = True
        return self._connection

    def _apply_read_only(self, connection: Any) -> None:
        setter = getattr(connection, "set_session", None)
        if callable(setter):
            setter(readonly=True, autocommit=True)
            return
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")

    def close(self) -> None:
        """Close the connection when this extractor owns it."""
        if self._connection is not None and self._owns_connection:
            try:
                self._connection.close()
            except Exception as exc:  # pragma: no cover - close failures are not actionable
                logger.debug("Ignoring error while closing the catalog connection: %s", exc)
        self._connection = None
        self._session_configured = False

    def __enter__(self) -> GPCatalogExtractor:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # --------------------------------------------------------------------- queries

    def _rollback(self) -> None:
        rollback = getattr(self._connection, "rollback", None)
        if callable(rollback):
            try:
                rollback()
            except Exception as exc:  # pragma: no cover - best effort after a failed query
                logger.debug("Rollback after a failed catalog query did not succeed: %s", exc)

    def _fetch(self, sql: str, *, optional_label: str = "") -> list[tuple[Any, ...]]:
        """Run a read-only query, degrading to an empty result for optional objects."""
        connection = self.connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall() or []
        except Exception as exc:
            if optional_label and _is_missing_object_error(exc):
                self._rollback()
                self._warn(f"{optional_label} is not available in this catalog; skipping")
                return []
            raise
        return [tuple(row) for row in rows]

    def _warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def functions_sql(self) -> str:
        return FUNCTIONS_SQL_TEMPLATE.format(
            schema_filter=_schema_filter("n.nspname", self.exclude_schemas)
        )

    def views_sql(self) -> str:
        return VIEWS_SQL_TEMPLATE.format(
            schema_filter=_schema_filter("schemaname", self.exclude_schemas)
        )

    def materialized_views_sql(self) -> str:
        return MATVIEWS_SQL_TEMPLATE.format(
            schema_filter=_schema_filter("schemaname", self.exclude_schemas)
        )

    def columns_sql(self) -> str:
        return COLUMNS_SQL_TEMPLATE.format(
            schema_filter=_schema_filter("table_schema", self.exclude_schemas)
        )

    def external_tables_sql(self) -> str:
        return EXTERNAL_TABLES_SQL_TEMPLATE.format(
            schema_filter=_schema_filter("n.nspname", self.exclude_schemas)
        )

    def fetch_functions(self) -> list[CatalogObject]:
        """Fetch ``plpgsql``/``sql`` function definitions via ``pg_get_functiondef``."""
        objects: list[CatalogObject] = []
        for row in self._fetch(self.functions_sql()):
            schema_name, function_name, definition = str(row[0]), str(row[1]), str(row[2] or "")
            objects.append(
                CatalogObject(
                    kind="function",
                    schema=schema_name,
                    name=function_name,
                    definition=definition.strip(),
                )
            )
        return objects

    def fetch_views(self) -> list[CatalogObject]:
        """Fetch view definitions from ``pg_views``."""
        return [
            CatalogObject(kind="view", schema=str(row[0]), name=str(row[1]), definition=str(row[2] or ""))
            for row in self._fetch(self.views_sql())
        ]

    def fetch_materialized_views(self) -> list[CatalogObject]:
        """Fetch materialized view definitions from ``pg_matviews`` when present."""
        return [
            CatalogObject(kind="matview", schema=str(row[0]), name=str(row[1]), definition=str(row[2] or ""))
            for row in self._fetch(self.materialized_views_sql(), optional_label="pg_matviews")
        ]

    def fetch_columns(self) -> list[ColumnRecord]:
        """Fetch column metadata from ``information_schema.columns``."""
        records: list[ColumnRecord] = []
        for row in self._fetch(self.columns_sql()):
            try:
                position = int(row[4]) if len(row) > 4 and row[4] is not None else 0
            except (TypeError, ValueError):
                position = 0
            records.append(
                ColumnRecord(
                    schema=str(row[0]),
                    table=str(row[1]),
                    column=str(row[2]),
                    data_type=str(row[3] or ""),
                    ordinal_position=position,
                )
            )
        return records

    def fetch_external_tables(self) -> list[CatalogObject]:
        """Fetch GreenPlum external tables (source nodes); empty on vanilla PostgreSQL."""
        objects: list[CatalogObject] = []
        for row in self._fetch(self.external_tables_sql(), optional_label="pg_exttable"):
            schema_name, table_name = str(row[0]), str(row[1])
            location = str(row[2] or "") if len(row) > 2 else ""
            fmt = str(row[3] or "") if len(row) > 3 else ""
            objects.append(
                CatalogObject(
                    kind="external_table",
                    schema=schema_name,
                    name=table_name,
                    definition=f"EXTERNAL TABLE {schema_name}.{table_name} LOCATION {location} FORMAT {fmt}",
                    metadata={"location": location, "format": fmt},
                )
            )
        return objects

    def snapshot(self, *, refresh: bool = False) -> CatalogSnapshot:
        """Fetch (and cache) the full catalog snapshot."""
        if self._snapshot is not None and not refresh:
            return self._snapshot
        external_tables = self.fetch_external_tables()
        snapshot = CatalogSnapshot(
            functions=self.fetch_functions(),
            views=self.fetch_views(),
            materialized_views=self.fetch_materialized_views(),
            external_tables=external_tables,
            columns=self.fetch_columns(),
        )
        snapshot.warnings = list(self.warnings)
        self._snapshot = snapshot
        return snapshot

    # --------------------------------------------------------------------- outputs

    def iter_functions(self, *, only_changed: bool = False) -> Iterator[CatalogObject]:
        """Yield function definitions, optionally limited to objects changed since last run."""
        functions = self.snapshot().functions
        if not only_changed:
            yield from functions
            return
        for obj in functions:
            if self.state.hashes.get(obj.key) != obj.definition_hash:
                yield obj

    def changed_objects(self) -> CatalogDiff:
        """Diff the current catalog against the persisted state file."""
        return self.state.diff(self.snapshot().objects())

    def dump_ddl_text(self, *, include_functions: bool = False) -> str:
        """Render DDL text in the format ``SchemaRegistry.load_ddl`` parses."""
        snapshot = self.snapshot()
        blocks: list[str] = ["-- Generated by llm4lineage GPCatalogExtractor (read-only catalog dump)"]

        table_objects = snapshot.table_objects()
        for obj in table_objects:
            if obj.kind == "external_table":
                location = obj.metadata.get("location", "")
                blocks.append(
                    f"-- EXTERNAL TABLE (source) {obj.qualified_name}"
                    + (f" LOCATION {location}" if location else "")
                )
            blocks.append(obj.definition)

        described = {(obj.schema.lower(), obj.name.lower()) for obj in table_objects}
        for obj in snapshot.external_tables:
            if (obj.schema.lower(), obj.name.lower()) not in described:
                blocks.append(f"-- EXTERNAL TABLE (source, columns unavailable) {obj.qualified_name}")

        for obj in snapshot.views:
            blocks.append(self._safe_statement(render_create_view(obj), obj))
        for obj in snapshot.materialized_views:
            blocks.append(self._safe_statement(render_create_view(obj, materialized=True), obj))

        if include_functions:
            for obj in snapshot.functions:
                blocks.append(self._safe_statement(self._function_statement(obj), obj))

        return "\n\n".join(block for block in blocks if block) + "\n"

    def _safe_statement(self, statement: str, obj: CatalogObject) -> str:
        """Comment out statements sqlglot cannot parse so the dump stays loadable."""
        if _is_parseable(statement, self.dialect):
            return statement
        self._warn(f"Definition of {obj.kind} {obj.qualified_name} is not parseable; emitted as comment")
        commented = "\n".join(f"-- {line}" for line in statement.splitlines())
        return f"-- UNPARSEABLE {obj.kind} {obj.qualified_name}\n{commented}"

    @staticmethod
    def _function_statement(obj: CatalogObject) -> str:
        definition = (obj.definition or "").strip()
        if not definition:
            return ""
        return definition if definition.endswith(";") else definition + ";"

    def dump_functions_sql(self) -> str:
        """Render every function definition as one SQL script."""
        snapshot = self.snapshot()
        blocks = ["-- Generated by llm4lineage GPCatalogExtractor (function definitions)"]
        for obj in snapshot.functions:
            statement = self._function_statement(obj)
            if statement:
                blocks.append(f"-- {obj.kind}: {obj.qualified_name}\n{statement}")
        return "\n\n".join(blocks) + "\n"

    def dump_csv(self, out_dir: str | Path) -> dict[str, str]:
        """Write columns/views/functions/external-table CSVs and return their paths."""
        target = Path(out_dir)
        target.mkdir(parents=True, exist_ok=True)
        snapshot = self.snapshot()
        written: dict[str, str] = {}

        columns_path = target / "columns.csv"
        with columns_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["schema", "table", "column", "type", "ordinal_position"])
            for record in snapshot.columns:
                writer.writerow(
                    [
                        record.schema,
                        record.table,
                        record.column,
                        record.data_type,
                        record.ordinal_position,
                    ]
                )
        written["columns"] = str(columns_path)

        views_path = target / "views.csv"
        with views_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["schema", "table_name", "kind", "view_def"])
            for obj in list(snapshot.views) + list(snapshot.materialized_views):
                writer.writerow([obj.schema, obj.name, obj.kind, obj.definition])
        written["views"] = str(views_path)

        functions_path = target / "functions.csv"
        with functions_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["schema", "function", "definition_hash", "definition"])
            for obj in snapshot.functions:
                writer.writerow([obj.schema, obj.name, obj.definition_hash, obj.definition])
        written["functions"] = str(functions_path)

        external_path = target / "external_tables.csv"
        with external_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["schema", "table", "location", "format"])
            for obj in snapshot.external_tables:
                writer.writerow(
                    [obj.schema, obj.name, obj.metadata.get("location", ""), obj.metadata.get("format", "")]
                )
        written["external_tables"] = str(external_path)

        return written

    def dump(
        self,
        out_dir: str | Path,
        *,
        incremental: bool = True,
        include_functions_in_ddl: bool = False,
    ) -> DumpReport:
        """Write the dump, skipping all work when nothing changed since the last run."""
        target = Path(out_dir)
        snapshot = self.snapshot()
        objects = snapshot.objects()
        diff = self.state.diff(objects)
        report = DumpReport(
            out_dir=str(target),
            changed=list(diff.changed),
            removed=list(diff.removed),
            unchanged=len(diff.unchanged),
            state_path=str(self.state.path),
        )

        if incremental and not diff.has_changes:
            report.skipped = True
            report.warnings = list(self.warnings)
            return report

        target.mkdir(parents=True, exist_ok=True)
        ddl_path = target / "ddl.sql"
        ddl_path.write_text(
            self.dump_ddl_text(include_functions=include_functions_in_ddl), encoding="utf-8"
        )
        functions_path = target / "functions.sql"
        functions_path.write_text(self.dump_functions_sql(), encoding="utf-8")
        changed_path = target / "changed_objects.json"
        changed_path.write_text(json.dumps(diff.to_dict(), indent=2) + "\n", encoding="utf-8")

        files = [str(ddl_path), str(functions_path), str(changed_path)]
        files.extend(self.dump_csv(target).values())

        self.state.apply(objects).save()
        report.files = files
        report.warnings = list(self.warnings)
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump GreenPlum catalog DDL (read-only)")
    parser.add_argument("--dsn", default="", help="Connection string; falls back to env/.env")
    parser.add_argument("--out", default="data/gp_dump", help="Output directory for the dump")
    parser.add_argument("--state", default=DEFAULT_STATE_PATH, help="Path to the incremental state file")
    parser.add_argument(
        "--read-only",
        dest="read_only",
        action="store_true",
        default=True,
        help="Open a read-only session (default)",
    )
    parser.add_argument(
        "--no-read-only",
        dest="read_only",
        action="store_false",
        help="Allow a writable session (not recommended)",
    )
    parser.add_argument("--full", action="store_true", help="Ignore the state file and dump everything")
    parser.add_argument(
        "--exclude-schema",
        action="append",
        default=None,
        help="Schema to exclude (repeatable); defaults to system schemas",
    )
    parser.add_argument(
        "--include-functions-in-ddl",
        action="store_true",
        help="Append function definitions to ddl.sql (off by default)",
    )
    parser.add_argument("--env-file", default="", help="Optional path to a .env file")
    args = parser.parse_args(argv)

    excluded = tuple(args.exclude_schema) if args.exclude_schema else None
    extractor = GPCatalogExtractor(
        dsn=args.dsn or None,
        read_only=args.read_only,
        exclude_schemas=excluded,
        state_path=args.state,
        env_file=args.env_file or None,
    )
    try:
        report = extractor.dump(
            args.out,
            incremental=not args.full,
            include_functions_in_ddl=args.include_functions_in_ddl,
        )
    except GPCatalogError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - surfaces driver errors without the DSN
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        extractor.close()

    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
