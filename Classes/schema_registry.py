"""DDL-backed schema registry for sqlglot column qualification."""

from __future__ import annotations

import csv
import io
import re
from typing import Any

try:
    import sqlglot
    from sqlglot import exp
except Exception:  # pragma: no cover
    sqlglot = None
    exp = None

SchemaDict = dict[str, dict[str, dict[str, str]]]
ViewKey = tuple[str, str]


def _normalize_table_key(schema: str | None, table: str) -> tuple[str, str]:
    table = (table or "").strip().strip('"')
    if "." in table:
        parts = [part.strip().strip('"') for part in table.split(".") if part.strip()]
        if len(parts) >= 2:
            return parts[-2].lower(), parts[-1].lower()
    schema_name = (schema or "public").strip().strip('"').lower()
    return schema_name, table.lower()


class DDLParser:
    """Parse CREATE TABLE / CREATE VIEW DDL into a nested schema dict."""

    def __init__(self, dialect: str = "postgres"):
        self.dialect = dialect

    def parse_text(self, ddl_text: str) -> SchemaDict:
        schema, _views = self.parse_registry(ddl_text)
        return schema

    def parse_registry(self, ddl_text: str) -> tuple[SchemaDict, dict[ViewKey, Any]]:
        schema: SchemaDict = {}
        views: dict[ViewKey, Any] = {}
        if not ddl_text or sqlglot is None or exp is None:
            return schema, views

        for statement in sqlglot.parse(ddl_text, read=self.dialect):
            if statement is None:
                continue
            if isinstance(statement, exp.Create):
                self._ingest_create(statement, schema, views)
        return schema, views

    def _ingest_create(
        self,
        statement: exp.Create,
        schema: SchemaDict,
        views: dict[ViewKey, Any] | None = None,
    ) -> None:
        target = statement.this
        if target is None:
            return

        if isinstance(target, exp.Schema):
            table_expr = target.this
            column_container = target
        else:
            table_expr = target
            column_container = statement.expression

        if table_expr is None:
            return

        schema_name = (
            getattr(table_expr, "db", None)
            or getattr(table_expr, "catalog", None)
            or "public"
        )
        table_name = getattr(table_expr, "name", None) or str(table_expr)
        schema_key, table_key = _normalize_table_key(str(schema_name), str(table_name))

        columns: dict[str, str] = {}
        if isinstance(column_container, exp.Schema):
            for column_def in column_container.expressions:
                if isinstance(column_def, exp.ColumnDef):
                    col_name = column_def.name
                    col_type = column_def.args.get("kind")
                    type_sql = col_type.sql(dialect=self.dialect) if col_type is not None else "UNKNOWN"
                    columns[col_name.lower()] = type_sql
        elif isinstance(column_container, exp.Select):
            for select_expr in column_container.expressions:
                alias = select_expr.alias_or_name
                if alias:
                    columns[alias.lower()] = "UNKNOWN"

        if columns:
            schema.setdefault(schema_key, {}).setdefault(table_key, {}).update(columns)

        is_view = str(getattr(statement, "kind", "") or "").upper() == "VIEW"
        if is_view and views is not None and isinstance(statement.expression, exp.Select):
            views[(schema_key, table_key)] = statement.expression


class SchemaRegistry:
    """In-memory schema catalog with sqlglot-compatible export."""

    def __init__(self, dialect: str = "postgres"):
        self.dialect = dialect
        self._schema: SchemaDict = {}
        self._views: dict[ViewKey, Any] = {}
        self._ddl_parser = DDLParser(dialect=dialect)

    @property
    def views(self) -> dict[ViewKey, Any]:
        return self._views

    @property
    def tables(self) -> SchemaDict:
        return self._schema

    def load_ddl(self, ddl_text: str) -> SchemaRegistry:
        schema, views = self._ddl_parser.parse_registry(ddl_text)
        self.merge(schema)
        for key, view_select in views.items():
            self._views[key] = view_select
        return self

    def load_csv(self, csv_text: str, *, delimiter: str = ",") -> SchemaRegistry:
        reader = csv.DictReader(io.StringIO(csv_text), delimiter=delimiter)
        for row in reader:
            schema_name = (row.get("schema") or row.get("schema_name") or "public").strip()
            table_name = (row.get("table") or row.get("table_name") or "").strip()
            column_name = (row.get("column") or row.get("column_name") or "").strip()
            column_type = (row.get("type") or row.get("data_type") or "UNKNOWN").strip()
            if not table_name or not column_name:
                continue
            schema_key, table_key = _normalize_table_key(schema_name, table_name)
            self._schema.setdefault(schema_key, {}).setdefault(table_key, {})[
                column_name.lower()
            ] = column_type
        return self

    def merge(self, other: SchemaDict | SchemaRegistry) -> SchemaRegistry:
        payload = other.tables if isinstance(other, SchemaRegistry) else other
        for schema_name, tables in (payload or {}).items():
            for table_name, columns in tables.items():
                bucket = self._schema.setdefault(schema_name, {}).setdefault(table_name, {})
                bucket.update(columns)
        return self

    def has_tables(self) -> bool:
        return bool(self._schema)

    def is_view(self, schema: str, table: str) -> bool:
        schema_key, table_key = _normalize_table_key(schema, table)
        return (schema_key, table_key) in self._views

    def get_view_select(self, schema: str, table: str) -> Any:
        schema_key, table_key = _normalize_table_key(schema, table)
        return self._views.get((schema_key, table_key))

    @staticmethod
    def table_keys_from_expression(table_expr: Any) -> tuple[str, str]:
        if table_expr is None:
            return "public", ""
        schema_name = (
            getattr(table_expr, "db", None)
            or getattr(table_expr, "catalog", None)
            or "public"
        )
        table_name = getattr(table_expr, "name", None) or str(table_expr)
        return _normalize_table_key(str(schema_name), str(table_name))

    def table_columns(self, schema: str, table: str) -> dict[str, str]:
        schema_key, table_key = _normalize_table_key(schema, table)
        return dict(self._schema.get(schema_key, {}).get(table_key, {}))

    def to_sqlglot_schema(self) -> dict[str, dict[str, dict[str, str]]]:
        """Return a nested dict usable by sqlglot optimizer qualify functions."""
        return self._schema

    def qualify_expression(self, expression: Any) -> Any:
        """Qualify columns in a parsed sqlglot expression when schema is available."""
        if expression is None or not self.has_tables():
            return expression
        try:
            from sqlglot.optimizer.qualify_columns import qualify_columns

            return qualify_columns(expression.copy(), schema=self.to_sqlglot_schema())
        except Exception:
            return expression

    @classmethod
    def from_ddl_file(cls, path: str, *, dialect: str = "postgres") -> SchemaRegistry:
        text = open(path, encoding="utf-8").read()
        return cls(dialect=dialect).load_ddl(text)

    def load_sql_corpus(self, sql_text: str, *, chunk_size: int = 100) -> SchemaRegistry:
        """Infer output column shapes from a batch of INSERT/SELECT statements."""
        statements = [part.strip() for part in re.split(r";\s*", sql_text or "") if part.strip()]
        for index in range(0, len(statements), chunk_size):
            for statement in statements[index : index + chunk_size]:
                self.infer_from_sql(statement)
        return self

    def split_statements(self, ddl_text: str) -> list[str]:
        if not ddl_text:
            return []
        if sqlglot is not None:
            return [stmt.sql(dialect=self.dialect) for stmt in sqlglot.parse(ddl_text, read=self.dialect) if stmt]
        return [part.strip() for part in re.split(r";\s*", ddl_text) if part.strip()]

    def load_ddl_chunked(self, ddl_text: str, *, chunk_size: int = 200) -> SchemaRegistry:
        """Incrementally load large DDL batches statement-by-statement."""
        statements = self.split_statements(ddl_text)
        if not statements:
            return self
        if len(statements) <= chunk_size:
            return self.load_ddl(ddl_text)

        parser = self._ddl_parser
        for index in range(0, len(statements), chunk_size):
            batch = ";\n".join(statements[index : index + chunk_size])
            schema, views = parser.parse_registry(batch)
            self.merge(schema)
            for key, view_select in views.items():
                self._views[key] = view_select
        return self

    def infer_from_sql(self, sql: str) -> SchemaRegistry:
        """Best-effort column inference from INSERT/SELECT (deterministic, no LLM)."""
        if sqlglot is None or exp is None or not sql.strip():
            return self

        try:
            tree = sqlglot.parse_one(sql, read=self.dialect)
        except Exception:
            return self

        select_node = tree.find(exp.Select) if hasattr(tree, "find") else None
        if select_node is None:
            return self

        inferred: SchemaDict = {"inferred": {"query_output": {}}}
        for expression in select_node.expressions or []:
            alias = expression.alias_or_name
            if alias:
                inferred["inferred"]["query_output"][alias.lower()] = "UNKNOWN"
        if inferred["inferred"]["query_output"]:
            self.merge(inferred)
        return self
