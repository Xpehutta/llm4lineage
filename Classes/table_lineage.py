"""Deterministic table-level lineage extraction via sqlglot."""

from __future__ import annotations

import logging
from typing import Any

try:
    import sqlglot
    from sqlglot import exp
except Exception:  # pragma: no cover
    sqlglot = None
    exp = None

logger = logging.getLogger(__name__)


def _table_name(node: Any, dialect: str) -> str:
    if node is None:
        return ""
    if hasattr(node, "sql"):
        return str(node.sql(dialect=dialect)).strip().strip('"')
    return str(node).strip()


def _qualified_table_name(node: Any, dialect: str) -> str:
    # `INSERT INTO t (a, b)` wraps the table in a Schema node holding the
    # column list; unwrap it so the target is `t` and not `t (a, b)`.
    if exp is not None and isinstance(node, exp.Schema) and node.this is not None:
        node = node.this
    if exp is not None and isinstance(node, exp.Table):
        parts = [part for part in (node.catalog, node.db, node.name) if part]
        if parts:
            return ".".join(str(part) for part in parts).lower()
    return _table_name(node, dialect).lower()


def _cte_alias_name(cte: Any) -> str:
    alias = getattr(cte, "alias", None)
    if alias is None:
        return ""
    if isinstance(alias, str):
        return alias.strip().strip('"').lower()
    if hasattr(alias, "name") and alias.name:
        return str(alias.name).strip().strip('"').lower()
    if hasattr(alias, "this"):
        return _table_name(alias.this, dialect="postgres").strip().strip('"').lower()
    return str(alias).strip().strip('"').lower()


def _collect_cte_names(expression: Any) -> set[str]:
    """Collect CTE alias names defined in WITH clauses (any nesting level)."""
    if expression is None or exp is None:
        return set()
    names: set[str] = set()
    for with_node in expression.find_all(exp.With):
        for cte in with_node.expressions or []:
            name = _cte_alias_name(cte)
            if name:
                names.add(name)
    return names


def _table_reference_name(table: Any, dialect: str) -> str:
    """Return the referenced relation name (CTE/subquery alias or physical table)."""
    if exp is not None and isinstance(table, exp.Table):
        parts = [part for part in (table.catalog, table.db, table.name) if part]
        if parts:
            return ".".join(str(part) for part in parts).lower()
    return _table_name(table, dialect).lower()


def _collect_physical_tables(expression: Any, dialect: str, cte_scope: Any = None) -> set[str]:
    if expression is None or exp is None:
        return set()
    cte_names = _collect_cte_names(expression) | _collect_cte_names(cte_scope)
    tables: set[str] = set()
    for table in expression.find_all(exp.Table):
        qualified = _qualified_table_name(table, dialect)
        if not qualified or qualified.startswith("("):
            continue
        reference = _table_reference_name(table, dialect)
        bare = reference.split(".")[-1]
        if bare in cte_names or reference in cte_names:
            continue
        tables.add(qualified)
    return tables


def _collect_source_tables(statement: Any, body: Any, dialect: str) -> set[str]:
    """Physical tables feeding *body*, including any WITH attached to *statement*.

    `WITH x AS (...) INSERT INTO t SELECT * FROM x` parses with the WITH clause
    hanging off the INSERT rather than off the SELECT, so the CTE bodies have to
    be scanned separately or their base tables are lost and `x` itself is
    mistaken for a physical table.
    """
    tables = _collect_physical_tables(body, dialect, cte_scope=statement)
    if exp is not None and statement is not None:
        for with_node in statement.find_all(exp.With):
            tables |= _collect_physical_tables(with_node, dialect, cte_scope=statement)
    return tables


def _has_create_property(tree: Any, property_type: type) -> bool:
    props = getattr(tree, "args", {}).get("properties") if tree is not None else None
    if props is None:
        return False
    return any(isinstance(item, property_type) for item in (props.expressions or []))


def classify_create(tree: Any) -> str:
    """Return statement_type for a sqlglot ``Create`` node."""
    if exp is None or not isinstance(tree, exp.Create):
        return "unknown"
    kind = str(getattr(tree, "kind", "") or "").upper()
    if kind == "VIEW":
        if _has_create_property(tree, exp.MaterializedProperty):
            return "create_materialized_view"
        return "create_view"
    if kind == "TABLE":
        if tree.expression is not None:
            return "create_table_as"
        return "create_table"
    return "create"


def create_has_query_body(tree: Any) -> bool:
    """True when CREATE … AS <query> (CTAS / VIEW / MATVIEW)."""
    if exp is None or tree is None or tree.expression is None:
        return False
    body = tree.expression
    if isinstance(body, (exp.Query, exp.Select, exp.Union, exp.Subquery)):
        return True
    return body.find(exp.Query) is not None


def extract_create_ddl(sql: str, dialect: str = "postgres") -> str:
    """Extract CREATE TABLE / VIEW / MATERIALIZED VIEW statements from a script.

    Useful for auto-feeding SchemaRegistry from the same SQL file that is being
    analyzed (DDL + INSERT in one upload).
    """
    if sqlglot is None or exp is None or not sql.strip():
        return ""
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except Exception:
        return ""
    parts: list[str] = []
    for statement in statements or []:
        if statement is None or not isinstance(statement, exp.Create):
            continue
        kind = str(getattr(statement, "kind", "") or "").upper()
        if kind not in {"TABLE", "VIEW"}:
            continue
        try:
            parts.append(statement.sql(dialect=dialect))
        except Exception as exc:
            logger.debug("Skipping CREATE statement that failed to render: %s", exc)
            continue
    return ";\n".join(parts)


def extract_table_lineage(sql: str, dialect: str = "postgres") -> dict[str, Any]:
    """Extract target + physical source tables from INSERT/MERGE/UPDATE/CREATE/SELECT."""
    if sqlglot is None or exp is None:
        return {"target": "", "sources": [], "statement_type": "unknown", "parser_used": False}

    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception as exc:
        return {"target": "", "sources": [], "statement_type": "unknown", "parser_used": False, "error": str(exc)}

    statement_type = "select"
    target = ""

    if isinstance(tree, exp.Insert):
        statement_type = "insert"
        target = _qualified_table_name(tree.this, dialect)
        sources = _collect_source_tables(tree, tree.expression, dialect)
    elif isinstance(tree, exp.Merge):
        statement_type = "merge"
        target = _qualified_table_name(tree.this, dialect)
        sources = _collect_physical_tables(tree, dialect)
        sources.discard(target)
    elif isinstance(tree, exp.Update):
        statement_type = "update"
        target = _qualified_table_name(tree.this, dialect)
        sources = _collect_physical_tables(tree, dialect)
        sources.discard(target)
    elif isinstance(tree, exp.Create):
        statement_type = classify_create(tree)
        target = _qualified_table_name(tree.this, dialect)
        if create_has_query_body(tree):
            sources = _collect_source_tables(tree, tree.expression, dialect)
            sources.discard(target)
        else:
            sources = set()
    else:
        sources = _collect_physical_tables(tree, dialect)

    return {
        "target": target,
        "sources": sorted(sources),
        "statement_type": statement_type,
        "parser_used": True,
    }
