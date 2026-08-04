"""Deterministic table-level lineage extraction via sqlglot."""

from __future__ import annotations

from typing import Any, Dict, Set

try:
    import sqlglot
    from sqlglot import exp
except Exception:  # pragma: no cover
    sqlglot = None
    exp = None


def _table_name(node: Any, dialect: str) -> str:
    if node is None:
        return ""
    if hasattr(node, "sql"):
        return str(node.sql(dialect=dialect)).strip().strip('"')
    return str(node).strip()


def _qualified_table_name(node: Any, dialect: str) -> str:
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


def _collect_cte_names(expression: Any) -> Set[str]:
    """Collect CTE alias names defined in WITH clauses (any nesting level)."""
    if expression is None or exp is None:
        return set()
    names: Set[str] = set()
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


def _collect_physical_tables(expression: Any, dialect: str) -> Set[str]:
    if expression is None or exp is None:
        return set()
    cte_names = _collect_cte_names(expression)
    tables: Set[str] = set()
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


def extract_table_lineage(sql: str, dialect: str = "postgres") -> Dict[str, Any]:
    """Extract target + physical source tables from INSERT/MERGE/UPDATE/SELECT."""
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
        sources = _collect_physical_tables(tree.expression, dialect)
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
    elif isinstance(tree, exp.Create) and str(getattr(tree, "kind", "")).upper() == "TABLE":
        statement_type = "create_table_as"
        target = _qualified_table_name(tree.this, dialect)
        sources = _collect_physical_tables(tree.expression, dialect)
    else:
        sources = _collect_physical_tables(tree, dialect)

    return {
        "target": target,
        "sources": sorted(sources),
        "statement_type": statement_type,
        "parser_used": True,
    }
