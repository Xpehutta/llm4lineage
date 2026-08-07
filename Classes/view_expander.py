"""Expand view references into their underlying SELECT bodies before qualification."""

from __future__ import annotations

from typing import Any

try:
    from sqlglot import exp
except Exception:  # pragma: no cover
    exp = None

from Classes.schema_registry import SchemaRegistry


class ViewExpander:
    """Inline CREATE VIEW bodies from a schema registry into a parsed AST."""

    def __init__(self, dialect: str = "postgres"):
        self.dialect = dialect

    def expand(self, expression: Any, registry: SchemaRegistry, _stack: set[tuple[str, str]] | None = None) -> Any:
        if expression is None or exp is None or registry is None:
            return expression

        stack = set(_stack or ())
        expanded = expression.copy()
        for select in expanded.find_all(exp.Select):
            self._expand_select_sources(select, registry, stack)
        return expanded

    def _expand_select_sources(
        self,
        select: Any,
        registry: SchemaRegistry,
        stack: set[tuple[str, str]],
    ) -> None:
        from_clause = select.find(exp.From)
        if from_clause is None:
            return

        if isinstance(from_clause.this, exp.Table):
            from_clause.set("this", self._maybe_inline_table(from_clause.this, registry, stack))

        for join in select.args.get("joins") or []:
            if isinstance(join.this, exp.Table):
                join.set("this", self._maybe_inline_table(join.this, registry, stack))

    def _maybe_inline_table(
        self,
        table: Any,
        registry: SchemaRegistry,
        stack: set[tuple[str, str]],
    ) -> Any:
        schema_key, table_key = registry.table_keys_from_expression(table)
        view_key = (schema_key, table_key)
        if not registry.is_view(schema_key, table_key):
            return table
        if view_key in stack:
            return table

        view_select = registry.get_view_select(schema_key, table_key)
        if view_select is None:
            return table

        alias_name = table.alias_or_name or table_key
        inlined = self.expand(view_select.copy(), registry, stack | {view_key})
        return exp.Subquery(
            this=inlined,
            alias=exp.TableAlias(this=exp.to_identifier(alias_name)),
        )
