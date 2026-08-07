"""Extract column-level lineage from a parsed SQL AST."""

import logging
from typing import Any

from sqlglot import exp
from sqlglot import lineage as sqlglot_lineage
from sqlglot.lineage import Node

from Classes.pipeline.exceptions import LineageExtractionError

logger = logging.getLogger(__name__)


class ColumnLineageExtractor:
    """Extract column-level lineage using sqlglot's built-in lineage module."""

    def __init__(
        self,
        dialect: str = "postgres",
        include_intermediate: bool = False,
        schema_catalog: dict[str, list[str]] | None = None,
    ):
        self.dialect = dialect
        self.include_intermediate = include_intermediate
        self.schema_catalog = schema_catalog
        # Note: include_intermediate is reserved for subclass extensions.
        # The base class ignores it.

    def extract(self, tree: exp.Expression) -> list[dict[str, Any]]:
        """Return a list of lineage records, one per output column.

        Raises:
            LineageExtractionError: If no SELECT is found or extraction fails.
        """
        try:
            if self._find_outermost_select(tree) is None:
                raise LineageExtractionError("No SELECT statement found in AST.")

            lineage_nodes = sqlglot_lineage.lineage(
                column=None,
                sql=tree,
                schema=self._to_sqlglot_schema(self.schema_catalog),
                dialect=self.dialect,
            )

            if not lineage_nodes:
                raise LineageExtractionError("No output columns found for lineage extraction.")

            records: list[dict[str, Any]] = []
            for target_name, node in lineage_nodes.items():
                records.append(self._node_to_record(target_name, node, tree))

            logger.debug("Extracted lineage for %d output columns.", len(records))
            return records

        except LineageExtractionError:
            raise
        except Exception as exc:
            raise LineageExtractionError(
                f"Lineage extraction failed: {exc}"
            ) from exc

    def _node_to_record(
        self,
        target_name: str,
        node: Node,
        tree: exp.Expression,
    ) -> dict[str, Any]:
        if target_name == "*":
            return {
                "target_column": "*",
                "source_columns": [],
                "expression": "*",
                "used_tables": self._extract_tables_from_tree(tree),
            }

        source_columns = self._collect_leaf_sources(node)
        expression = self._node_expression_sql(node)
        union_branches = self._collect_union_branches(node)

        return {
            "target_column": target_name,
            "source_columns": source_columns,
            "expression": expression,
            "union_branches": union_branches,
            "literal_values": self._literal_values_from_branches(union_branches),
            "used_tables": sorted(
                {ref["table"] for ref in source_columns if ref.get("table")}
            ),
        }

    @staticmethod
    def _to_sqlglot_schema(
        schema_catalog: dict[str, list[str]] | None,
    ) -> dict[str, dict[str, str]] | None:
        if not schema_catalog:
            return None
        return {
            table: {column: "UNKNOWN" for column in columns}
            for table, columns in schema_catalog.items()
        }

    @staticmethod
    def _is_positional_union_name(name: str) -> bool:
        return bool(name) and name.isdigit()

    @staticmethod
    def _column_refs_from_expression(expression: Any) -> list[dict[str, str | None]]:
        if not isinstance(expression, exp.Expression):
            return []

        refs: list[dict[str, str | None]] = []
        seen: set[tuple[str | None, str]] = set()
        for col in expression.find_all(exp.Column):
            table = col.table or None
            column = col.name or ""
            if not column:
                continue
            key = (table, column)
            if key in seen:
                continue
            seen.add(key)
            refs.append({"table": table, "column": column})
        return refs

    @staticmethod
    def _is_constant_expression(expression: Any) -> bool:
        if not isinstance(expression, exp.Expression):
            return False

        root = expression.this if isinstance(expression, exp.Alias) else expression
        if isinstance(root, (exp.Literal, exp.Null)):
            return True
        if isinstance(root, exp.Cast):
            return ColumnLineageExtractor._is_constant_expression(root.this)
        return False

    @staticmethod
    def _refs_from_leaf(node: Node) -> list[dict[str, str | None]]:
        name = node.name or ""

        if ColumnLineageExtractor._is_positional_union_name(name):
            expr_refs = ColumnLineageExtractor._column_refs_from_expression(node.expression)
            if expr_refs:
                return expr_refs
            if ColumnLineageExtractor._is_constant_expression(node.expression):
                return []
            return []

        if name:
            return [ColumnLineageExtractor._parse_source_ref(name)]

        return ColumnLineageExtractor._column_refs_from_expression(node.expression)

    @staticmethod
    def _ast_position(expression: Any) -> tuple[tuple[str, int], ...]:
        """Locate ``expression`` inside its tree as a path of (arg, index) steps.

        Sorting on this reproduces the order the nodes appear in the SQL text.
        """
        if not isinstance(expression, exp.Expression):
            return ()

        path: list[tuple[str, int]] = []
        current: Any = expression
        while current is not None and current.parent is not None:
            path.append((current.arg_key or "", current.index or 0))
            current = current.parent
        path.reverse()
        return tuple(path)

    @staticmethod
    def _ordered_leaves(node: Node) -> list[Node]:
        """Return the lineage leaves below ``node`` in a reproducible order.

        sqlglot gathers a column's sources through ``set(...)`` of expressions
        whose hash is salted by PYTHONHASHSEED, so the order it reports differs
        between interpreter runs. Anything derived from leaf position - notably
        ``branch_index`` and the node ids built from it - would otherwise change
        from run to run. Ordering by position in the AST restores the order the
        branches actually have in the SQL.
        """
        leaves = [current for current in node.walk() if not current.downstream]
        return sorted(
            leaves,
            key=lambda leaf: (
                ColumnLineageExtractor._ast_position(leaf.expression),
                str(leaf.name or ""),
                str(leaf.source_name or ""),
            ),
        )

    @staticmethod
    def _collect_leaf_sources(node: Node) -> list[dict[str, str | None]]:
        refs: list[dict[str, str | None]] = []
        seen: set[tuple[str | None, str | None]] = set()

        for current in ColumnLineageExtractor._ordered_leaves(node):
            for ref in ColumnLineageExtractor._refs_from_leaf(current):
                key = (ref["table"], ref["column"])
                if key in seen:
                    continue
                seen.add(key)
                refs.append(ref)

        if not refs:
            for ref in ColumnLineageExtractor._column_refs_from_expression(node.expression):
                key = (ref["table"], ref["column"])
                if key in seen:
                    continue
                seen.add(key)
                refs.append(ref)

        return refs

    @staticmethod
    def _parse_source_ref(name: str) -> dict[str, str | None]:
        table, _, column = name.rpartition(".")
        if table:
            return {"table": table, "column": column}
        return {"table": None, "column": name}

    @staticmethod
    def _literal_values_from_branches(
        union_branches: list[dict[str, Any]],
    ) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for branch in union_branches:
            if branch.get("kind") != "literal":
                continue
            expression = branch.get("expression")
            if not expression or expression in seen:
                continue
            seen.add(expression)
            values.append(expression)
        return values

    def _format_literal_expression(self, expression: Any) -> str:
        if not isinstance(expression, exp.Expression):
            return str(expression)

        if isinstance(expression, exp.Alias):
            inner = self._format_literal_core(expression.this)
            alias = expression.alias
            return f"{inner} AS {alias}"
        return self._format_literal_core(expression)

    def _format_literal_core(self, expression: Any) -> str:
        if isinstance(expression, exp.Cast) and self.dialect == "postgres":
            type_sql = expression.to.sql(dialect=self.dialect).lower() if expression.to else "text"
            if isinstance(expression.this, exp.Literal):
                return f"{expression.this.sql(dialect=self.dialect)}::{type_sql}"
            if isinstance(expression.this, exp.Null):
                return f"NULL::{type_sql}"
        if isinstance(expression, exp.Literal):
            return expression.sql(dialect=self.dialect)
        if isinstance(expression, exp.Null):
            return "NULL"
        return str(expression.sql(dialect=self.dialect))

    @staticmethod
    def _physical_table_from_expression(expression: Any) -> str | None:
        if not isinstance(expression, exp.Expression):
            return None
        tables = list(expression.find_all(exp.Table))
        if not tables:
            return None
        table = tables[0]
        parts = [part for part in (table.catalog, table.db, table.name) if part]
        return ".".join(parts) if parts else table.name

    @staticmethod
    def _extract_literal_value(expression: Any) -> str | None:
        if not isinstance(expression, exp.Expression):
            return None

        root = expression.this if isinstance(expression, exp.Alias) else expression
        if isinstance(root, exp.Cast):
            inner = ColumnLineageExtractor._extract_literal_value(root.this)
            if inner is not None:
                return inner
            root = root.this
        if isinstance(root, exp.Literal):
            return str(root.this)
        if isinstance(root, exp.Null):
            return "NULL"
        return None

    def _collect_union_branches(self, node: Node) -> list[dict[str, Any]]:
        branches: list[dict[str, Any]] = []
        leaves = self._ordered_leaves(node)

        for index, current in enumerate(leaves, start=1):
            expression = current.expression
            expr_sql = self._node_expression_sql(current)

            if self._is_constant_expression(expression) or (
                self._is_positional_union_name(current.name or "")
                and not self._column_refs_from_expression(expression)
            ):
                formatted = self._format_literal_expression(expression)
                branches.append(
                    {
                        "branch_index": index,
                        "kind": "literal",
                        "expression": formatted,
                        "literal_value": self._extract_literal_value(expression),
                    }
                )
                continue

            refs = self._refs_from_leaf(current)
            if not refs:
                continue

            for ref in refs:
                branches.append(
                    {
                        "branch_index": index,
                        "kind": "column_ref",
                        "expression": expr_sql,
                        "table_alias": ref.get("table"),
                        "column": ref["column"],
                        "physical_table": self._physical_table_from_expression(expression),
                    }
                )
        return branches

    def _node_expression_sql(self, node: Node) -> str:
        expression = node.expression
        if isinstance(expression, exp.Expression):
            return expression.sql(dialect=self.dialect)
        return str(expression)

    @staticmethod
    def _find_outermost_select(tree: exp.Expression) -> exp.Select | None:
        """Return the outermost SELECT node."""
        if isinstance(tree, exp.Select):
            return tree
        for node in tree.walk():
            if isinstance(node, exp.Select):
                return node
        return None

    @staticmethod
    def _extract_tables_from_tree(tree: exp.Expression) -> list[str]:
        """Best-effort list of table names referenced in the AST."""
        tables: set[str] = set()
        for tbl in tree.find_all(exp.Table):
            if tbl.name:
                tables.add(tbl.name)
        return sorted(tables)
