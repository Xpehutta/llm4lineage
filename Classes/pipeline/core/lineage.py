"""Extract column-level lineage from a parsed SQL AST."""

import logging
from typing import Any, Dict, List, Optional

from sqlglot import exp
from sqlglot import lineage as sqlglot_lineage
from sqlglot.lineage import Node

from Classes.pipeline.exceptions import LineageExtractionError

logger = logging.getLogger(__name__)


class ColumnLineageExtractor:
    """Extract column-level lineage using sqlglot's built-in lineage module."""

    def __init__(
        self,
        dialect: str = "spark",
        include_intermediate: bool = False,
        schema_catalog: Optional[Dict[str, List[str]]] = None,
    ):
        self.dialect = dialect
        self.include_intermediate = include_intermediate
        self.schema_catalog = schema_catalog
        # Note: include_intermediate is reserved for subclass extensions.
        # The base class ignores it.

    def extract(self, tree: exp.Expression) -> List[Dict[str, Any]]:
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

            records: List[Dict[str, Any]] = []
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
    ) -> Dict[str, Any]:
        if target_name == "*":
            return {
                "target_column": "*",
                "source_columns": [],
                "expression": "*",
                "used_tables": self._extract_tables_from_tree(tree),
            }

        source_columns = self._collect_leaf_sources(node)
        expression = self._node_expression_sql(node)

        return {
            "target_column": target_name,
            "source_columns": source_columns,
            "expression": expression,
            "used_tables": sorted(
                {ref["table"] for ref in source_columns if ref.get("table")}
            ),
        }

    @staticmethod
    def _to_sqlglot_schema(
        schema_catalog: Optional[Dict[str, List[str]]],
    ) -> Optional[Dict[str, Dict[str, str]]]:
        if not schema_catalog:
            return None
        return {
            table: {column: "UNKNOWN" for column in columns}
            for table, columns in schema_catalog.items()
        }

    @staticmethod
    def _collect_leaf_sources(node: Node) -> List[Dict[str, Optional[str]]]:
        refs: List[Dict[str, Optional[str]]] = []
        seen: set[tuple[Optional[str], str]] = set()

        for current in node.walk():
            if current.downstream:
                continue
            ref = ColumnLineageExtractor._parse_source_ref(current.name)
            key = (ref["table"], ref["column"])
            if key in seen:
                continue
            seen.add(key)
            refs.append(ref)

        return refs

    @staticmethod
    def _parse_source_ref(name: str) -> Dict[str, Optional[str]]:
        table, _, column = name.rpartition(".")
        if table:
            return {"table": table, "column": column}
        return {"table": None, "column": name}

    def _node_expression_sql(self, node: Node) -> str:
        expression = node.expression
        if isinstance(expression, exp.Expression):
            return expression.sql(dialect=self.dialect)
        return str(expression)

    @staticmethod
    def _find_outermost_select(tree: exp.Expression) -> Optional[exp.Select]:
        """Return the outermost SELECT node."""
        if isinstance(tree, exp.Select):
            return tree
        for node in tree.walk():
            if isinstance(node, exp.Select):
                return node
        return None

    @staticmethod
    def _extract_tables_from_tree(tree: exp.Expression) -> List[str]:
        """Best-effort list of table names referenced in the AST."""
        tables: set[str] = set()
        for tbl in tree.find_all(exp.Table):
            if tbl.name:
                tables.add(tbl.name)
        return sorted(tables)
