"""Extract column-level lineage from a parsed SQL AST."""

import logging
from typing import Any, Dict, List, Optional

from sqlglot import exp

from Classes.pipeline.exceptions import LineageExtractionError

logger = logging.getLogger(__name__)


class ColumnLineageExtractor:
    """Extract column-level lineage from a parsed SQL AST."""

    def __init__(self, dialect: str = "spark", include_intermediate: bool = False):
        self.dialect = dialect
        self.include_intermediate = include_intermediate
        # Note: include_intermediate is reserved for subclass extensions.
        # The base class ignores it.

    def extract(self, tree: exp.Expression) -> List[Dict[str, Any]]:
        """Return a list of lineage records, one per output column.

        Raises:
            LineageExtractionError: If no SELECT is found or extraction fails.
        """
        try:
            select = self._find_outermost_select(tree)
            if select is None:
                raise LineageExtractionError("No SELECT statement found in AST.")

            lineage: List[Dict[str, Any]] = []

            for proj in select.expressions:
                if isinstance(proj, exp.Star):
                    lineage.append({
                        "target_column": "*",
                        "source_columns": [],
                        "expression": "*",
                        "used_tables": self._extract_tables_from_select(select),
                    })
                    continue

                target_name = proj.alias or proj.sql(dialect=self.dialect)
                source_cols = self._extract_column_refs(proj)

                lineage.append({
                    "target_column": target_name,
                    "source_columns": source_cols,
                    "expression": proj.sql(dialect=self.dialect),
                    "used_tables": sorted(
                        {c["table"] for c in source_cols if c.get("table")}
                    ),
                })

            logger.debug("Extracted lineage for %d output columns.", len(lineage))
            return lineage

        except LineageExtractionError:
            raise
        except Exception as exc:
            raise LineageExtractionError(
                f"Lineage extraction failed: {exc}"
            ) from exc

    @staticmethod
    def _find_outermost_select(tree: exp.Expression) -> Optional[exp.Select]:
        """Return the outermost SELECT node."""
        if isinstance(tree, exp.Select):
            return tree
        for node in tree.walk():
            if isinstance(node, exp.Select):
                return node
        return None

    def _extract_column_refs(
        self, expr: exp.Expression
    ) -> List[Dict[str, Optional[str]]]:
        refs: List[Dict[str, Optional[str]]] = []
        for node in expr.walk():
            if isinstance(node, exp.Column):
                refs.append({
                    "table": node.table or None,
                    "column": node.name,
                })
        return refs

    @staticmethod
    def _extract_tables_from_select(select: exp.Select) -> List[str]:
        """Best-effort list of table names referenced in FROM / JOIN."""
        tables: set[str] = set()
        for tbl in select.find_all(exp.Table):
            if tbl.name:
                tables.add(tbl.name)
        return sorted(tables)
