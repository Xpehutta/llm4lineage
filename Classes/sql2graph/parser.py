"""Deterministic sqlglot-based SQL2Graph parser."""
from __future__ import annotations

import copy
import logging
import re
from typing import Any

from Classes.pipeline.core.lineage import ColumnLineageExtractor
from Classes.pipeline.core.parser import SQLParser
from Classes.pipeline.core.serializer import ASTSerializer
from Classes.pipeline.exceptions import ParsingError
from Classes.schema_registry import SchemaRegistry
from Classes.view_expander import ViewExpander

logger = logging.getLogger(__name__)

try:
    import sqlglot  # type: ignore[import-not-found]
    from sqlglot import exp  # type: ignore[import-not-found]
    from sqlglot.errors import ParseError  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    sqlglot = None
    exp = None
    ParseError = Exception  # type: ignore[misc, assignment]

class SQL2GraphParser:
    """Parse SQL into a compact structure suitable for prompt context."""

    def __init__(self, dialect: str = "postgres", schema_registry: SchemaRegistry | None = None):
        self.sqlglot_available = sqlglot is not None
        self._default_dialect = dialect
        self.schema_registry = schema_registry

    def _parse_tree(self, sql: str, dialect: str | None):
        if not self.sqlglot_available:
            return None, None
        effective_dialect = dialect or self._default_dialect
        parser = SQLParser(dialect=effective_dialect, error_on_incomplete=True)
        try:
            return parser.parse(sql), None
        except ParsingError as exc:
            return None, str(exc)

    @staticmethod
    def _get_arg(node: Any, key: str) -> Any:
        """Read an AST arg across sqlglot versions (e.g. 'with' vs 'with_')."""
        if not hasattr(node, "args"):
            return None
        return node.args.get(key) or node.args.get(f"{key}_")

    @staticmethod
    def _strip_leading_clause(text: str, clause: str) -> str:
        if not text:
            return ""
        pattern = rf"^\s*{re.escape(clause)}\s+"
        return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    def _collect_column_refs_from_condition(self, condition_sql: str, dialect: str | None) -> list[dict[str, str | None]]:
        """Extract alias.column refs from a boolean condition using sqlglot."""
        if not condition_sql or not self.sqlglot_available:
            return []

        # Wraps a fragment so sqlglot can parse it. Never executed against a
        # database, so the interpolation is not an injection vector.
        probe_sql = f"SELECT 1 FROM __t WHERE {condition_sql}"  # noqa: S608
        tree, parse_error = self._parse_tree(probe_sql, dialect)
        if parse_error or tree is None:
            return []

        refs: list[dict[str, str | None]] = []
        seen = set()
        where_node = tree.find(exp.Where)
        if not where_node:
            return refs

        for col in where_node.find_all(exp.Column):
            alias = col.table or None
            column = col.name
            key = (alias, column)
            if key in seen:
                continue
            seen.add(key)
            refs.append({"table_alias": alias, "column": column})
        return refs

    def _extract_subgraph_blocks(self, tree: Any, dialect: str | None) -> list[dict[str, Any]]:
        """Extract structural blocks for subgraph rendering (CTE, JOIN, UNION branches)."""
        blocks: list[dict[str, Any]] = []

        # CTE blocks
        with_expr = self._get_arg(tree, "with") or tree.find(exp.With)
        if with_expr:
            for cte in with_expr.expressions:
                cte_sql = self._expression_to_sql(cte.this, dialect)
                blocks.append(
                    {
                        "id": f"cte::{cte.alias_or_name}",
                        "type": "cte",
                        "name": cte.alias_or_name,
                        "sql": cte_sql,
                    }
                )

        # JOIN blocks
        join_counter = 0
        for select_node in tree.find_all(exp.Select):
            for join in select_node.args.get("joins") or []:
                on_raw = self._expression_to_sql(join.args.get("on"), dialect)
                on_condition = self._strip_leading_clause(on_raw, "ON")
                join_columns = self._collect_column_refs_from_condition(on_condition, dialect)
                right_alias = join.this.alias_or_name if hasattr(join.this, "alias_or_name") else self._expression_to_sql(join.this, dialect)
                blocks.append(
                    {
                        "id": f"subjoin::{join_counter}",
                        "type": "subjoin",
                        "name": f"{(join.args.get('kind') or 'INNER').upper()}::{right_alias}",
                        "sql": f"JOIN {self._expression_to_sql(join.this, dialect)} ON {on_condition}",
                        "join_columns": join_columns,
                    }
                )
                join_counter += 1

        # UNION branch blocks
        union_counter = 0
        for union in tree.find_all(exp.Union):
            branches = [("left", union.this), ("right", union.expression)]
            for side, branch in branches:
                if branch is None:
                    continue
                select_aliases: list[str] = []
                if isinstance(branch, exp.Select):
                    for sel in branch.expressions or []:
                        alias_name = sel.alias_or_name if hasattr(sel, "alias_or_name") else None
                        if alias_name:
                            select_aliases.append(alias_name)
                blocks.append(
                    {
                        "id": f"union::{union_counter}::{side}",
                        "type": "union_block",
                        "name": f"union_{union_counter}_{side}",
                        "sql": self._expression_to_sql(branch, dialect),
                        "select_aliases": select_aliases,
                    }
                )
            union_counter += 1

        return blocks

    @staticmethod
    def _expression_to_sql(expression: Any, dialect: str | None) -> str:
        if expression is None:
            return ""
        try:
            return expression.sql(dialect=dialect)
        except Exception:
            return str(expression)

    @staticmethod
    def _statement_context(tree: Any, dialect: str | None) -> dict[str, str | None]:
        """Detect ETL statement type and insert target (spec section 2)."""
        statement_type = "select"
        target_table: str | None = None

        if exp is not None and isinstance(tree, exp.Insert):
            statement_type = "insert"
            insert_target = tree.this
            if insert_target is not None:
                table_expr = insert_target.this if hasattr(insert_target, "this") else insert_target
                target_table = SQL2GraphParser._expression_to_sql(table_expr, dialect)
        elif exp is not None and isinstance(tree, exp.Create):
            statement_type = "create_table_as"
            created = tree.this
            if created is not None:
                target_table = (
                    created.sql(dialect=dialect)
                    if hasattr(created, "sql")
                    else str(created)
                )

        return {"statement_type": statement_type, "target_table": target_table}

    def simplify(
        self,
        sql: str,
        dialect: str | None = None,
        *,
        use_schema: bool = True,
    ) -> dict[str, Any]:
        if not self.sqlglot_available:
            return {"raw_sql": sql, "parser_used": False, "subgraph_blocks": []}

        tree, parse_error = self._parse_tree(sql, dialect)
        if parse_error:
            return {
                "raw_sql": sql,
                "parser_used": False,
                "subgraph_blocks": [],
                "parse_error": parse_error,
            }
        if tree is None:
            return {"raw_sql": sql, "parser_used": False, "subgraph_blocks": []}

        schema_applied = False
        views_expanded = False
        if use_schema and self.schema_registry is not None:
            if self.schema_registry.views:
                tree = ViewExpander(dialect or self._default_dialect).expand(tree, self.schema_registry)
                views_expanded = True
            if self.schema_registry.has_tables():
                qualified = self.schema_registry.qualify_expression(tree)
                if qualified is not None:
                    tree = qualified
                    schema_applied = True

        effective_dialect = dialect or self._default_dialect
        lineage_extractor = ColumnLineageExtractor(dialect=effective_dialect)
        serializer = ASTSerializer()
        column_lineage: list[dict[str, Any]] = []
        ast_summary: dict[str, Any] = {}
        try:
            column_lineage = lineage_extractor.extract(tree)
            ast_summary = serializer.serialize(tree)
        except Exception as exc:
            # The caller still gets a usable simplified query without these two
            # enrichments, but the reason must not vanish.
            logger.debug("Column lineage/AST enrichment failed: %s", exc)

        statement = self._statement_context(tree, dialect)
        if not isinstance(tree, exp.Select) and not tree.find(exp.Select):
            return {
                "raw_sql": sql,
                "parser_used": True,
                "subgraph_blocks": [],
                **statement,
            }

        select_node = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)

        from_tables = []
        from_node = self._get_arg(select_node, "from")
        if from_node:
            for table in from_node.find_all(exp.Table):
                from_tables.append(
                    {
                        "table": self._expression_to_sql(table.this, dialect),
                        "alias": table.alias_or_name,
                    }
                )

        joins = []
        deterministic_joins = []
        for join in select_node.args.get("joins") or []:
            if isinstance(join.this, exp.Table):
                # Render the bare table name; the alias is reported separately.
                right_table = self._expression_to_sql(join.this.this, dialect)
            else:
                right_table = self._expression_to_sql(join.this, dialect)
            on_raw = self._expression_to_sql(join.args.get("on"), dialect)
            on_condition = self._strip_leading_clause(on_raw, "ON")
            join_columns = self._collect_column_refs_from_condition(on_condition, dialect)
            joins.append(
                {
                    "type": (join.args.get("kind") or "INNER").upper(),
                    "right_table": right_table,
                    "alias": join.this.alias_or_name if hasattr(join.this, "alias_or_name") else right_table,
                    "on": on_raw,
                }
            )
            deterministic_joins.append(
                {
                    "type": (join.args.get("kind") or "INNER").upper(),
                    "left_alias": "",
                    "right_alias": join.this.alias_or_name if hasattr(join.this, "alias_or_name") else right_table,
                    "condition": on_condition,
                    "join_columns": join_columns[:2] if len(join_columns) >= 2 else [],
                }
            )

        group_by_columns = []
        group = select_node.args.get("group")
        if group:
            for group_expr in group.expressions:
                if isinstance(group_expr, exp.Column):
                    group_by_columns.append(
                        {
                            "table_alias": group_expr.table or None,
                            "column": group_expr.name,
                        }
                    )

        ctes = []
        with_expr = self._get_arg(tree, "with") or tree.find(exp.With)
        if with_expr:
            for cte in with_expr.expressions:
                ctes.append(
                    {
                        "alias": cte.alias_or_name,
                        "query": self._expression_to_sql(cte.this, dialect),
                    }
                )

        where_raw = self._expression_to_sql(select_node.args.get("where"), dialect)
        having_raw = self._expression_to_sql(select_node.args.get("having"), dialect)
        where_condition = self._strip_leading_clause(where_raw, "WHERE")
        having_condition = self._strip_leading_clause(having_raw, "HAVING")

        deterministic_filters = []
        if where_condition:
            deterministic_filters.append(
                {
                    "clause": "WHERE",
                    "condition": where_condition,
                    "columns_used": self._collect_column_refs_from_condition(where_condition, dialect),
                }
            )
        if having_condition:
            deterministic_filters.append(
                {
                    "clause": "HAVING",
                    "condition": having_condition,
                    "columns_used": self._collect_column_refs_from_condition(having_condition, dialect),
                }
            )

        return {
            "parser_used": True,
            **statement,
            "select": {
                "columns": [
                    self._expression_to_sql(expression, dialect)
                    for expression in (select_node.expressions or [])
                ],
                "aliases": [
                    expression.alias_or_name if hasattr(expression, "alias_or_name") else ""
                    for expression in (select_node.expressions or [])
                ],
            },
            "from": from_tables,
            "joins": joins,
            "where": where_raw,
            "group_by": group_by_columns,
            "having": having_raw,
            "ctes": ctes,
            "deterministic_filters": deterministic_filters,
            "deterministic_joins": deterministic_joins,
            "subgraph_blocks": self._extract_subgraph_blocks(tree, dialect),
            "column_lineage": column_lineage,
            "ast_summary": ast_summary,
            "schema_applied": schema_applied,
            "views_expanded": views_expanded,
            "operators": self._extract_operators(
                select_node.expressions or [],
                self._extract_subgraph_blocks(tree, dialect),
                column_lineage,
            ),
        }

    @staticmethod
    def _extract_operators(
        select_expressions: list[Any],
        subgraph_blocks: list[dict[str, Any]],
        column_lineage: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        operators: list[dict[str, Any]] = []
        seen: set[str] = set()

        for block in subgraph_blocks or []:
            if block.get("type") != "union_block":
                continue
            op_id = str(block.get("id") or block.get("name") or "union")
            if op_id in seen:
                continue
            seen.add(op_id)
            operators.append(
                {
                    "type": "union",
                    "id": op_id,
                    "union_type": "ALL",
                    "sql": block.get("sql", ""),
                }
            )

        for col in column_lineage or []:
            target = col.get("target_column") or ""
            expression = col.get("expression") or ""
            if SQL2GraphParser._looks_aggregate(expression):
                operators.append(
                    {
                        "type": "aggregate",
                        "target_column": target,
                        "function": SQL2GraphParser._parse_aggregate_function_name(expression),
                        "expression": expression,
                    }
                )
            if SQL2GraphParser._looks_window(expression):
                operators.append(
                    {
                        "type": "window",
                        "target_column": target,
                        "expression": expression,
                    }
                )
            branches = col.get("union_branches") or []
            if len(branches) > 1:
                key = f"union::{target}"
                if key not in seen:
                    seen.add(key)
                    operators.append(
                        {
                            "type": "union",
                            "id": key,
                            "target_column": target,
                            "union_type": "ALL",
                            "branch_count": len(branches),
                        }
                    )

        for expression in select_expressions or []:
            sql_text = expression.sql() if hasattr(expression, "sql") else str(expression)
            if SQL2GraphParser._looks_transformation(sql_text) and not SQL2GraphParser._looks_aggregate(sql_text):
                alias = expression.alias_or_name if hasattr(expression, "alias_or_name") else ""
                if alias:
                    operators.append(
                        {
                            "type": "transformation",
                            "target_column": alias,
                            "function": "EXPR",
                            "expression": sql_text,
                        }
                    )
        return operators

    @staticmethod
    def _parse_aggregate_function_name(expression: str) -> str | None:
        match = re.search(r"\b(SUM|COUNT|AVG|MIN|MAX)\s*\(", expression or "", re.IGNORECASE)
        return match.group(1).upper() if match else None

    @staticmethod
    def _looks_transformation(expression: str) -> bool:
        return bool(re.search(r"\b(CASE|CAST|COALESCE|::)\b", expression or "", re.IGNORECASE))

    @staticmethod
    def _normalize_deterministic_join(join: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(join or {})
        cols = list(normalized.get("join_columns") or [])
        while len(cols) < 2:
            fallback_alias = normalized.get("right_alias") or normalized.get("left_alias") or "unknown"
            cols.append({"table_alias": fallback_alias, "column": "unknown"})
        normalized["join_columns"] = cols[:2]
        normalized.setdefault("type", "INNER")
        normalized.setdefault("left_alias", "")
        normalized.setdefault("right_alias", "")
        normalized.setdefault("condition", "")
        return normalized

    @staticmethod
    def _looks_aggregate(expression: str) -> bool:
        return bool(re.search(r"\b(SUM|COUNT|AVG|MIN|MAX)\s*\(", expression or "", re.IGNORECASE))

    @staticmethod
    def _looks_window(expression: str) -> bool:
        return bool(re.search(r"\bOVER\s*\(", expression or "", re.IGNORECASE))

    @staticmethod
    def _cte_alias_lookup(simplified: dict[str, Any]) -> dict[str, str]:
        """Map query aliases (and CTE names) to canonical CTE names."""
        cte_names = {
            str(cte.get("alias", "")).strip().lower(): str(cte.get("alias", "")).strip()
            for cte in simplified.get("ctes", [])
            if cte.get("alias")
        }
        if not cte_names:
            return {}

        alias_map: dict[str, str] = {name.lower(): name for name in cte_names.values()}
        candidates = list(simplified.get("from", []) or [])
        for join in simplified.get("joins", []) or []:
            candidates.append({"table": join.get("right_table"), "alias": join.get("alias")})

        for item in candidates:
            table = str(item.get("table") or "").strip().strip('"').lower()
            alias = str(item.get("alias") or "").strip()
            if table not in cte_names:
                continue
            cte_name = cte_names[table]
            alias_map[table] = cte_name
            if alias:
                alias_map[alias.lower()] = cte_name
        return alias_map

    @staticmethod
    def _parse_cte_passthrough(expression: str, simplified: dict[str, Any]) -> dict[str, str] | None:
        alias_to_cte = SQL2GraphParser._cte_alias_lookup(simplified)
        if not alias_to_cte:
            return None
        token = (expression or "").strip().split()[0]
        match = re.match(r"^(\w+)\.(\w+)$", token)
        if not match:
            return None
        alias, column = match.group(1), match.group(2)
        if alias.lower() not in alias_to_cte:
            return None
        return {"table_alias": alias, "column": column}

    @staticmethod
    def _physical_dependencies_from_branches(
        union_branches: list[dict[str, Any]],
        fallback: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        seen: set[tuple[str | None, str]] = set()
        for branch in union_branches:
            if branch.get("kind") != "column_ref":
                continue
            physical_table = branch.get("physical_table")
            table_alias = physical_table or branch.get("table_alias")
            column = branch.get("column")
            if not column:
                continue
            key = (physical_table or table_alias, column)
            if key in seen:
                continue
            seen.add(key)
            dep: dict[str, Any] = {"table_alias": table_alias, "column": column}
            if physical_table:
                dep["physical_table"] = physical_table
            resolved.append(dep)
        return resolved or fallback

    def _enrich_output_columns_from_ctes(
        self,
        extraction: dict[str, Any],
        simplified: dict[str, Any],
    ) -> None:
        ctes_by_name = {
            str(cte.get("alias", "")).strip(): cte
            for cte in extraction.get("ctes", [])
            if cte.get("alias")
        }
        alias_to_cte = self._cte_alias_lookup(simplified)
        if not ctes_by_name or not alias_to_cte:
            return

        for output in extraction.get("output_columns", []):
            passthrough = self._parse_cte_passthrough(output.get("expression", ""), simplified)
            if not passthrough:
                continue
            cte_name = alias_to_cte.get(passthrough["table_alias"].lower())
            if not cte_name:
                continue
            cte_col = next(
                (
                    col
                    for col in ctes_by_name.get(cte_name, {}).get("output_columns", [])
                    if col.get("alias") == passthrough["column"]
                ),
                None,
            )
            if not cte_col:
                continue
            cte_deps = list(cte_col.get("dependencies") or [])
            if cte_col.get("literal_values"):
                output["derivation_kind"] = "literal"
                output["dependencies"] = []
            elif cte_deps:
                output["derivation_kind"] = "cte_passthrough"
                output["dependencies"] = cte_deps
            else:
                output["derivation_kind"] = "cte_passthrough"
                output["dependencies"] = []
            if cte_col.get("union_branches"):
                output["union_branches"] = cte_col["union_branches"]
            if cte_col.get("literal_values"):
                output["literal_values"] = cte_col["literal_values"]

    def _resolve_cte_dependencies(
        self,
        table_alias: str | None,
        column: str,
        *,
        ctes_by_name: dict[str, dict[str, Any]],
        alias_to_cte: dict[str, str],
        visiting: set[tuple[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        if not table_alias:
            return [{"table_alias": table_alias, "column": column}]

        cte_name = alias_to_cte.get(table_alias.lower()) or alias_to_cte.get(table_alias)
        if not cte_name:
            return [{"table_alias": table_alias, "column": column}]

        visit_key = (cte_name, column)
        visiting = visiting or set()
        if visit_key in visiting:
            return [{"table_alias": table_alias, "column": column}]
        visiting.add(visit_key)

        cte_scope = ctes_by_name.get(cte_name)
        if not cte_scope:
            return [{"table_alias": table_alias, "column": column}]

        cte_column = next(
            (col for col in cte_scope.get("output_columns", []) if col.get("alias") == column),
            None,
        )
        if not cte_column:
            return [{"table_alias": table_alias, "column": column}]

        nested_deps = cte_column.get("dependencies") or []
        if not nested_deps:
            return []

        return [{"table_alias": table_alias, "column": column}]

    def _materialize_output_dependencies(
        self,
        extraction: dict[str, Any],
        simplified: dict[str, Any],
    ) -> dict[str, Any]:
        ctes_by_name = {
            str(cte.get("alias", "")).strip(): cte
            for cte in extraction.get("ctes", [])
            if cte.get("alias")
        }
        alias_to_cte = self._cte_alias_lookup(simplified)
        if not ctes_by_name or not alias_to_cte:
            return extraction

        for output in extraction.get("output_columns", []):
            resolved: list[dict[str, Any]] = []
            seen: set[tuple[str | None, str]] = set()
            for dep in output.get("dependencies") or []:
                for ref in self._resolve_cte_dependencies(
                    dep.get("table_alias"),
                    dep["column"],
                    ctes_by_name=ctes_by_name,
                    alias_to_cte=alias_to_cte,
                ):
                    key = (ref.get("table_alias"), ref["column"])
                    if key in seen:
                        continue
                    seen.add(key)
                    resolved.append(ref)
            output["dependencies"] = resolved
        return extraction

    def overlay_deterministic_column_lineage(
        self,
        extracted: dict[str, Any],
        deterministic: dict[str, Any],
    ) -> dict[str, Any]:
        """Restore sqlglot-derived per-column lineage after optional LLM edits."""
        det_by_alias = {
            str(col.get("alias", "")).strip(): col
            for col in deterministic.get("output_columns", [])
            if col.get("alias")
        }
        lineage_fields = ("dependencies", "derivation_kind", "literal_values", "union_branches")
        for output in extracted.get("output_columns", []):
            det_col = det_by_alias.get(str(output.get("alias", "")).strip())
            if not det_col:
                continue
            for field in lineage_fields:
                if field in det_col:
                    output[field] = copy.deepcopy(det_col[field])
        return extracted

    def build_deterministic_extraction(
        self,
        simplified: dict[str, Any],
        dialect: str | None = None,
    ) -> dict[str, Any]:
        """Build SQL2GraphExtraction-compatible JSON purely from sqlglot output."""
        if not simplified.get("parser_used"):
            return {
                "ctes": [],
                "output_columns": [],
                "filters": [],
                "joins": [],
                "group_by_columns": [],
            }

        output_columns: list[dict[str, Any]] = []
        for entry in simplified.get("column_lineage") or []:
            expression = str(entry.get("expression") or entry.get("target_column") or "")
            union_branches = list(entry.get("union_branches") or [])
            literal_values = list(entry.get("literal_values") or [])
            source_deps = [
                {"table_alias": ref.get("table"), "column": ref["column"]}
                for ref in entry.get("source_columns") or []
            ]
            passthrough = self._parse_cte_passthrough(expression, simplified)

            if passthrough:
                derivation_kind = "cte_passthrough"
                physical = self._physical_dependencies_from_branches(union_branches, [])
                dependencies = physical if physical else []
            elif literal_values or (
                union_branches and all(branch.get("kind") == "literal" for branch in union_branches)
            ):
                derivation_kind = "literal"
                dependencies = []
            else:
                derivation_kind = "column_ref"
                dependencies = self._physical_dependencies_from_branches(union_branches, source_deps)

            output_columns.append(
                {
                    "alias": entry["target_column"],
                    "expression": expression,
                    "dependencies": dependencies,
                    "aggregate": self._looks_aggregate(expression),
                    "window_function": self._looks_window(expression),
                    "derivation_kind": derivation_kind,
                    "literal_values": literal_values,
                    "union_branches": union_branches,
                }
            )

        ctes: list[dict[str, Any]] = []
        for cte in simplified.get("ctes") or []:
            cte_sql = cte.get("query") or ""
            nested = self.simplify(cte_sql, dialect=dialect) if cte_sql else {"parser_used": False}
            nested_extraction = self.build_deterministic_extraction(nested, dialect=dialect)
            nested_extraction["alias"] = cte.get("alias") or "cte"
            ctes.append(nested_extraction)

        extraction = {
            "ctes": ctes,
            "output_columns": output_columns,
            "filters": list(simplified.get("deterministic_filters") or []),
            "joins": [
                self._normalize_deterministic_join(join)
                for join in simplified.get("deterministic_joins") or []
            ],
            "group_by_columns": list(simplified.get("group_by") or []),
        }
        self._enrich_output_columns_from_ctes(extraction, simplified)
        return self._materialize_output_dependencies(extraction, simplified)
