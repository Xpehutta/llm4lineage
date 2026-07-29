import hashlib
import html
import json
import re
import time
import copy
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
from pydantic import BaseModel, Field, ValidationError, field_validator

from langchain_core.messages import HumanMessage, SystemMessage

from Classes.helper_classes import HuggingFaceLLMAdapter, resolve_model_name, resolve_provider
from Classes.pipeline.core.lineage import ColumnLineageExtractor
from Classes.pipeline.core.parser import SQLParser
from Classes.pipeline.core.serializer import ASTSerializer
from Classes.pipeline.exceptions import ParsingError
from Classes.pipeline.llm_helpers import create_chat_model, resolve_hf_token

try:
    import sqlglot  # type: ignore[import-not-found]
    from sqlglot import exp  # type: ignore[import-not-found]
    from sqlglot.errors import ParseError  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    sqlglot = None
    exp = None
    ParseError = Exception  # type: ignore[misc, assignment]


class ColumnRef(BaseModel):
    table_alias: Optional[str] = None
    column: str
    physical_table: Optional[str] = None

    @field_validator("table_alias")
    def normalize_alias(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() if isinstance(value, str) and value.strip() else None

    @field_validator("column")
    def normalize_column(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("column cannot be empty")
        return cleaned

    def node_id(self) -> str:
        alias = self.table_alias or "unknown"
        return f"{alias}.{self.column}"


class OutputColumn(BaseModel):
    alias: str
    expression: str = ""
    dependencies: List[ColumnRef] = Field(default_factory=list)
    aggregate: bool = False
    window_function: bool = False
    derivation_kind: Optional[str] = None
    literal_values: List[str] = Field(default_factory=list)
    union_branches: List[Dict[str, Any]] = Field(default_factory=list)


class FilterSpec(BaseModel):
    clause: str
    condition: str
    columns_used: List[ColumnRef] = Field(default_factory=list)


class JoinSpec(BaseModel):
    type: str
    left_alias: str
    right_alias: str
    condition: str
    join_columns: List[ColumnRef]

    @field_validator("join_columns")
    def validate_join_pair(cls, value: List[ColumnRef]) -> List[ColumnRef]:
        if len(value) != 2:
            raise ValueError("join_columns must contain exactly two entries")
        return value


class SQL2GraphExtraction(BaseModel):
    ctes: List["SQL2GraphExtractionCTE"] = Field(default_factory=list)
    output_columns: List[OutputColumn]
    filters: List[FilterSpec] = Field(default_factory=list)
    joins: List[JoinSpec] = Field(default_factory=list)
    group_by_columns: List[ColumnRef] = Field(default_factory=list)


class SQL2GraphExtractionCTE(BaseModel):
    alias: str
    output_columns: List[OutputColumn]
    filters: List[FilterSpec] = Field(default_factory=list)
    joins: List[JoinSpec] = Field(default_factory=list)
    group_by_columns: List[ColumnRef] = Field(default_factory=list)
    ctes: List["SQL2GraphExtractionCTE"] = Field(default_factory=list)


SQL2GraphExtraction.model_rebuild()
SQL2GraphExtractionCTE.model_rebuild()


class SQL2GraphParser:
    """Parse SQL into a compact structure suitable for prompt context."""

    def __init__(self, dialect: str = "spark"):
        self.sqlglot_available = sqlglot is not None
        self._default_dialect = dialect

    def _parse_tree(self, sql: str, dialect: Optional[str]):
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

    def _collect_column_refs_from_condition(self, condition_sql: str, dialect: Optional[str]) -> List[Dict[str, Optional[str]]]:
        """Extract alias.column refs from a boolean condition using sqlglot."""
        if not condition_sql or not self.sqlglot_available:
            return []

        probe_sql = f"SELECT 1 FROM __t WHERE {condition_sql}"
        tree, parse_error = self._parse_tree(probe_sql, dialect)
        if parse_error or tree is None:
            return []

        refs: List[Dict[str, Optional[str]]] = []
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

    def _extract_subgraph_blocks(self, tree: Any, dialect: Optional[str]) -> List[Dict[str, Any]]:
        """Extract structural blocks for subgraph rendering (CTE, JOIN, UNION branches)."""
        blocks: List[Dict[str, Any]] = []

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
                select_aliases: List[str] = []
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
    def _expression_to_sql(expression: Any, dialect: Optional[str]) -> str:
        if expression is None:
            return ""
        try:
            return expression.sql(dialect=dialect)
        except Exception:
            return str(expression)

    @staticmethod
    def _statement_context(tree: Any, dialect: Optional[str]) -> Dict[str, Optional[str]]:
        """Detect ETL statement type and insert target (spec section 2)."""
        statement_type = "select"
        target_table: Optional[str] = None

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

    def simplify(self, sql: str, dialect: Optional[str] = None) -> Dict[str, Any]:
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

        effective_dialect = dialect or self._default_dialect
        lineage_extractor = ColumnLineageExtractor(dialect=effective_dialect)
        serializer = ASTSerializer()
        column_lineage: List[Dict[str, Any]] = []
        ast_summary: Dict[str, Any] = {}
        try:
            column_lineage = lineage_extractor.extract(tree)
            ast_summary = serializer.serialize(tree)
        except Exception:
            pass

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
        }

    @staticmethod
    def _normalize_deterministic_join(join: Dict[str, Any]) -> Dict[str, Any]:
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
    def _cte_alias_lookup(simplified: Dict[str, Any]) -> Dict[str, str]:
        """Map query aliases (and CTE names) to canonical CTE names."""
        cte_names = {
            str(cte.get("alias", "")).strip().lower(): str(cte.get("alias", "")).strip()
            for cte in simplified.get("ctes", [])
            if cte.get("alias")
        }
        if not cte_names:
            return {}

        alias_map: Dict[str, str] = {name.lower(): name for name in cte_names.values()}
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
    def _parse_cte_passthrough(expression: str, simplified: Dict[str, Any]) -> Optional[Dict[str, str]]:
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
        union_branches: List[Dict[str, Any]],
        fallback: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        resolved: List[Dict[str, Any]] = []
        seen: set[tuple[Optional[str], str]] = set()
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
            dep: Dict[str, Any] = {"table_alias": table_alias, "column": column}
            if physical_table:
                dep["physical_table"] = physical_table
            resolved.append(dep)
        return resolved or fallback

    def _enrich_output_columns_from_ctes(
        self,
        extraction: Dict[str, Any],
        simplified: Dict[str, Any],
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
        table_alias: Optional[str],
        column: str,
        *,
        ctes_by_name: Dict[str, Dict[str, Any]],
        alias_to_cte: Dict[str, str],
        visiting: Optional[Set[tuple[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
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
        extraction: Dict[str, Any],
        simplified: Dict[str, Any],
    ) -> Dict[str, Any]:
        ctes_by_name = {
            str(cte.get("alias", "")).strip(): cte
            for cte in extraction.get("ctes", [])
            if cte.get("alias")
        }
        alias_to_cte = self._cte_alias_lookup(simplified)
        if not ctes_by_name or not alias_to_cte:
            return extraction

        for output in extraction.get("output_columns", []):
            resolved: List[Dict[str, Any]] = []
            seen: set[tuple[Optional[str], str]] = set()
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
        extracted: Dict[str, Any],
        deterministic: Dict[str, Any],
    ) -> Dict[str, Any]:
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
        simplified: Dict[str, Any],
        dialect: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build SQL2GraphExtraction-compatible JSON purely from sqlglot output."""
        if not simplified.get("parser_used"):
            return {
                "ctes": [],
                "output_columns": [],
                "filters": [],
                "joins": [],
                "group_by_columns": [],
            }

        output_columns: List[Dict[str, Any]] = []
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

        ctes: List[Dict[str, Any]] = []
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


class SQL2GraphLLMExtractor:
    """LLM-backed extractor for column-level lineage JSON."""

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        hf_token: Optional[str] = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        max_retries: int = 3,
        enable_refinement: bool = True,
    ):
        if not resolve_hf_token(hf_token):
            raise ValueError("HF_TOKEN is required for SQL2Graph extraction.")

        model = resolve_model_name(model)
        provider = resolve_provider(provider)
        self.model = model
        self.provider = provider
        self.max_retries = max_retries
        self.enable_refinement = enable_refinement
        self.chat_model = create_chat_model(
            model=model,
            provider=provider,
            hf_token=hf_token,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        self.chat_adapter = HuggingFaceLLMAdapter(self.chat_model)
        self.structured_llm = self._try_create_structured_llm(self.chat_model)

        self.verification_system_prompt = (
            "You are a strict SQL lineage verifier. You receive deterministic column-level lineage "
            "produced by sqlglot plus the original SQL. Verify the draft against the SQL and return "
            "corrected JSON only when you find concrete issues. "
            "Return keys: ctes, output_columns, filters, joins, group_by_columns. "
            "Preserve correct sqlglot-derived fields; do not invent columns absent from the SQL."
        )
        self.enhancement_system_prompt = (
            "You are a SQL lineage enhancer. You receive a verified column-level lineage draft. "
            "Apply targeted enhancements inferable from the SQL: complete missing dependencies, "
            "filters, join keys, and CTE scopes. Preserve fields that are already correct."
        )
        # Backward-compatible aliases
        self.system_prompt = self.verification_system_prompt
        self.refinement_system_prompt = self.enhancement_system_prompt

    @property
    def _llm_system_prompt(self) -> str:
        return getattr(self, "verification_system_prompt", None) or getattr(self, "system_prompt", "")

    @staticmethod
    def _try_create_structured_llm(chat_model: Any) -> Any:
        """Return a structured-output runnable when the model supports it."""
        try:
            return chat_model.with_structured_output(SQL2GraphExtraction)
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_structured_result(result: Any) -> SQL2GraphExtraction:
        if isinstance(result, SQL2GraphExtraction):
            return result
        if isinstance(result, dict):
            return SQL2GraphExtraction.model_validate(result)
        raise TypeError(f"Unexpected structured LLM result type: {type(result)!r}")

    def _invoke_structured_extraction(self, messages: List[Any]) -> SQL2GraphExtraction:
        """Invoke the LLM and return validated SQL2GraphExtraction."""
        structured_llm = getattr(self, "structured_llm", None)
        if structured_llm is not None:
            return self._coerce_structured_result(structured_llm.invoke(messages))

        response_text = self._invoke_messages_text(messages)
        payload = self._normalize_scope_payload(self._extract_json(response_text))
        return SQL2GraphExtraction.model_validate(payload)

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise

    @staticmethod
    def _extract_column_refs_from_text(text: str) -> List[Dict[str, str]]:
        """Best-effort extraction of alias.column pairs from condition text."""
        pattern = r'([A-Za-z_][\w\$]*)\.(?:"([^"]+)"|([A-Za-z_][\w\$]*))'
        refs: List[Dict[str, str]] = []
        seen = set()
        for match in re.findall(pattern, text or ""):
            alias = match[0]
            column = match[1] or match[2]
            key = (alias, column)
            if key in seen:
                continue
            seen.add(key)
            refs.append({"table_alias": alias, "column": column})
        return refs

    @staticmethod
    def _coerce_column_ref(value: Any) -> Optional[Dict[str, Optional[str]]]:
        """
        Coerce common LLM column-reference variants into {"table_alias", "column"}.

        Accepts "alias.column" / "column" strings and dicts using alternative keys
        (table/alias instead of table_alias, name instead of column).
        """
        if isinstance(value, str):
            text = value.strip().strip('"')
            if not text:
                return None
            alias, _, column = text.rpartition(".")
            return {"table_alias": alias or None, "column": column.strip('"')}
        if isinstance(value, dict):
            column = value.get("column") or value.get("name")
            if not column or not str(column).strip():
                return None
            alias = value.get("table_alias") or value.get("table") or value.get("alias")
            return {"table_alias": str(alias) if alias else None, "column": str(column).strip()}
        return None

    @classmethod
    def _coerce_join_columns(cls, join: Dict[str, Any]) -> List[Dict[str, Optional[str]]]:
        """Coerce join_columns variants, including {"left_column", "right_column"} pairs."""
        raw = join.get("join_columns")
        items = raw if isinstance(raw, list) else ([raw] if raw else [])

        refs: List[Dict[str, Optional[str]]] = []
        for item in items:
            if isinstance(item, dict) and ("left_column" in item or "right_column" in item):
                for side, alias_key in (("left_column", "left_alias"), ("right_column", "right_alias")):
                    ref = cls._coerce_column_ref(item.get(side))
                    if ref:
                        if not ref.get("table_alias"):
                            ref["table_alias"] = join.get(alias_key) or None
                        refs.append(ref)
            else:
                ref = cls._coerce_column_ref(item)
                if ref:
                    refs.append(ref)
        return refs

    @classmethod
    def _normalize_scope_payload(cls, scope: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize partially structured LLM payload into schema-compatible shape."""
        normalized = dict(scope or {})
        normalized.setdefault("output_columns", [])
        normalized.setdefault("filters", [])
        normalized.setdefault("joins", [])
        normalized.setdefault("ctes", [])
        normalized.setdefault("group_by_columns", [])

        fixed_outputs = []
        for item in normalized.get("output_columns", []):
            out = dict(item or {})
            deps = out.get("dependencies") or []
            out["dependencies"] = [ref for ref in (cls._coerce_column_ref(dep) for dep in deps) if ref]
            fixed_outputs.append(out)
        normalized["output_columns"] = fixed_outputs

        fixed_filters = []
        for item in normalized.get("filters", []):
            filt = dict(item or {})
            columns_used = filt.get("columns_used") or []
            filt["columns_used"] = [ref for ref in (cls._coerce_column_ref(col) for col in columns_used) if ref]
            clause = filt.get("clause")

            # Some model outputs place the full predicate in `clause` and omit `condition`.
            if not filt.get("condition"):
                if isinstance(clause, str) and any(token in clause for token in ["=", ">", "<", "(", ")", " and ", " or "]):
                    filt["condition"] = clause
                    filt["clause"] = "WHERE"
                else:
                    filt["condition"] = ""

            if not filt.get("clause"):
                filt["clause"] = "WHERE"
            fixed_filters.append(filt)
        normalized["filters"] = fixed_filters

        fixed_joins = []
        for item in normalized.get("joins", []):
            join = dict(item or {})
            join.setdefault("type", "INNER")
            join.setdefault("left_alias", "")
            join.setdefault("right_alias", "")
            join.setdefault("condition", "")

            join_columns = cls._coerce_join_columns(join)
            if len(join_columns) >= 2:
                join["join_columns"] = join_columns[:2]
            else:
                extracted = cls._extract_column_refs_from_text(join.get("condition", ""))
                if len(extracted) >= 2:
                    join["join_columns"] = extracted[:2]
                else:
                    join["join_columns"] = [
                        {"table_alias": join.get("left_alias") or "unknown_left", "column": "unknown"},
                        {"table_alias": join.get("right_alias") or "unknown_right", "column": "unknown"},
                    ]
            fixed_joins.append(join)
        normalized["joins"] = fixed_joins

        normalized["group_by_columns"] = [
            ref
            for ref in (cls._coerce_column_ref(col) for col in normalized.get("group_by_columns", []) or [])
            if ref
        ]

        fixed_ctes = []
        for cte in normalized.get("ctes", []):
            cte_copy = dict(cte or {})
            cte_copy = cls._normalize_scope_payload(cte_copy)
            cte_copy.setdefault("alias", "cte")
            fixed_ctes.append(cte_copy)
        normalized["ctes"] = fixed_ctes

        return normalized

    @staticmethod
    def _is_auth_error(error: Exception) -> bool:
        marker = str(error).lower()
        return any(s in marker for s in ["401", "unauthorized", "bad credentials", "forbidden"])

    def _build_verification_prompt(
        self,
        sql: str,
        schema: Optional[Dict[str, Any]],
        simplified_query: Optional[Dict[str, Any]],
        deterministic_draft: Dict[str, Any],
        validation_error: Optional[str] = None,
    ) -> str:
        return "\n".join(
            [
                "Original SQL:",
                sql,
                "",
                "Schema JSON (optional):",
                json.dumps(schema or {}, indent=2),
                "",
                "Sqlglot simplify summary (optional):",
                json.dumps(simplified_query or {}, indent=2),
                "",
                "Deterministic sqlglot lineage draft to verify:",
                json.dumps(deterministic_draft, indent=2),
                "",
                "Tasks:",
                "1. Verify each output column dependency against the SQL.",
                "2. Confirm filters, join keys, and CTE scopes match the SQL.",
                "3. Preserve correct sqlglot fields; do not replace valid dependencies.",
                "4. Return ONLY JSON: ctes, output_columns, filters, joins, group_by_columns.",
                "5. If the draft is already correct, return it with minimal or no changes.",
            ]
            + (
                ["", "Previous output failed validation:", validation_error, "Fix and return corrected JSON only."]
                if validation_error
                else []
            )
        )

    def _build_refinement_prompt(
        self,
        sql: str,
        schema: Optional[Dict[str, Any]],
        simplified_query: Optional[Dict[str, Any]],
        draft_payload: Dict[str, Any],
    ) -> str:
        return "\n".join(
            [
                "Original SQL:",
                sql,
                "",
                "Schema JSON (optional):",
                json.dumps(schema or {}, indent=2),
                "",
                "Simplified query structure (optional):",
                json.dumps(simplified_query or {}, indent=2),
                "",
                "Draft lineage JSON to repair:",
                json.dumps(draft_payload, indent=2),
                "",
                "Repair requirements:",
                "- Ensure every filter has non-empty clause and condition.",
                "- Ensure joins have 2 concrete join_columns when inferable from join condition.",
                "- Ensure output dependencies are complete for expressions/functions/CASE.",
                "- Preserve existing good fields and avoid inventing unrelated aliases.",
                "- Keep strict JSON shape compatible with: ctes, output_columns, filters, joins, group_by_columns.",
            ]
        )

    @staticmethod
    def _response_to_text(response: Any) -> str:
        """Normalize chat response object to text."""
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)

    def _invoke_messages_text(self, messages: List[Any]) -> str:
        """Invoke chat using adapter with compatibility for older adapter versions."""
        if hasattr(self.chat_adapter, "invoke_messages"):
            return self.chat_adapter.invoke_messages(messages)

        # Backward compatibility: older adapter may only expose invoke(prompt).
        if hasattr(self.chat_adapter, "invoke"):
            raw = self.chat_adapter.invoke(messages)
            return self._response_to_text(raw)

        # Last resort fallback to direct chat model invocation.
        raw = self.chat_model.invoke(messages)
        return self._response_to_text(raw)

    def _refine_payload_with_llm(
        self,
        sql: str,
        schema: Optional[Dict[str, Any]],
        simplified_query: Optional[Dict[str, Any]],
        draft_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt = self._build_refinement_prompt(
            sql=sql,
            schema=schema,
            simplified_query=simplified_query,
            draft_payload=draft_payload,
        )
        messages = [
            SystemMessage(content=self.enhancement_system_prompt),
            HumanMessage(content=prompt),
        ]
        structured_llm = getattr(self, "structured_llm", None)
        if structured_llm is not None:
            refined = self._coerce_structured_result(structured_llm.invoke(messages))
            return self._normalize_scope_payload(refined.model_dump())

        response_text = self._invoke_messages_text(messages)
        refined = self._extract_json(response_text)
        return self._normalize_scope_payload(refined)

    def _build_user_prompt(
        self,
        sql: str,
        schema: Optional[Dict[str, Any]],
        simplified_query: Optional[Dict[str, Any]],
        validation_error: Optional[str] = None,
    ) -> str:
        parts = [
            "SQL:",
            sql,
            "",
            "Schema JSON (optional):",
            json.dumps(schema or {}, indent=2),
            "",
            "Simplified query structure (optional):",
            json.dumps(simplified_query or {}, indent=2),
            "",
            "Output format constraints:",
            "- output_columns[].dependencies[] must include table_alias and column",
            "- filters[].columns_used[] must include all columns in condition",
            "- joins[].join_columns must contain exactly 2 entries",
            "- include ctes recursively when present",
            "- include group_by_columns when GROUP BY is present",
        ]
        if validation_error:
            parts.extend(["", "Previous output failed validation:", validation_error, "Fix and return corrected JSON only."])
        return "\n".join(parts)

    def _invoke_verification_payload(
        self,
        sql: str,
        draft: Dict[str, Any],
        *,
        schema: Optional[Dict[str, Any]] = None,
        simplified_query: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        last_validation_error: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                user_prompt = self._build_verification_prompt(
                    sql=sql,
                    schema=schema,
                    simplified_query=simplified_query,
                    deterministic_draft=draft,
                    validation_error=last_validation_error,
                )
                validated = self._invoke_structured_extraction(
                    [
                        SystemMessage(content=self.verification_system_prompt),
                        HumanMessage(content=user_prompt),
                    ]
                )
                return self._normalize_scope_payload(validated.model_dump())
            except ValidationError as exc:
                last_validation_error = str(exc)
                if attempt == self.max_retries:
                    break
            except Exception as exc:
                if self._is_auth_error(exc):
                    return {
                        "error": "Hugging Face authentication failed for SQL2Graph extractor.",
                        "details": str(exc),
                    }
                if attempt == self.max_retries:
                    break
                time.sleep(min(10, 2**attempt))

        try:
            return SQL2GraphExtraction.model_validate(draft).model_dump()
        except ValidationError:
            return {
                "error": "LLM verification failed and sqlglot draft is invalid",
                "details": last_validation_error,
            }

    def verify(
        self,
        sql: str,
        deterministic_draft: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
        simplified_query: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Step 3: verify sqlglot draft with LLM."""
        draft = self._normalize_scope_payload(dict(deterministic_draft or {}))
        return self._invoke_verification_payload(
            sql,
            draft,
            schema=schema,
            simplified_query=simplified_query,
        )

    def enhance(
        self,
        sql: str,
        verified_payload: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
        simplified_query: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Step 4: enhance a verified draft with targeted LLM repairs."""
        if not self.enable_refinement:
            return self._normalize_scope_payload(dict(verified_payload or {}))

        draft = self._normalize_scope_payload(dict(verified_payload or {}))
        try:
            return self._refine_payload_with_llm(
                sql=sql,
                schema=schema,
                simplified_query=simplified_query,
                draft_payload=draft,
            )
        except ValidationError as exc:
            return {
                "error": "LLM enhancement validation failed",
                "details": str(exc),
            }
        except Exception as exc:
            if self._is_auth_error(exc):
                return {
                    "error": "Hugging Face authentication failed for SQL2Graph extractor.",
                    "details": str(exc),
                }
            return {"error": "LLM enhancement failed", "details": str(exc)}

    def verify_and_enhance(
        self,
        sql: str,
        deterministic_draft: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
        simplified_query: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run verification then optional enhancement."""
        verified = self.verify(
            sql=sql,
            deterministic_draft=deterministic_draft,
            schema=schema,
            simplified_query=simplified_query,
        )
        if "error" in verified:
            return verified
        return self.enhance(
            sql=sql,
            verified_payload=verified,
            schema=schema,
            simplified_query=simplified_query,
        )

    def extract(
        self,
        sql: str,
        schema: Optional[Dict[str, Any]] = None,
        simplified_query: Optional[Dict[str, Any]] = None,
        deterministic_draft: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Verify/enhance a sqlglot draft when available; otherwise cold-start extraction."""
        draft = deterministic_draft
        if draft is None and simplified_query and simplified_query.get("parser_used"):
            draft = SQL2GraphParser().build_deterministic_extraction(simplified_query)

        if draft:
            return self.verify_and_enhance(
                sql=sql,
                deterministic_draft=draft,
                schema=schema,
                simplified_query=simplified_query,
            )

        last_validation_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                user_prompt = self._build_user_prompt(
                    sql=sql,
                    schema=schema,
                    simplified_query=simplified_query,
                    validation_error=last_validation_error,
                )
                validated = self._invoke_structured_extraction(
                    [SystemMessage(content=self._llm_system_prompt), HumanMessage(content=user_prompt)]
                )
                return self._normalize_scope_payload(validated.model_dump())
            except ValidationError as exc:
                last_validation_error = str(exc)
                if attempt == self.max_retries:
                    return {
                        "error": "LLM output validation failed",
                        "details": last_validation_error,
                    }
            except Exception as exc:
                if self._is_auth_error(exc):
                    return {
                        "error": "Hugging Face authentication failed for SQL2Graph extractor.",
                        "details": str(exc),
                    }
                if attempt == self.max_retries:
                    return {"error": "SQL2Graph extraction failed", "details": str(exc)}
                time.sleep(min(10, 2**attempt))

        return {"error": "SQL2Graph extraction failed", "details": "unknown error"}


class SQL2GraphBuilder:
    """Build a column-level lineage graph from structured extraction JSON."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def _add_source_column(self, ref: ColumnRef) -> str:
        node_id = ref.node_id()
        self.graph.add_node(
            node_id,
            node_type="source_column",
            table_alias=ref.table_alias or "unknown",
            column=ref.column,
        )
        return node_id

    def _add_filter_node(self, clause: str, condition: str) -> str:
        digest = hashlib.md5(f"{clause}:{condition}".encode("utf-8")).hexdigest()[:12]
        node_id = f"filter_{digest}"
        self.graph.add_node(node_id, node_type="filter", clause=clause, condition=condition)
        return node_id

    def _add_scope(
        self,
        scope: Dict[str, Any],
        output_prefix: str,
        output_node_type: str,
    ) -> List[str]:
        output_nodes = []

        for output in scope.get("output_columns", []):
            output_obj = OutputColumn.model_validate(output)
            out_id = f"{output_prefix}.{output_obj.alias}"
            self.graph.add_node(
                out_id,
                node_type=output_node_type,
                alias=output_obj.alias,
                expression=output_obj.expression,
                aggregate=output_obj.aggregate,
                window_function=output_obj.window_function,
            )
            output_nodes.append(out_id)

            for dep in output_obj.dependencies:
                dep_node = self._add_source_column(dep)
                self.graph.add_edge(dep_node, out_id, edge_type="DERIVED_FROM")

            if output_obj.aggregate:
                for group_ref in scope.get("group_by_columns", []):
                    grp = ColumnRef.model_validate(group_ref)
                    grp_node = self._add_source_column(grp)
                    self.graph.add_edge(grp_node, out_id, edge_type="GROUPED_BY")

        for filt in scope.get("filters", []):
            f = FilterSpec.model_validate(filt)
            filter_node = self._add_filter_node(f.clause, f.condition)
            for used in f.columns_used:
                col_node = self._add_source_column(used)
                self.graph.add_edge(col_node, filter_node, edge_type="USES_COLUMN")
            for out in output_nodes:
                self.graph.add_edge(filter_node, out, edge_type="FILTERED_BY")

        for join in scope.get("joins", []):
            j = JoinSpec.model_validate(join)
            left = self._add_source_column(j.join_columns[0])
            right = self._add_source_column(j.join_columns[1])
            self.graph.add_edge(left, right, edge_type="JOINS_ON", join_type=j.type, condition=j.condition)

        for cte in scope.get("ctes", []):
            cte_obj = SQL2GraphExtractionCTE.model_validate(cte)
            self._add_scope(cte_obj.model_dump(), output_prefix=cte_obj.alias, output_node_type="source_column")

        return output_nodes

    def build(self, extraction: Dict[str, Any]) -> nx.MultiDiGraph:
        validated = SQL2GraphExtraction.model_validate(extraction)
        self.graph = nx.MultiDiGraph()
        self._add_scope(validated.model_dump(), output_prefix="output", output_node_type="output_column")
        return self.graph

    def link_cte_aliases(self, alias_map: Dict[str, str]) -> int:
        """
        Connect CTE output nodes to alias-qualified references of the same column.

        When the main query aliases a CTE (e.g. ``JOIN recent_orders r``), the LLM
        dependencies reference ``r.total`` while the CTE scope produces
        ``recent_orders.total``. This links them so the lineage chain stays connected
        (spec section 7.9 / 8.3).
        """
        if not alias_map:
            return 0
        added = 0
        for node, attrs in list(self.graph.nodes(data=True)):
            if attrs.get("node_type") != "source_column":
                continue
            cte_name = alias_map.get(attrs.get("table_alias"))
            if not cte_name:
                continue
            cte_node = f"{cte_name}.{attrs.get('column')}"
            if cte_node != node and cte_node in self.graph.nodes:
                if not self.graph.has_edge(cte_node, node):
                    self.graph.add_edge(cte_node, node, edge_type="DERIVED_FROM")
                    added += 1
        return added

    def materialize_transitive_derived_from(self, output_node_type: str = "output_column") -> int:
        """
        Add direct DERIVED_FROM edges from ultimate source columns to outputs.

        Implements spec section 7.10: when lineage passes through intermediate
        column nodes (e.g. CTE passthrough), materialize shortcut edges so
        downstream consumers can query source-to-target without walking the chain.
        """
        added = 0
        for target, attrs in list(self.graph.nodes(data=True)):
            if attrs.get("node_type") != output_node_type:
                continue

            stack = [target]
            visited = set()
            leaves: set = set()
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)

                predecessors = [
                    source
                    for source, _, edge_data in self.graph.in_edges(node, data=True)
                    if edge_data.get("edge_type") == "DERIVED_FROM"
                ]
                if not predecessors:
                    if node != target:
                        leaves.add(node)
                    continue
                stack.extend(predecessors)

            for source in leaves:
                if source == target:
                    continue
                has_direct = any(
                    tgt == target and edge_data.get("edge_type") == "DERIVED_FROM"
                    for _, tgt, edge_data in self.graph.out_edges(source, data=True)
                )
                if not has_direct:
                    self.graph.add_edge(source, target, edge_type="DERIVED_FROM", transitive=True)
                    added += 1
        return added

    def ensure_acyclic(self) -> List[str]:
        """
        Break any remaining directed cycles so the graph is a DAG.

        Prefers removing transitive shortcut edges, then other edges participating
        in the first detected cycle.
        """
        warnings: List[str] = []
        while self.graph.number_of_edges() > 0 and not nx.is_directed_acyclic_graph(self.graph):
            try:
                cycle = nx.find_cycle(self.graph)
            except nx.NetworkXNoCycle:
                break

            removable = None
            for u, v, key in cycle:
                edge_data = self.graph.edges[u, v, key]
                if edge_data.get("transitive"):
                    removable = (u, v, key)
                    break
            if removable is None:
                removable = cycle[0]

            u, v, key = removable
            edge_type = self.graph.edges[u, v, key].get("edge_type", "")
            self.graph.remove_edge(u, v, key)
            warnings.append(f"Removed cyclic edge: {u} -> {v} ({edge_type})")

        return warnings

    def to_node_link(self) -> Dict[str, Any]:
        # Keep "links" key for backward compatibility in notebook/UI code.
        try:
            return json_graph.node_link_data(self.graph, edges="links")
        except TypeError:
            return json_graph.node_link_data(self.graph)

    def to_dot(self) -> str:
        lines = ["digraph SQL2Graph {"]
        for node, attrs in self.graph.nodes(data=True):
            label = attrs.get("alias") or node
            lines.append(f'  "{node}" [label="{label}\\n({attrs.get("node_type", "node")})"];')
        for source, target, attrs in self.graph.edges(data=True):
            lines.append(f'  "{source}" -> "{target}" [label="{attrs.get("edge_type", "")}"];')
        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        lines = ["flowchart TD"]
        for source, target, attrs in self.graph.edges(data=True):
            edge_label = attrs.get("edge_type", "")
            lines.append(f'  "{source}" -->|{edge_label}| "{target}"')
        return "\n".join(lines)


class SQL2GraphVisualizer:
    """Render SQL2Graph node-link JSON output."""

    EDGE_COLORS = {
        "DERIVED_FROM": "#1f77b4",
        "FILTERED_BY": "#ff7f0e",
        "USES_COLUMN": "#2ca02c",
        "JOINS_ON": "#d62728",
        "GROUPED_BY": "#9467bd",
        "CHUNK_LINK": "#444444",
        "CONTAINS": "#999999",
        "JOIN": "#d62728",
        "INSERT": "#2ca02c",
        "UNION": "#1f77b4",
        "UNION ALL": "#1f77b4",
    }

    LINEAGE_EDGE_TYPES = frozenset(
        {
            "DERIVED_FROM",
            "FILTERED_BY",
            "USES_COLUMN",
            "GROUPED_BY",
            "JOINS_ON",
            "CHUNK_LINK",
            "CONTAINS",
            "JOIN",
            "INSERT",
            "UNION",
            "UNION ALL",
            "UNION DISTINCT",
            "INTERSECT",
            "EXCEPT",
        }
    )

    HIGHLIGHT_SELECTED_COLOR = "#FF5722"
    HIGHLIGHT_LINEAGE_COLOR = "#FFC107"
    HIGHLIGHT_DIMMED_COLOR = "#E8E8E8"

    NODE_COLORS = {
        "source_column": "#90EE90",
        "output_column": "#ADD8E6",
        "filter": "#F6D186",
        "join": "#F08080",
        "chunk": "#ADD8E6",
    }

    CHUNK_TYPE_COLORS = {
        "target": "#FFB6C1",
        "cte": "#DDA0DD",
        "query": "#ADD8E6",
    }

    NODE_SHAPES = {
        "source_column": "dot",
        "output_column": "box",
        "filter": "diamond",
        "join": "triangle",
    }

    NODE_TYPE_LABELS = {
        "source_column": "Source column",
        "output_column": "Output column",
        "filter": "Filter",
        "join": "Join",
    }

    @staticmethod
    def graph_from_node_link(graph_json: Dict[str, Any]) -> nx.MultiDiGraph:
        # Support both historic "links" and newer "edges".
        if "links" in graph_json:
            try:
                graph = json_graph.node_link_graph(graph_json, edges="links")
            except TypeError:
                graph = json_graph.node_link_graph(graph_json)
        elif "edges" in graph_json and "links" not in graph_json:
            normalized = dict(graph_json)
            normalized["links"] = normalized.get("edges", [])
            graph = json_graph.node_link_graph(normalized)
        else:
            graph = json_graph.node_link_graph(graph_json)

        if isinstance(graph, nx.MultiDiGraph):
            return graph

        directed = nx.MultiDiGraph()
        directed.add_nodes_from(graph.nodes(data=True))
        for source, target, _key, data in graph.edges(keys=True, data=True):
            directed.add_edge(source, target, **data)
        return directed

    @staticmethod
    def _hierarchical_layout(graph: nx.MultiDiGraph) -> Dict[Any, Tuple[float, float]]:
        """Layer nodes by topological order for DAG visualization."""
        if not nx.is_directed_acyclic_graph(graph):
            return nx.spring_layout(graph, seed=42, k=1.4)

        layers: Dict[Any, int] = {}
        for node in nx.topological_sort(graph):
            preds = list(graph.predecessors(node))
            layers[node] = 0 if not preds else max(layers[p] for p in preds) + 1

        by_layer: Dict[int, List[Any]] = {}
        for node, layer in layers.items():
            by_layer.setdefault(layer, []).append(node)

        pos: Dict[Any, Tuple[float, float]] = {}
        max_layer = max(layers.values()) if layers else 0
        for layer, nodes in by_layer.items():
            y = 1.0 - (layer / max_layer) if max_layer else 0.5
            spacing = 1.0 / (len(nodes) + 1)
            for index, node in enumerate(sorted(nodes)):
                pos[node] = ((index + 1) * spacing - 0.5, y)
        return pos

    @classmethod
    def draw(
        cls,
        graph_json: Dict[str, Any],
        figsize: Tuple[int, int] = (16, 10),
        with_labels: bool = True,
        layout: str = "spring",
        title: str = "SQL2Graph Column Lineage",
    ):
        graph = cls.graph_from_node_link(graph_json)
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot visualize empty graph.")

        if layout in {"hierarchical", "dag"}:
            pos = cls._hierarchical_layout(graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(graph)
        elif layout == "shell":
            pos = nx.shell_layout(graph)
        elif nx.is_directed_acyclic_graph(graph):
            pos = cls._hierarchical_layout(graph)
        else:
            pos = nx.spring_layout(graph, seed=42, k=1.4)

        plt.figure(figsize=figsize)

        node_colors = [
            cls.NODE_COLORS.get(graph.nodes[node].get("node_type", ""), "#CCCCCC")
            for node in graph.nodes()
        ]
        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=node_colors,
            node_size=1300,
            edgecolors="black",
            linewidths=0.8,
        )

        grouped_edges: Dict[str, List[Tuple[str, str, int]]] = {}
        for source, target, key, attrs in graph.edges(keys=True, data=True):
            edge_type = attrs.get("edge_type", "OTHER")
            grouped_edges.setdefault(edge_type, []).append((source, target, key))

        for edge_type, triples in grouped_edges.items():
            color = cls.EDGE_COLORS.get(edge_type, "#7f7f7f")
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=[(source, target) for source, target, _ in triples],
                edge_color=color,
                width=1.8,
                alpha=0.8,
                arrows=True,
                arrowsize=14,
                connectionstyle="arc3,rad=0.08",
            )

        if with_labels:
            labels = {}
            for node, attrs in graph.nodes(data=True):
                alias = attrs.get("alias")
                if alias:
                    labels[node] = alias
                else:
                    labels[node] = node if len(node) < 36 else f"{node[:33]}..."
            nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)

        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        return graph

    @classmethod
    def _node_display_label(cls, node_id: str, attrs: Dict[str, Any]) -> str:
        label = attrs.get("label")
        if label:
            return str(label)
        alias = attrs.get("alias")
        if alias:
            return str(alias)
        if len(node_id) <= 28:
            return node_id
        return f"{node_id[:25]}..."

    @classmethod
    def _node_hover_title(cls, node_id: str, attrs: Dict[str, Any]) -> str:
        parts = [
            f"<b>{html.escape(cls._node_display_label(node_id, attrs))}</b>",
            f"Type: {html.escape(cls.NODE_TYPE_LABELS.get(attrs.get('node_type', ''), attrs.get('node_type', 'unknown')))}",
            f"ID: {html.escape(node_id)}",
        ]
        for key in ("table_alias", "column", "expression", "table"):
            value = attrs.get(key)
            if value:
                parts.append(f"{key.replace('_', ' ').title()}: {html.escape(str(value))}")
        return "<br>".join(parts)

    @staticmethod
    def _format_detail_block(title: str, rows: List[Tuple[str, str]]) -> str:
        if not rows:
            return ""
        body = "".join(
            f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
            for label, value in rows
            if value
        )
        if not body:
            return ""
        return f"<h4>{html.escape(title)}</h4><table class='detail-table'>{body}</table>"

    @classmethod
    def _node_detail_html(cls, graph: nx.MultiDiGraph, node_id: str) -> str:
        if node_id not in graph.nodes:
            return "<p>Node not found.</p>"
        attrs = dict(graph.nodes[node_id])
        node_type = attrs.get("node_type", "")
        rows = [
            ("Label", cls._node_display_label(node_id, attrs)),
            ("Type", cls.NODE_TYPE_LABELS.get(node_type, node_type or "unknown")),
            ("ID", node_id),
            ("Chunk type", str(attrs.get("chunk_type") or "")),
            ("SQL", str(attrs.get("sql") or "")[:1200]),
            ("Table alias", str(attrs.get("table_alias") or "")),
            ("Column", str(attrs.get("column") or "")),
            ("Table", str(attrs.get("table") or "")),
            ("Expression", str(attrs.get("expression") or "")),
        ]
        incoming = []
        for source, _, edge_attrs in graph.in_edges(node_id, data=True):
            incoming.append(
                f"{source} <span class='edge-tag'>{edge_attrs.get('edge_type', 'EDGE')}</span>"
            )
        outgoing = []
        for _, target, edge_attrs in graph.out_edges(node_id, data=True):
            outgoing.append(
                f"{target} <span class='edge-tag'>{edge_attrs.get('edge_type', 'EDGE')}</span>"
            )
        detail = cls._format_detail_block("Node", rows)
        if incoming:
            detail += "<h4>Incoming lineage</h4><ul>" + "".join(
                f"<li>{item}</li>" for item in incoming
            ) + "</ul>"
        if outgoing:
            detail += "<h4>Outgoing lineage</h4><ul>" + "".join(
                f"<li>{item}</li>" for item in outgoing
            ) + "</ul>"
        return detail or "<p>No details available.</p>"

    @classmethod
    def _edge_detail_html(cls, graph: nx.MultiDiGraph, edge_id: str) -> str:
        for source, target, key, attrs in graph.edges(keys=True, data=True):
            current_id = f"{source}->{target}:{key}"
            if current_id != edge_id:
                continue
            rows = [
                ("Type", str(attrs.get("edge_type") or "")),
                ("From", source),
                ("To", target),
            ]
            for key_name, value in attrs.items():
                if key_name == "edge_type" or value in (None, ""):
                    continue
                rows.append((key_name.replace("_", " ").title(), str(value)))
            return cls._format_detail_block("Edge", rows) or "<p>No edge details available.</p>"
        return "<p>Edge not found.</p>"

    @classmethod
    def to_interactive_html(
        cls,
        graph_json: Dict[str, Any],
        height: str = "780px",
        title: str = "SQL2Graph Column Lineage",
    ) -> str:
        """Build a self-contained interactive HTML view (vis.js) with click details."""
        graph = cls.graph_from_node_link(graph_json)
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot visualize empty graph.")

        is_dag = nx.is_directed_acyclic_graph(graph)
        vis_nodes: List[Dict[str, Any]] = []
        node_details: Dict[str, str] = {}
        for node_id, attrs in graph.nodes(data=True):
            node_type = attrs.get("node_type", "")
            vis_nodes.append(
                {
                    "id": node_id,
                    "label": cls._node_display_label(node_id, attrs),
                    "title": cls._node_hover_title(node_id, attrs),
                    "group": node_type or "other",
                    "shape": cls.NODE_SHAPES.get(node_type, "dot"),
                    "color": {
                        "background": cls.NODE_COLORS.get(node_type, "#CCCCCC"),
                        "border": "#2f2f2f",
                        "highlight": {"background": "#fff3bf", "border": "#e67700"},
                    },
                    "font": {"size": 14, "face": "Inter, Arial, sans-serif"},
                    "margin": 10,
                }
            )
            node_details[node_id] = cls._node_detail_html(graph, node_id)

        vis_edges: List[Dict[str, Any]] = []
        edge_details: Dict[str, str] = {}
        for source, target, key, attrs in graph.edges(keys=True, data=True):
            edge_type = attrs.get("edge_type", "EDGE")
            edge_id = f"{source}->{target}:{key}"
            vis_edges.append(
                {
                    "id": edge_id,
                    "from": source,
                    "to": target,
                    "label": edge_type.replace("_", " "),
                    "title": html.escape(edge_type),
                    "arrows": "to",
                    "color": {"color": cls.EDGE_COLORS.get(edge_type, "#7f7f7f"), "highlight": "#111"},
                    "width": 2,
                    "smooth": {"type": "curvedCW", "roundness": 0.12},
                    "font": {"size": 11, "align": "middle", "strokeWidth": 0},
                }
            )
            edge_details[edge_id] = cls._edge_detail_html(graph, edge_id)

        node_legend = "".join(
            f"<span class='legend-item'><i style='background:{color}'></i>{html.escape(cls.NODE_TYPE_LABELS.get(node_type, node_type))}</span>"
            for node_type, color in cls.NODE_COLORS.items()
        )
        edge_legend = "".join(
            f"<span class='legend-item'><i style='background:{color}'></i>{html.escape(edge_type.replace('_', ' '))}</span>"
            for edge_type, color in cls.EDGE_COLORS.items()
        )

        physics_options = (
            {
                "enabled": True,
                "hierarchicalRepulsion": {
                    "nodeDistance": 140,
                    "centralGravity": 0.0,
                    "springLength": 120,
                    "springConstant": 0.01,
                },
                "solver": "hierarchicalRepulsion",
            }
            if is_dag
            else {
                "enabled": True,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -40,
                    "centralGravity": 0.01,
                    "springLength": 120,
                    "avoidOverlap": 1,
                },
                "stabilization": {"iterations": 150},
            }
        )

        layout_options = (
            {
                "hierarchical": {
                    "enabled": True,
                    "direction": "UD",
                    "sortMethod": "directed",
                    "levelSeparation": 170,
                    "nodeSpacing": 180,
                    "treeSpacing": 220,
                }
            }
            if is_dag
            else {}
        )

        payload = {
            "nodes": vis_nodes,
            "edges": vis_edges,
            "nodeDetails": node_details,
            "edgeDetails": edge_details,
            "options": {
                "layout": layout_options,
                "physics": physics_options,
                "interaction": {
                    "hover": True,
                    "multiselect": True,
                    "navigationButtons": True,
                    "keyboard": True,
                    "tooltipDelay": 120,
                },
                "nodes": {"borderWidth": 1.5, "shadow": True},
                "edges": {"shadow": False, "selectionWidth": 2},
            },
            "groups": {
                node_type: {
                    "color": {"background": color, "border": "#2f2f2f"},
                    "shape": cls.NODE_SHAPES.get(node_type, "dot"),
                }
                for node_type, color in cls.NODE_COLORS.items()
            },
        }

        stats = (
            f"{graph.number_of_nodes()} nodes · {graph.number_of_edges()} edges · "
            f"{'DAG' if is_dag else 'cyclic'}"
        )
        payload_json = json.dumps(payload, ensure_ascii=False)
        payload_json = payload_json.replace("</", "<\\/")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, Arial, sans-serif;
    }}
    body {{
      margin: 0;
      background: #f7f8fb;
      color: #1f2937;
    }}
    .toolbar {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      padding: 12px 16px;
      background: #ffffff;
      border-bottom: 1px solid #e5e7eb;
    }}
    .toolbar input, .toolbar select, .toolbar button {{
      font: inherit;
      padding: 8px 10px;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      background: #fff;
    }}
    .toolbar button {{
      cursor: pointer;
      background: #eef2ff;
    }}
    .stats {{
      margin-left: auto;
      color: #6b7280;
      font-size: 13px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 0;
      height: calc({height} - 56px);
      min-height: 520px;
    }}
    #network {{
      background: #ffffff;
      border-right: 1px solid #e5e7eb;
    }}
    #detail-panel {{
      background: #fcfcfd;
      padding: 16px;
      overflow: auto;
    }}
    #detail-panel h3 {{
      margin: 0 0 8px;
      font-size: 18px;
    }}
    #detail-content {{
      font-size: 14px;
      line-height: 1.45;
    }}
    .detail-table {{
      width: 100%;
      border-collapse: collapse;
      margin: 8px 0 14px;
      font-size: 13px;
    }}
    .detail-table th {{
      text-align: left;
      vertical-align: top;
      width: 38%;
      color: #6b7280;
      padding: 6px 8px 6px 0;
      font-weight: 600;
    }}
    .detail-table td {{
      padding: 6px 0;
      word-break: break-word;
    }}
    .edge-tag {{
      display: inline-block;
      margin-left: 6px;
      padding: 1px 6px;
      border-radius: 999px;
      background: #eef2ff;
      color: #3730a3;
      font-size: 11px;
    }}
    .legend {{
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      padding: 10px 16px 14px;
      background: #ffffff;
      border-top: 1px solid #e5e7eb;
      font-size: 12px;
    }}
    .legend-item i {{
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 3px;
      margin-right: 6px;
      vertical-align: -2px;
    }}
    .hint {{
      color: #6b7280;
      font-size: 13px;
      margin-bottom: 12px;
    }}
    ul {{
      margin: 0 0 12px 18px;
      padding: 0;
    }}
  </style>
</head>
<body>
  <div class="toolbar">
    <input id="search" type="search" placeholder="Search nodes..." aria-label="Search nodes" />
    <select id="type-filter" aria-label="Filter by node type">
      <option value="">All node types</option>
      <option value="source_column">Source columns</option>
      <option value="output_column">Output columns</option>
      <option value="filter">Filters</option>
      <option value="join">Joins</option>
    </select>
    <button id="fit-btn" type="button">Fit view</button>
    <button id="reset-btn" type="button">Reset selection</button>
    <span class="stats">{html.escape(stats)}</span>
  </div>
  <div class="layout">
    <div id="network"></div>
    <aside id="detail-panel">
      <h3>{html.escape(title)}</h3>
      <p class="hint">Click a node or edge to inspect lineage. Drag nodes, scroll to zoom, use arrow keys to pan.</p>
      <div id="detail-content">Select a node or edge to see details here.</div>
    </aside>
  </div>
  <div class="legend">
    <div>{node_legend}</div>
    <div>{edge_legend}</div>
  </div>
  <script>
    const payload = {payload_json};
    const nodes = new vis.DataSet(payload.nodes);
    const edges = new vis.DataSet(payload.edges);
    const container = document.getElementById("network");
    const detailContent = document.getElementById("detail-content");
    const network = new vis.Network(container, {{ nodes, edges }}, payload.options);
    network.setOptions({{ groups: payload.groups }});

    function setDetail(html) {{
      detailContent.innerHTML = html;
    }}

    function highlightNodes(matchingIds) {{
      const matchSet = new Set(matchingIds);
      const updates = payload.nodes.map((node) => {{
        if (matchSet.size === 0) {{
          return {{ id: node.id, hidden: false, opacity: 1 }};
        }}
        const matched = matchSet.has(node.id);
        return {{
          id: node.id,
          hidden: !matched,
          opacity: matched ? 1 : 0.15,
        }};
      }});
      nodes.update(updates);
      const edgeUpdates = payload.edges.map((edge) => {{
        if (matchSet.size === 0) {{
          return {{ id: edge.id, hidden: false }};
        }}
        const matched = matchSet.has(edge.from) || matchSet.has(edge.to);
        return {{ id: edge.id, hidden: !matched }};
      }});
      edges.update(edgeUpdates);
      if (matchingIds.length > 0) {{
        network.fit({{ nodes: matchingIds, animation: true }});
      }}
    }}

    network.on("click", (params) => {{
      if (params.nodes.length > 0) {{
        const nodeId = params.nodes[0];
        setDetail(payload.nodeDetails[nodeId] || "<p>No details available.</p>");
        return;
      }}
      if (params.edges.length > 0) {{
        const edgeId = params.edges[0];
        setDetail(payload.edgeDetails[edgeId] || "<p>No details available.</p>");
        return;
      }}
      setDetail("<p>Select a node or edge to see details here.</p>");
    }});

    network.on("doubleClick", (params) => {{
      if (params.nodes.length === 0) {{
        return;
      }}
      const nodeId = params.nodes[0];
      const connected = network.getConnectedNodes(nodeId);
      highlightNodes([nodeId, ...connected]);
      setDetail(payload.nodeDetails[nodeId] || "<p>No details available.</p>");
    }});

    document.getElementById("search").addEventListener("input", (event) => {{
      const query = event.target.value.trim().toLowerCase();
      if (!query) {{
        highlightNodes([]);
        return;
      }}
      const matches = payload.nodes
        .filter((node) => {{
          return String(node.id).toLowerCase().includes(query)
            || String(node.label).toLowerCase().includes(query)
            || String(node.group || "").toLowerCase().includes(query);
        }})
        .map((node) => node.id);
      highlightNodes(matches);
      if (matches.length === 1) {{
        setDetail(payload.nodeDetails[matches[0]] || "<p>No details available.</p>");
      }}
    }});

    document.getElementById("type-filter").addEventListener("change", (event) => {{
      const selected = event.target.value;
      if (!selected) {{
        highlightNodes([]);
        return;
      }}
      const matches = payload.nodes
        .filter((node) => node.group === selected)
        .map((node) => node.id);
      highlightNodes(matches);
    }});

    document.getElementById("fit-btn").addEventListener("click", () => network.fit({{ animation: true }}));
    document.getElementById("reset-btn").addEventListener("click", () => {{
      document.getElementById("search").value = "";
      document.getElementById("type-filter").value = "";
      highlightNodes([]);
      network.unselectAll();
      setDetail("<p>Select a node or edge to see details here.</p>");
    }});

    network.once("stabilizationIterationsDone", () => network.fit({{ animation: true }}));
    if (!payload.options.physics.enabled) {{
      network.fit({{ animation: false }});
    }}
  </script>
</body>
</html>"""

    @staticmethod
    def _parse_height(height: str) -> int:
        value = str(height).strip().lower().removesuffix("px")
        try:
            return max(400, int(float(value)))
        except ValueError:
            return 780

    @classmethod
    def _edge_lineage_type(cls, edge_data: Dict[str, Any]) -> str:
        edge_type = str(edge_data.get("edge_type") or edge_data.get("link_type") or "").strip().upper()
        if edge_type == "UNIONALL":
            return "UNION ALL"
        return edge_type

    @classmethod
    def _iter_lineage_neighbors(cls, graph: nx.MultiDiGraph, node: str) -> List[str]:
        neighbors: List[str] = []
        for pred, _target, _key, edge_data in graph.in_edges(node, keys=True, data=True):
            if cls._edge_lineage_type(edge_data) in cls.LINEAGE_EDGE_TYPES:
                neighbors.append(pred)
        for _source, succ, _key, edge_data in graph.out_edges(node, keys=True, data=True):
            if cls._edge_lineage_type(edge_data) in cls.LINEAGE_EDGE_TYPES:
                neighbors.append(succ)
        return neighbors

    @classmethod
    def collect_lineage_nodes(cls, graph: nx.MultiDiGraph, start: str) -> Set[str]:
        """Collect all nodes connected to ``start`` via lineage edge types."""
        if start not in graph:
            return set()

        visited: Set[str] = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor in cls._iter_lineage_neighbors(graph, node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return visited

    @classmethod
    def _node_marker_color(cls, attrs: Dict[str, Any]) -> str:
        if attrs.get("node_type") == "chunk":
            return cls.CHUNK_TYPE_COLORS.get(attrs.get("chunk_type", "query"), "#ADD8E6")
        return cls.NODE_COLORS.get(attrs.get("node_type", ""), "#CCCCCC")

    @classmethod
    def _build_plotly_figure(
        cls,
        graph: nx.MultiDiGraph,
        title: str,
    ):
        """Build a Plotly figure for notebook-native pan/zoom/click interaction."""
        import plotly.graph_objects as go

        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot visualize empty graph.")

        pos = cls._hierarchical_layout(graph)
        node_ids = list(graph.nodes())

        edge_traces: List[Any] = []
        seen_edge_types: set = set()
        for source, target, _key, attrs in graph.edges(keys=True, data=True):
            edge_type = cls._edge_lineage_type(attrs) or "OTHER"
            seen_edge_types.add(edge_type)

        for edge_type in sorted(seen_edge_types):
            edge_x: List[Optional[float]] = []
            edge_y: List[Optional[float]] = []
            for source, target, _key, attrs in graph.edges(keys=True, data=True):
                if (cls._edge_lineage_type(attrs) or "OTHER") != edge_type:
                    continue
                x0, y0 = pos[source]
                x1, y1 = pos[target]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            if not edge_x:
                continue
            edge_traces.append(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line=dict(width=2, color=cls.EDGE_COLORS.get(edge_type, "#7f7f7f")),
                    hoverinfo="skip",
                    name=edge_type.replace("_", " "),
                    legendgroup="edges",
                )
            )

        node_x = [pos[node_id][0] for node_id in node_ids]
        node_y = [pos[node_id][1] for node_id in node_ids]
        node_labels = [cls._node_display_label(node_id, graph.nodes[node_id]) for node_id in node_ids]
        node_hover = []
        for node_id in node_ids:
            attrs = graph.nodes[node_id]
            hover_lines = [
                cls._node_display_label(node_id, attrs),
                f"Type: {cls.NODE_TYPE_LABELS.get(attrs.get('node_type', ''), attrs.get('node_type', 'unknown'))}",
                f"ID: {node_id}",
            ]
            for key in ("table_alias", "column", "expression", "table"):
                value = attrs.get(key)
                if value:
                    hover_lines.append(f"{key.replace('_', ' ').title()}: {value}")
            node_hover.append("<br>".join(hover_lines))

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_labels,
            textposition="top center",
            textfont=dict(size=10),
            hovertext=node_hover,
            hoverinfo="text",
            marker=dict(
                size=[
                    26
                    if graph.nodes[node_id].get("node_type") == "chunk"
                    else (22 if graph.nodes[node_id].get("node_type") == "output_column" else 16)
                    for node_id in node_ids
                ],
                color=[cls._node_marker_color(graph.nodes[n]) for n in node_ids],
                line=dict(width=1.5, color="#333333"),
            ),
            name="Nodes",
            showlegend=False,
        )

        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title=title,
            hovermode="closest",
            dragmode="pan",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
            plot_bgcolor="#ffffff",
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        return fig, node_ids

    @classmethod
    def _display_plotly_interactive(
        cls,
        graph_json: Dict[str, Any],
        title: str,
        height: str = "780px",
    ) -> nx.MultiDiGraph:
        """Pan/zoom/click graph using Plotly FigureWidget (works reliably in Jupyter)."""
        try:
            import plotly.graph_objects as go
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError as exc:
            raise RuntimeError(
                "Interactive Plotly view requires plotly and ipywidgets. "
                "Install with: uv pip install plotly ipywidgets"
            ) from exc

        graph = cls.graph_from_node_link(graph_json)
        fig, node_ids = cls._build_plotly_figure(graph, title)
        try:
            fig_widget = go.FigureWidget(fig)
        except ImportError as exc:
            raise RuntimeError(
                "Plotly FigureWidget requires anywidget. "
                "Install with: uv sync  (or: uv pip install anywidget), then restart the kernel."
            ) from exc
        fig_widget.update_layout(height=cls._parse_height(height))

        detail = widgets.HTML(
            value=(
                "<p><b>Click a node</b> to highlight its full upstream/downstream lineage. "
                "Drag to pan, scroll to zoom, hover for quick info.</p>"
            ),
            layout=widgets.Layout(
                width="100%",
                min_height="120px",
                border="1px solid #e5e7eb",
                padding="12px",
                overflow="auto",
            ),
        )
        node_trace_idx = len(fig_widget.data) - 1

        default_colors = [cls._node_marker_color(graph.nodes[node_id]) for node_id in node_ids]
        default_sizes = [
            26
            if graph.nodes[node_id].get("node_type") == "chunk"
            else (22 if graph.nodes[node_id].get("node_type") == "output_column" else 16)
            for node_id in node_ids
        ]
        selected: Dict[str, Optional[int]] = {"index": None}

        def apply_highlight(selected_idx: Optional[int]) -> None:
            colors = list(default_colors)
            sizes = list(default_sizes)
            if selected_idx is not None:
                node_id = node_ids[selected_idx]
                lineage_nodes = cls.collect_lineage_nodes(graph, node_id)
                for index, current_id in enumerate(node_ids):
                    if current_id not in lineage_nodes:
                        colors[index] = cls.HIGHLIGHT_DIMMED_COLOR
                        sizes[index] = max(10, default_sizes[index] - 4)
                        continue
                    if current_id == node_id:
                        colors[index] = cls.HIGHLIGHT_SELECTED_COLOR
                        sizes[index] = 30
                    else:
                        colors[index] = cls.HIGHLIGHT_LINEAGE_COLOR
                        sizes[index] = max(default_sizes[index], 20)
            with fig_widget.batch_update():
                fig_widget.data[node_trace_idx].marker.color = tuple(colors)
                fig_widget.data[node_trace_idx].marker.size = tuple(sizes)

        def on_click(trace, points, _selector) -> None:
            if not points.point_inds:
                return
            selected_idx = points.point_inds[0]
            selected["index"] = selected_idx
            apply_highlight(selected_idx)
            node_id = node_ids[selected_idx]
            lineage_nodes = cls.collect_lineage_nodes(graph, node_id)
            detail.value = (
                cls._node_detail_html(graph, node_id)
                + f"<p><i>Highlighted lineage: {len(lineage_nodes)} node(s)</i></p>"
            )

        fig_widget.data[node_trace_idx].on_click(on_click)

        reset_btn = widgets.Button(description="Clear selection", layout=widgets.Layout(width="140px"))
        reset_btn.on_click(
            lambda _btn: (
                selected.update(index=None),
                apply_highlight(None),
                detail.__setattr__(
                    "value",
                    "<p><b>Click a node</b> to highlight its full lineage. "
                    "Drag to pan, scroll to zoom.</p>",
                ),
            )
        )
        display(widgets.VBox([fig_widget, widgets.HBox([reset_btn]), detail]))
        return graph

    @staticmethod
    def _display_interactive_html(html_doc: str, width: str = "100%", height: str = "780px") -> None:
        """Embed interactive HTML in Jupyter (IPython IFrame has no srcdoc support)."""
        try:
            from IPython.display import HTML, display
        except ImportError as exc:
            raise RuntimeError(
                "Interactive visualization requires IPython (Jupyter). "
                "Use to_interactive_html() and open the HTML in a browser."
            ) from exc

        srcdoc = html.escape(html_doc, quote=True)
        display(
            HTML(
                f'<iframe width="{width}" height="{height}" '
                f'srcdoc="{srcdoc}" '
                f'sandbox="allow-scripts allow-same-origin" '
                f'frameborder="0" style="border:0;width:100%;"></iframe>'
            )
        )

    @classmethod
    def show_interactive(
        cls,
        graph_json: Dict[str, Any],
        height: str = "780px",
        title: str = "SQL2Graph Column Lineage",
        backend: str = "plotly",
    ) -> nx.MultiDiGraph:
        """Display an interactive graph in Jupyter (click nodes for details)."""
        if backend == "html":
            graph = cls.graph_from_node_link(graph_json)
            html_doc = cls.to_interactive_html(graph_json, height=height, title=title)
            cls._display_interactive_html(html_doc, height=height)
            return graph
        try:
            return cls._display_plotly_interactive(graph_json, title=title, height=height)
        except (ImportError, RuntimeError) as exc:
            import warnings

            warnings.warn(
                f"Plotly widget backend unavailable ({exc}); falling back to HTML viewer.",
                stacklevel=2,
            )
            graph = cls.graph_from_node_link(graph_json)
            html_doc = cls.to_interactive_html(graph_json, height=height, title=title)
            cls._display_interactive_html(html_doc, height=height)
            return graph

    @classmethod
    def explore(
        cls,
        result: Dict[str, Any],
        height: str = "780px",
        backend: str = "plotly",
    ) -> None:
        """Notebook explorer: switch between full graph and subgraphs interactively."""
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError as exc:
            raise RuntimeError("explore() requires ipywidgets (installed with Jupyter).") from exc

        if "error" in result:
            raise ValueError(result.get("error", "Pipeline result contains an error."))

        graph_options: List[Tuple[str, Dict[str, Any]]] = [("Full graph", result["graph"])]
        for index, subgraph in enumerate(result.get("subgraphs", [])):
            label = f"[{index}] {subgraph.get('type')} / {subgraph.get('name')}"
            graph_options.append((label, subgraph.get("graph") or {"nodes": [], "links": []}))

        graph_output = widgets.Output(
            layout=widgets.Layout(width="100%", overflow="visible"),
        )
        dropdown = widgets.Dropdown(
            options=[(label, idx) for idx, (label, _) in enumerate(graph_options)],
            value=0,
            description="View:",
            layout=widgets.Layout(width="70%"),
        )
        summary = widgets.HTML(
            value=(
                "<p><b>Interactive lineage explorer</b> — pick a graph, click nodes for details, "
                "drag to pan, scroll to zoom.</p>"
            )
        )

        def refresh(_=None) -> None:
            label, graph_json = graph_options[dropdown.value]
            with graph_output:
                graph_output.clear_output(wait=True)
                if not graph_json.get("nodes"):
                    display(widgets.HTML(f"<p><b>{html.escape(label)}</b> — no mapped nodes in this subgraph yet.</p>"))
                    return
                if backend == "html":
                    cls._display_interactive_html(
                        cls.to_interactive_html(graph_json, height=height, title=label),
                        height=height,
                    )
                else:
                    cls._display_plotly_interactive(graph_json, title=label, height=height)

        dropdown.observe(refresh, names="value")
        refresh()
        display(widgets.VBox([summary, dropdown, graph_output]))


class SQL2GraphValidator:
    """Deterministic checks for extraction payload and graph integrity."""

    @staticmethod
    def validate_extraction(extraction: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            SQL2GraphExtraction.model_validate(extraction)
            return True, "valid"
        except ValidationError as exc:
            return False, str(exc)

    @staticmethod
    def validate_graph(graph: nx.MultiDiGraph, schema: Optional[Dict[str, Any]] = None) -> List[str]:
        warnings = []
        for source, target, _ in graph.edges(data=True):
            if source not in graph.nodes:
                warnings.append(f"Dangling edge source: {source}")
            if target not in graph.nodes:
                warnings.append(f"Dangling edge target: {target}")

        if schema and isinstance(schema, dict):
            alias_columns: Dict[str, set] = {}
            for table in schema.get("tables", []):
                alias = table.get("alias") or table.get("name")
                cols = {col.get("name") for col in table.get("columns", []) if col.get("name")}
                if alias:
                    alias_columns[alias] = cols

            for node, attrs in graph.nodes(data=True):
                if attrs.get("node_type") != "source_column":
                    continue
                alias = attrs.get("table_alias")
                column = attrs.get("column")
                if alias in alias_columns and column not in alias_columns[alias]:
                    warnings.append(f"Unknown column reference: {node}")

        if graph.number_of_edges() > 0 and not nx.is_directed_acyclic_graph(graph):
            try:
                cycle = nx.find_cycle(graph)
                warnings.append(f"Graph contains a directed cycle: {cycle[:3]}")
            except nx.NetworkXNoCycle:
                pass

        return warnings


class SQL2GraphPipeline:
    """End-to-end SQL-to-column-lineage graph pipeline.

    Stages:
      1. chunking  — split SQL into logical chunks (CTEs, UNION branches, target)
      2. parsing   — sqlglot simplify + deterministic column extraction
      3. verifying — optional LLM review of the sqlglot draft
      4. enhancing — optional LLM targeted repairs on the verified draft
      5. combining — merge extraction, build lineage graph, validate
    """

    PIPELINE_STEP_ORDER = ("chunking", "parsing", "verifying", "enhancing", "combining")

    def __init__(
        self,
        llm_extractor: Optional["SQL2GraphLLMExtractor"] = None,
        parser: Optional[SQL2GraphParser] = None,
        builder: Optional[SQL2GraphBuilder] = None,
        validator: Optional[SQL2GraphValidator] = None,
        chunk_parser: Optional[Any] = None,
    ):
        self.llm_extractor = llm_extractor
        self.parser = parser or SQL2GraphParser()
        self.builder = builder or SQL2GraphBuilder()
        self.validator = validator or SQL2GraphValidator()
        self.chunk_parser = chunk_parser

    @staticmethod
    def _collect_alias_columns_from_sql(sql_text: str) -> List[str]:
        pairs = re.findall(r'([A-Za-z_][\w\$]*)\.(?:"([^"]+)"|([A-Za-z_][\w\$]*))', sql_text or "")
        return [f"{alias}.{quoted or plain}" for alias, quoted, plain in pairs]

    @staticmethod
    def _cte_alias_map(extracted: Dict[str, Any], simplified: Dict[str, Any]) -> Dict[str, str]:
        """Map table aliases used in the main query to the CTE names they refer to."""
        cte_names = {
            str(cte.get("alias", "")).strip().lower(): str(cte.get("alias", "")).strip()
            for cte in extracted.get("ctes", [])
            if cte.get("alias")
        }
        if not cte_names or not simplified.get("parser_used"):
            return {}

        candidates = list(simplified.get("from", []) or [])
        for join in simplified.get("joins", []) or []:
            candidates.append({"table": join.get("right_table"), "alias": join.get("alias")})

        alias_map: Dict[str, str] = {}
        for item in candidates:
            table = str(item.get("table") or "").strip().strip('"').lower()
            alias = str(item.get("alias") or "").strip()
            if alias and table in cte_names and alias.lower() != table:
                alias_map[alias] = cte_names[table]
        return alias_map

    def _build_subgraphs(
        self,
        simplified: Dict[str, Any],
        graph: nx.MultiDiGraph,
    ) -> List[Dict[str, Any]]:
        """Build subgraph payloads for CTEs, JOIN blocks, and UNION branches."""
        subgraphs: List[Dict[str, Any]] = []
        blocks = simplified.get("subgraph_blocks", [])

        for block in blocks:
            block_type = block.get("type")
            node_candidates = set()

            if block_type == "cte":
                prefix = f"{block.get('name')}."
                node_candidates = {node for node in graph.nodes if str(node).startswith(prefix)}
            elif block_type == "subjoin":
                for ref in block.get("join_columns", []):
                    alias = ref.get("table_alias")
                    column = ref.get("column")
                    if alias and column:
                        node_candidates.add(f"{alias}.{column}")
            elif block_type == "union_block":
                for alias in block.get("select_aliases", []):
                    node_candidates.add(f"output.{alias}")
                for node_id in self._collect_alias_columns_from_sql(block.get("sql", "")):
                    node_candidates.add(node_id)

            existing_nodes = {node for node in node_candidates if node in graph.nodes}
            if existing_nodes:
                subgraph_obj = graph.subgraph(existing_nodes).copy()
                try:
                    subgraph_json = json_graph.node_link_data(subgraph_obj, edges="links")
                except TypeError:
                    subgraph_json = json_graph.node_link_data(subgraph_obj)
            else:
                subgraph_json = {"nodes": [], "links": []}

            subgraphs.append(
                {
                    "id": block.get("id"),
                    "type": block_type,
                    "name": block.get("name"),
                    "sql": block.get("sql", ""),
                    "graph": subgraph_json,
                }
            )

        return subgraphs

    @staticmethod
    def _build_metadata(sql: str) -> Dict[str, Any]:
        """Attach spec section 5 metadata to graph payloads."""
        return {
            "source_sql_hash": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "spec_version": "2.1",
            "implementation_profile": "column_level_v1",
        }

    @staticmethod
    def _step_status(step: str, status: str, **details: Any) -> Dict[str, Any]:
        return {"step": step, "status": status, **details}

    @staticmethod
    def _emit_step(
        pipeline_steps: Dict[str, Dict[str, Any]],
        step_name: str,
        status: str,
        step_callback: Optional[Callable[[str, Dict[str, Any], Dict[str, Dict[str, Any]]], None]] = None,
        **details: Any,
    ) -> None:
        pipeline_steps[step_name] = SQL2GraphPipeline._step_status(step_name, status, **details)
        if step_callback is not None:
            step_callback(step_name, pipeline_steps[step_name], dict(pipeline_steps))

    @staticmethod
    def diff_extraction(
        before: Optional[Dict[str, Any]],
        after: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Summarize structural differences between two extraction payloads."""
        before = before or {}
        after = after or {}
        changes: List[Dict[str, Any]] = []

        before_cols = {
            str(col.get("alias", "")).strip(): col
            for col in before.get("output_columns", [])
            if col.get("alias")
        }
        after_cols = {
            str(col.get("alias", "")).strip(): col
            for col in after.get("output_columns", [])
            if col.get("alias")
        }

        for alias in sorted(set(before_cols) | set(after_cols)):
            left = before_cols.get(alias)
            right = after_cols.get(alias)
            if left is None:
                changes.append({"area": "output_column", "alias": alias, "change": "added"})
                continue
            if right is None:
                changes.append({"area": "output_column", "alias": alias, "change": "removed"})
                continue
            for field in ("expression", "dependencies", "derivation_kind", "literal_values"):
                if left.get(field) != right.get(field):
                    changes.append(
                        {
                            "area": "output_column",
                            "alias": alias,
                            "field": field,
                            "before": left.get(field),
                            "after": right.get(field),
                        }
                    )

        for area in ("filters", "joins", "ctes"):
            before_count = len(before.get(area) or [])
            after_count = len(after.get(area) or [])
            if before_count != after_count:
                changes.append(
                    {
                        "area": area,
                        "change": "count_changed",
                        "before": before_count,
                        "after": after_count,
                    }
                )

        return {
            "change_count": len(changes),
            "changes": changes,
        }

    def _chunk_parser(self):
        if self.chunk_parser is not None:
            return self.chunk_parser
        from Classes.sql_chunk_classes import SQLLogicalChunkParser, SQLLogicalChunkPreParser

        return SQLLogicalChunkParser(
            pre_parser=SQLLogicalChunkPreParser(parser=self.parser),
        )

    def run(
        self,
        sql: str,
        schema: Optional[Dict[str, Any]] = None,
        dialect: Optional[str] = None,
        include_visualization: bool = False,
        use_llm_verify: bool = True,
        use_llm_enhance: bool = True,
        step_callback: Optional[Callable[[str, Dict[str, Any], Dict[str, Dict[str, Any]]], None]] = None,
    ) -> Dict[str, Any]:
        pipeline_steps: Dict[str, Dict[str, Any]] = {}
        warnings: List[str] = []
        pipeline_stage = "deterministic"
        verification_diff: Optional[Dict[str, Any]] = None
        enhancement_diff: Optional[Dict[str, Any]] = None

        # Step 1: chunking
        self._emit_step(pipeline_steps, "chunking", "running", step_callback)
        try:
            chunk_result = self._chunk_parser().preparse(sql, dialect=dialect)
            self._emit_step(
                pipeline_steps,
                "chunking",
                "completed",
                step_callback,
                chunk_count=len(chunk_result.get("chunks") or []),
                link_count=len(chunk_result.get("links") or []),
                statement_type=chunk_result.get("statement_type"),
                target_table=chunk_result.get("target_table"),
            )
        except Exception as exc:
            self._emit_step(pipeline_steps, "chunking", "failed", step_callback, error=str(exc))
            chunk_result = {"chunks": [], "links": []}
            warnings.append(f"Chunking failed: {exc}")

        # Step 2: parsing (sqlglot deterministic extraction)
        self._emit_step(pipeline_steps, "parsing", "running", step_callback)
        simplified = self.parser.simplify(sql, dialect=dialect)
        if not simplified.get("parser_used"):
            self._emit_step(
                pipeline_steps,
                "parsing",
                "failed",
                step_callback,
                error=simplified.get("parse_error") or "sqlglot could not parse the SQL.",
            )
            return {
                "error": simplified.get("parse_error") or "sqlglot could not parse the SQL.",
                "pipeline_steps": pipeline_steps,
                "chunks": chunk_result,
                "simplified_query": simplified,
            }

        deterministic = self.parser.build_deterministic_extraction(simplified, dialect=dialect)
        self._emit_step(
            pipeline_steps,
            "parsing",
            "completed",
            step_callback,
            output_column_count=len(deterministic.get("output_columns") or []),
            cte_count=len(deterministic.get("ctes") or []),
            target_table=simplified.get("target_table"),
        )

        extracted = deterministic
        verify_failed = False

        # Step 3: verifying (optional LLM)
        if use_llm_verify and self.llm_extractor is not None:
            self._emit_step(pipeline_steps, "verifying", "running", step_callback)
            verified_payload = self.llm_extractor.verify(
                sql=sql,
                deterministic_draft=deterministic,
                schema=schema,
                simplified_query=simplified,
            )
            if "error" in verified_payload:
                verify_failed = True
                warnings.append(str(verified_payload.get("error")))
                if verified_payload.get("details"):
                    warnings.append(str(verified_payload["details"]))
                extracted = deterministic
                pipeline_stage = "deterministic_fallback"
                self._emit_step(
                    pipeline_steps,
                    "verifying",
                    "fallback",
                    step_callback,
                    message="LLM verification failed; using sqlglot draft.",
                )
            else:
                verification_diff = self.diff_extraction(deterministic, verified_payload)
                extracted = verified_payload
                pipeline_stage = "llm_verified"
                self._emit_step(
                    pipeline_steps,
                    "verifying",
                    "completed",
                    step_callback,
                    change_count=verification_diff["change_count"],
                    diff=verification_diff,
                )
        else:
            reason = "LLM verification disabled."
            if use_llm_verify and self.llm_extractor is None:
                reason = "No LLM extractor configured."
            self._emit_step(pipeline_steps, "verifying", "skipped", step_callback, message=reason)

        # Step 4: enhancing (optional LLM)
        if use_llm_enhance and self.llm_extractor is not None and not verify_failed:
            self._emit_step(pipeline_steps, "enhancing", "running", step_callback)
            before_enhance = copy.deepcopy(extracted)
            if not self.llm_extractor.enable_refinement:
                self._emit_step(
                    pipeline_steps,
                    "enhancing",
                    "skipped",
                    step_callback,
                    message="LLM refinement disabled on extractor.",
                )
            else:
                enhanced = self.llm_extractor.enhance(
                    sql=sql,
                    verified_payload=extracted,
                    schema=schema,
                    simplified_query=simplified,
                )
                if "error" in enhanced:
                    warnings.append(str(enhanced.get("error")))
                    if enhanced.get("details"):
                        warnings.append(str(enhanced["details"]))
                    self._emit_step(
                        pipeline_steps,
                        "enhancing",
                        "fallback",
                        step_callback,
                        message="LLM enhancement failed; using previous draft.",
                    )
                else:
                    enhancement_diff = self.diff_extraction(before_enhance, enhanced)
                    extracted = enhanced
                    pipeline_stage = "llm_enhanced"
                    self._emit_step(
                        pipeline_steps,
                        "enhancing",
                        "completed",
                        step_callback,
                        change_count=enhancement_diff["change_count"],
                        diff=enhancement_diff,
                    )
        else:
            if verify_failed:
                message = "Skipped because verification failed."
            elif not use_llm_enhance:
                message = "LLM enhancement disabled."
            elif self.llm_extractor is None:
                message = "No LLM extractor configured."
            else:
                message = "LLM enhancement disabled."
            self._emit_step(pipeline_steps, "enhancing", "skipped", step_callback, message=message)

        if (use_llm_verify or use_llm_enhance) and self.llm_extractor is not None and pipeline_stage != "deterministic_fallback":
            extracted = copy.deepcopy(extracted)

        extracted = self.parser.overlay_deterministic_column_lineage(extracted, deterministic)
        extracted = self.parser._materialize_output_dependencies(extracted, simplified)

        try:
            SQL2GraphExtraction.model_validate(extracted)
        except ValidationError as exc:
            self._emit_step(pipeline_steps, "combining", "failed", step_callback, error=str(exc))
            return {
                "error": "Lineage extraction validation failed",
                "details": str(exc),
                "deterministic_extraction": deterministic,
                "simplified_query": simplified,
                "pipeline_steps": pipeline_steps,
                "chunks": chunk_result,
            }

        # Step 5: combining (graph build + validation)
        self._emit_step(pipeline_steps, "combining", "running", step_callback)
        graph = self.builder.build(extracted)
        self.builder.link_cte_aliases(self._cte_alias_map(extracted, simplified))
        self.builder.materialize_transitive_derived_from()
        dag_warnings = self.builder.ensure_acyclic()
        validation_warnings = self.validator.validate_graph(graph, schema=schema)
        warnings.extend(validation_warnings)
        warnings.extend(dag_warnings)
        graph_payload = self.builder.to_node_link()
        graph_payload["metadata"] = self._build_metadata(sql)
        graph_payload["metadata"]["is_dag"] = nx.is_directed_acyclic_graph(self.builder.graph)
        self._emit_step(
            pipeline_steps,
            "combining",
            "completed",
            step_callback,
            node_count=len(graph_payload.get("nodes") or []),
            edge_count=len(graph_payload.get("links") or []),
            is_dag=graph_payload["metadata"]["is_dag"],
        )

        response = {
            "graph": graph_payload,
            "metadata": graph_payload["metadata"],
            "warnings": warnings,
            "extraction": extracted,
            "deterministic_extraction": deterministic,
            "pipeline_stage": pipeline_stage,
            "pipeline_steps": pipeline_steps,
            "verification_diff": verification_diff,
            "enhancement_diff": enhancement_diff,
            "chunks": chunk_result,
            "chunk_graph": self._chunk_parser().to_node_link(chunk_result),
            "simplified_query": simplified,
            "subgraphs": self._build_subgraphs(simplified, graph),
        }
        if include_visualization:
            response["visualization"] = {
                "mermaid": self.builder.to_mermaid(),
                "dot": self.builder.to_dot(),
            }
        return response
