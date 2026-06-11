import hashlib
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
from pydantic import BaseModel, Field, ValidationError, field_validator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from Classes.helper_classes import HuggingFaceLLMAdapter

try:
    import sqlglot  # type: ignore[import-not-found]
    from sqlglot import exp  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    sqlglot = None
    exp = None


class ColumnRef(BaseModel):
    table_alias: Optional[str] = None
    column: str

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

    def __init__(self):
        self.sqlglot_available = sqlglot is not None

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
        try:
            tree = sqlglot.parse_one(probe_sql, read=dialect)
        except Exception:
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

    def simplify(self, sql: str, dialect: Optional[str] = None) -> Dict[str, Any]:
        if not self.sqlglot_available:
            return {"raw_sql": sql, "parser_used": False, "subgraph_blocks": []}

        tree = sqlglot.parse_one(sql, read=dialect)
        if not isinstance(tree, exp.Select) and not tree.find(exp.Select):
            return {"raw_sql": sql, "parser_used": True, "subgraph_blocks": []}

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
        }


class SQL2GraphLLMExtractor:
    """LLM-backed extractor for column-level lineage JSON."""

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        provider: str = "scaleway",
        hf_token: Optional[str] = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        max_retries: int = 3,
        enable_refinement: bool = True,
    ):
        if not hf_token:
            raise ValueError("HF_TOKEN is required for SQL2Graph extraction.")

        self.model = model
        self.provider = provider
        self.max_retries = max_retries
        self.enable_refinement = enable_refinement
        self.chat_model = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id=model,
                task="text-generation",
                provider=provider,
                huggingfacehub_api_token=hf_token,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
            )
        )
        self.chat_adapter = HuggingFaceLLMAdapter(self.chat_model)

        self.system_prompt = (
            "You are a SQL lineage expert. Return ONLY valid JSON for column-level lineage with keys: "
            "ctes, output_columns, filters, joins, and optionally group_by_columns. "
            "Each output column must include alias, expression, dependencies. "
            "Work step by step internally: first list all table aliases and their source tables, "
            "then analyse the SELECT list column by column, then extract filters and joins. "
            "Output only the final JSON."
        )
        self.refinement_system_prompt = (
            "You are a strict SQL lineage reviewer. Improve a draft lineage JSON using the original SQL. "
            "Return ONLY corrected JSON with the same top-level keys. "
            "Fix missing or weak fields: filter.condition, joins.join_columns, output dependencies, "
            "and ensure ctes are recursively valid."
        )

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
        response_text = self._invoke_messages_text(
            [
                SystemMessage(content=self.refinement_system_prompt),
                HumanMessage(content=prompt),
            ]
        )
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

    def extract(
        self,
        sql: str,
        schema: Optional[Dict[str, Any]] = None,
        simplified_query: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        last_validation_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                user_prompt = self._build_user_prompt(
                    sql=sql,
                    schema=schema,
                    simplified_query=simplified_query,
                    validation_error=last_validation_error,
                )
                response_text = self._invoke_messages_text(
                    [SystemMessage(content=self.system_prompt), HumanMessage(content=user_prompt)]
                )
                payload = self._extract_json(response_text)
                payload = self._normalize_scope_payload(payload)
                validated = SQL2GraphExtraction.model_validate(payload)
                extracted = validated.model_dump()

                # Second LLM pass for higher quality lineage on complex SQL.
                if self.enable_refinement:
                    try:
                        refined_payload = self._refine_payload_with_llm(
                            sql=sql,
                            schema=schema,
                            simplified_query=simplified_query,
                            draft_payload=extracted,
                        )
                        refined_validated = SQL2GraphExtraction.model_validate(refined_payload)
                        return refined_validated.model_dump()
                    except Exception:
                        # Keep first validated extraction if refinement fails.
                        return extracted

                return extracted
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
                    self.graph.add_edge(out_id, grp_node, edge_type="GROUPED_BY")

        for filt in scope.get("filters", []):
            f = FilterSpec.model_validate(filt)
            filter_node = self._add_filter_node(f.clause, f.condition)
            for used in f.columns_used:
                col_node = self._add_source_column(used)
                self.graph.add_edge(filter_node, col_node, edge_type="USES_COLUMN")
            for out in output_nodes:
                self.graph.add_edge(filter_node, out, edge_type="FILTERED_BY")

        for join in scope.get("joins", []):
            j = JoinSpec.model_validate(join)
            left = self._add_source_column(j.join_columns[0])
            right = self._add_source_column(j.join_columns[1])
            self.graph.add_edge(left, right, edge_type="JOINS_ON", join_type=j.type, condition=j.condition)
            self.graph.add_edge(right, left, edge_type="JOINS_ON", join_type=j.type, condition=j.condition)

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
        (spec section 3.3.2 rule 4 / appendix example).
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
    }

    NODE_COLORS = {
        "source_column": "#90EE90",
        "output_column": "#ADD8E6",
        "filter": "#F6D186",
        "join": "#F08080",
    }

    @staticmethod
    def graph_from_node_link(graph_json: Dict[str, Any]) -> nx.MultiDiGraph:
        # Support both historic "links" and newer "edges".
        if "links" in graph_json:
            try:
                return json_graph.node_link_graph(graph_json, edges="links")
            except TypeError:
                return json_graph.node_link_graph(graph_json)

        if "edges" in graph_json and "links" not in graph_json:
            normalized = dict(graph_json)
            normalized["links"] = normalized.get("edges", [])
            return json_graph.node_link_graph(normalized)

        return json_graph.node_link_graph(graph_json)

    @classmethod
    def draw(
        cls,
        graph_json: Dict[str, Any],
        figsize: Tuple[int, int] = (16, 10),
        with_labels: bool = True,
        layout: str = "spring",
    ):
        graph = cls.graph_from_node_link(graph_json)
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot visualize empty graph.")

        if layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(graph)
        elif layout == "shell":
            pos = nx.shell_layout(graph)
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

        plt.title("SQL2Graph Column Lineage")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        return graph


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

        return warnings


class SQL2GraphPipeline:
    """End-to-end SQL-to-column-lineage graph pipeline."""

    def __init__(
        self,
        llm_extractor: SQL2GraphLLMExtractor,
        parser: Optional[SQL2GraphParser] = None,
        builder: Optional[SQL2GraphBuilder] = None,
        validator: Optional[SQL2GraphValidator] = None,
    ):
        self.llm_extractor = llm_extractor
        self.parser = parser or SQL2GraphParser()
        self.builder = builder or SQL2GraphBuilder()
        self.validator = validator or SQL2GraphValidator()

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

    def run(
        self,
        sql: str,
        schema: Optional[Dict[str, Any]] = None,
        dialect: Optional[str] = None,
        include_visualization: bool = False,
    ) -> Dict[str, Any]:
        simplified = self.parser.simplify(sql, dialect=dialect)
        extracted = self.llm_extractor.extract(sql=sql, schema=schema, simplified_query=simplified)
        if "error" in extracted:
            return extracted

        graph = self.builder.build(extracted)
        self.builder.link_cte_aliases(self._cte_alias_map(extracted, simplified))
        warnings = self.validator.validate_graph(graph, schema=schema)
        response = {
            "graph": self.builder.to_node_link(),
            "warnings": warnings,
            "extraction": extracted,
            "simplified_query": simplified,
            "subgraphs": self._build_subgraphs(simplified, graph),
        }
        if include_visualization:
            response["visualization"] = {
                "mermaid": self.builder.to_mermaid(),
                "dot": self.builder.to_dot(),
            }
        return response
