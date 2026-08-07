"""Column-level lineage for PL/pgSQL function bodies.

The extractor splits a routine into individual SQL statements
(:mod:`Classes.plpgsql_splitter`), runs each one through the existing
sqlglot-based :class:`~Classes.sql2graph_classes.SQL2GraphParser`, and merges
the per-statement graphs into a single node-link payload.

Two properties make the merged graph useful across statement boundaries:

* Column nodes are keyed by *physical* table rather than by local alias, so a
  temp table written by one statement and read by the next resolves to the
  same node and the lineage chain stays connected.
* Every branch of ``IF``/``CASE`` and every loop body is included. The result
  is a conservative superset of any single invocation.

Anything that cannot be resolved statically — dynamic ``EXECUTE``, unparsable
fragments, recursive calls — is reported in ``unresolved`` rather than guessed
at, and any edge recovered from a dynamic statement is marked with
``provenance="unresolved"`` and low confidence.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph

from Classes.plpgsql_splitter import (
    PlpgsqlStmt,
    find_function_defs,
    is_plpgsql_function,
    split_function_body,
)
from Classes.schema_registry import SchemaRegistry
from Classes.sql2graph_classes import (
    SQL2GraphBuilder,
    SQL2GraphParser,
    SQL2GraphValidator,
)
from Classes.table_lineage import extract_table_lineage

try:
    import sqlglot
    from sqlglot import exp
except Exception:  # pragma: no cover - sqlglot is a hard dependency in practice
    sqlglot = None
    exp = None

__all__ = [
    "PlpgsqlLineageExtractor",
    "UnresolvedItem",
    "contains_plpgsql_function",
    "extract_plpgsql_lineage",
]

#: Confidence assigned to edges recovered from dynamic SQL.
DYNAMIC_CONFIDENCE = 0.3
DYNAMIC_PROVENANCE = "unresolved"


@dataclass
class UnresolvedItem:
    """A statement the extractor refused to resolve, with the reason why."""

    reason: str
    detail: str
    sql_fragment: str
    kind: str = ""
    line_start: int = 0
    line_end: int = 0
    function: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _FunctionScope:
    """Per-function bookkeeping carried while walking a routine."""

    name: str
    temp_tables: set[str] = field(default_factory=set)
    variables: set[str] = field(default_factory=set)


#: `SELECT ... INTO var` assigns to a PL/pgSQL variable; in plain SQL the same
#: syntax means CREATE TABLE AS, so it is stripped before handing SQL to sqlglot.
_SELECT_INTO_RE = re.compile(
    r"\s+INTO\s+(?:STRICT\s+)?(?:\"[^\"]+\"|[A-Za-z_]\w*)(?:\s*,\s*(?:\"[^\"]+\"|[A-Za-z_]\w*))*",
    re.IGNORECASE,
)
_FORMAT_LITERAL_RE = re.compile(r"format\s*\(\s*('(?:[^']|'')*')", re.IGNORECASE)
_FORMAT_PLACEHOLDER_RE = re.compile(r"%[ILs]")
_IDENT_RE = re.compile(r"[A-Za-z_][\w$]*")
#: Placeholder substituted for `%I`/`%s` so a format string can still be parsed.
DYNAMIC_PLACEHOLDER = "dynamic_placeholder"


class PlpgsqlLineageExtractor:
    """Build a lineage graph for the statements inside a PL/pgSQL routine."""

    def __init__(
        self,
        schema_registry: SchemaRegistry | None = None,
        dialect: str = "postgres",
        parser: SQL2GraphParser | None = None,
        max_depth: int = 5,
    ):
        self.dialect = dialect
        self.schema_registry = schema_registry
        self.parser = parser or SQL2GraphParser(dialect=dialect, schema_registry=schema_registry)
        self.validator = SQL2GraphValidator()
        self.max_depth = max_depth

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, create_function_sql: str) -> dict[str, Any]:
        """Extract lineage for the first PL/pgSQL routine in ``create_function_sql``.

        Additional routines in the same text are registered so that calls
        between them can be followed (bounded by ``max_depth``).
        """
        defs = find_function_defs(create_function_sql)
        if not defs:
            return self._empty_result(
                create_function_sql,
                error=(
                    "No PL/pgSQL function definition found: expected "
                    "`CREATE [OR REPLACE] FUNCTION ... AS $$ ... $$ LANGUAGE plpgsql`"
                ),
            )

        self._registry: dict[str, str] = dict(defs)
        self._graph = nx.MultiDiGraph()
        self._unresolved: list[UnresolvedItem] = []
        self._statements: list[dict[str, Any]] = []
        self._table_lineage: list[dict[str, Any]] = []
        self._temp_tables: set[str] = set()
        self._variables: set[str] = set()
        self._warnings: list[str] = []
        self._visiting: list[str] = []

        primary_name, primary_body = defs[0]
        self._walk_function(primary_name, primary_body, depth=0)

        warnings = list(self._warnings)
        warnings.extend(self._break_cycles())
        warnings.extend(self.validator.validate_graph(self._graph))

        payload = self._to_node_link()
        payload["metadata"] = self._build_metadata(create_function_sql, primary_name)

        return {
            "function": primary_name,
            "functions": [name for name, _ in defs],
            "graph": payload,
            "metadata": payload["metadata"],
            "pipeline_stage": "plpgsql",
            "statements": self._statements,
            "unresolved": [item.to_dict() for item in self._unresolved],
            "temp_tables": sorted(self._temp_tables),
            "variables": sorted(self._variables),
            "table_lineage": self._table_lineage,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Function walking
    # ------------------------------------------------------------------

    def _walk_function(self, name: str, body: str, *, depth: int) -> None:
        if depth > self.max_depth:
            self._unresolved.append(
                UnresolvedItem(
                    reason="max_depth_exceeded",
                    detail=f"Call depth limit of {self.max_depth} reached at {name}",
                    sql_fragment="",
                    function=name,
                )
            )
            return

        self._visiting.append(name)
        scope = _FunctionScope(name=name)
        try:
            for index, stmt in enumerate(split_function_body(body)):
                self._handle_statement(stmt, scope, index, depth)
        finally:
            self._visiting.pop()

    def _handle_statement(
        self,
        stmt: PlpgsqlStmt,
        scope: _FunctionScope,
        index: int,
        depth: int,
    ) -> None:
        if stmt.kind == "unknown":
            self._unresolved.append(self._unresolved_from(stmt, "unsupported_statement", "Unrecognised statement"))
            return

        if stmt.kind in {"assign", "declare"} and stmt.into:
            scope.variables.add(stmt.into)
            self._variables.add(stmt.into)
            # Only initialisers that are real queries carry lineage.
            if not self._looks_like_query(stmt.sql):
                return
        elif not stmt.is_lineage_bearing:
            return

        self._follow_calls(stmt, scope, depth)

        sql = stmt.sql
        if stmt.is_dynamic:
            resolved = self._resolve_dynamic(stmt)
            if resolved is None:
                self._unresolved.append(
                    self._unresolved_from(stmt, "dynamic_execute", stmt.dynamic_reason or "Dynamic SQL")
                )
                return
            sql = resolved
            self._unresolved.append(
                self._unresolved_from(
                    stmt,
                    "dynamic_execute",
                    f"{stmt.dynamic_reason or 'Dynamic SQL'}; recovered a best-effort parse",
                )
            )

        if stmt.kind == "select" and stmt.into:
            sql = _SELECT_INTO_RE.sub("", sql, count=1)

        self._ingest_sql(sql, stmt, scope, index)

    def _follow_calls(self, stmt: PlpgsqlStmt, scope: _FunctionScope, depth: int) -> None:
        """Inline lineage of other PL/pgSQL routines this statement invokes."""
        for candidate in self._called_functions(stmt.sql):
            if candidate in self._visiting:
                self._unresolved.append(
                    self._unresolved_from(
                        stmt,
                        "recursive_call",
                        f"{candidate} is already on the call stack "
                        f"({' -> '.join(self._visiting)}); not expanded again",
                    )
                )
                continue
            self._walk_function(candidate, self._registry[candidate], depth=depth + 1)

    def _called_functions(self, sql: str) -> list[str]:
        if not sql or not self._registry:
            return []
        lowered = sql.lower()
        found = []
        for name in self._registry:
            bare = name.split(".")[-1]
            for candidate in {name, bare}:
                if re.search(rf"\b{re.escape(candidate)}\s*\(", lowered):
                    found.append(name)
                    break
        return found

    # ------------------------------------------------------------------
    # Statement ingestion
    # ------------------------------------------------------------------

    def _ingest_sql(self, sql: str, stmt: PlpgsqlStmt, scope: _FunctionScope, index: int) -> None:
        table_info = extract_table_lineage(sql, dialect=self.dialect)
        if table_info.get("parser_used") and (table_info.get("target") or table_info.get("sources")):
            target_name = str(table_info.get("target") or "")
            self._table_lineage.append(
                {
                    # A placeholder target means the real name is only known at
                    # runtime, so report it as unknown rather than inventing one.
                    "target": "" if target_name == DYNAMIC_PLACEHOLDER else target_name,
                    "sources": table_info.get("sources"),
                    "statement_type": table_info.get("statement_type"),
                    "line_start": stmt.line_start,
                    "is_dynamic": stmt.is_dynamic,
                }
            )

        extraction, target = self._extract_columns(sql, stmt)
        if extraction is None:
            self._unresolved.append(
                self._unresolved_from(stmt, "parse_failed", "sqlglot could not parse the statement")
            )
            self._record_statement(stmt, target="", resolved=False)
            return

        prefix = self._output_prefix(target, stmt, scope, index)
        if stmt.kind == "create_temp" and target:
            self._temp_tables.add(target)

        self._merge_extraction(extraction, prefix, stmt, is_temp=target in self._temp_tables)
        self._record_statement(stmt, target=prefix, resolved=True)

    def _extract_columns(
        self, sql: str, stmt: PlpgsqlStmt
    ) -> tuple[dict[str, Any] | None, str]:
        """Return ``(extraction, target_table)`` for a single statement."""
        update = self._update_extraction(sql)
        if update is not None:
            return update

        try:
            simplified = self.parser.simplify(sql, dialect=self.dialect)
        except Exception:
            return None, ""
        if not simplified.get("parser_used"):
            return None, ""

        try:
            extraction = self.parser.build_deterministic_extraction(simplified, dialect=self.dialect)
        except Exception:
            return None, ""

        target = str(simplified.get("target_table") or "").strip()
        if not extraction.get("output_columns") and stmt.kind in {"delete", "perform", "call"}:
            # Read-only statements carry no output columns; sources are still
            # captured in table_lineage above.
            return {"ctes": [], "output_columns": [], "filters": [], "joins": [], "group_by_columns": []}, target
        return extraction, target

    def _update_extraction(self, sql: str) -> tuple[dict[str, Any], str] | None:
        """Column lineage for ``UPDATE`` statements, which the parser skips."""
        if sqlglot is None or exp is None:
            return None
        try:
            tree = sqlglot.parse_one(sql, read=self.dialect)
        except Exception:
            return None
        if not isinstance(tree, exp.Update):
            return None

        target_table = tree.this
        target = self._qualified(target_table)
        alias_map = self._alias_map(tree)

        output_columns: list[dict[str, Any]] = []
        for assignment in tree.expressions or []:
            left = getattr(assignment, "this", None)
            right = getattr(assignment, "expression", None)
            if left is None or right is None:
                continue
            alias = getattr(left, "name", None) or left.sql(dialect=self.dialect)
            dependencies = [
                self._column_ref(column, alias_map) for column in right.find_all(exp.Column)
            ]
            output_columns.append(
                {
                    "alias": alias,
                    "expression": right.sql(dialect=self.dialect),
                    "dependencies": dependencies,
                    "aggregate": False,
                    "window_function": False,
                    "literal_values": (
                        [right.sql(dialect=self.dialect)] if isinstance(right, exp.Literal) else []
                    ),
                }
            )

        filters: list[dict[str, Any]] = []
        where = tree.args.get("where")
        if where is not None:
            condition = where.this.sql(dialect=self.dialect) if where.this else where.sql(dialect=self.dialect)
            filters.append(
                {
                    "clause": "WHERE",
                    "condition": condition,
                    "columns_used": [
                        self._column_ref(column, alias_map) for column in where.find_all(exp.Column)
                    ],
                }
            )

        extraction = {
            "ctes": [],
            "output_columns": output_columns,
            "filters": filters,
            "joins": [],
            "group_by_columns": [],
        }
        return extraction, target

    def _alias_map(self, tree: Any) -> dict[str, str]:
        """Map local table aliases to their physical table names."""
        mapping: dict[str, str] = {}
        for table in tree.find_all(exp.Table):
            physical = self._qualified(table)
            if not physical:
                continue
            alias = table.alias_or_name
            if alias:
                mapping[str(alias).lower()] = physical
            mapping[physical.split(".")[-1]] = physical
        return mapping

    def _qualified(self, table: Any) -> str:
        if table is None or exp is None or not isinstance(table, exp.Table):
            return ""
        parts = [part for part in (table.catalog, table.db, table.name) if part]
        return ".".join(str(part) for part in parts).lower()

    def _column_ref(self, column: Any, alias_map: dict[str, str]) -> dict[str, Any]:
        table_alias = str(getattr(column, "table", "") or "").lower()
        physical = alias_map.get(table_alias, "")
        return {
            "table_alias": physical or table_alias or None,
            "column": column.name,
            "physical_table": physical or None,
        }

    # ------------------------------------------------------------------
    # Graph merging
    # ------------------------------------------------------------------

    def _output_prefix(
        self, target: str, stmt: PlpgsqlStmt, scope: _FunctionScope, index: int
    ) -> str:
        if target == DYNAMIC_PLACEHOLDER:
            # The real relation name lives in a runtime variable.
            return f"unresolved.{scope.name}#s{index}"
        if target:
            return target
        if stmt.into:
            return f"var.{stmt.into}"
        return f"{scope.name}#s{index}"

    def _merge_extraction(
        self,
        extraction: dict[str, Any],
        prefix: str,
        stmt: PlpgsqlStmt,
        *,
        is_temp: bool,
    ) -> None:
        canonical = self._canonicalize(extraction)
        builder = SQL2GraphBuilder()
        try:
            sub = builder.build(canonical)
        except Exception as exc:
            self._unresolved.append(
                self._unresolved_from(stmt, "build_failed", f"Graph build failed: {exc}")
            )
            return

        mapping = {
            node: f"{prefix}.{node[len('output.'):]}"
            for node in sub.nodes
            if node.startswith("output.")
        }
        if mapping:
            sub = nx.relabel_nodes(sub, mapping, copy=True)

        edge_extra: dict[str, Any] = {"statement_line": stmt.line_start}
        if stmt.is_dynamic:
            edge_extra.update(
                {
                    "confidence": DYNAMIC_CONFIDENCE,
                    "provenance": DYNAMIC_PROVENANCE,
                    "transform_type": "dynamic",
                    "sql_fragment": stmt.sql,
                    "verified": False,
                }
            )
        if stmt.context:
            edge_extra["conditional"] = True
            edge_extra["control_flow"] = ",".join(stmt.context)

        self._merge_graph(sub, edge_extra, target_prefix=prefix, is_temp=is_temp)

    def _canonicalize(self, extraction: dict[str, Any]) -> dict[str, Any]:
        """Rewrite local aliases to physical tables so nodes match across statements."""

        def fix_ref(ref: Any) -> Any:
            if not isinstance(ref, dict):
                return ref
            physical = ref.get("physical_table")
            if physical:
                ref = dict(ref)
                ref["table_alias"] = str(physical).lower()
            return ref

        def fix_scope(scope: Any) -> Any:
            if not isinstance(scope, dict):
                return scope
            scope = dict(scope)
            scope["output_columns"] = [
                {**col, "dependencies": [fix_ref(dep) for dep in col.get("dependencies") or []]}
                for col in scope.get("output_columns") or []
            ]
            scope["filters"] = [
                {**f, "columns_used": [fix_ref(c) for c in f.get("columns_used") or []]}
                for f in scope.get("filters") or []
            ]
            scope["joins"] = [
                {**j, "join_columns": [fix_ref(c) for c in j.get("join_columns") or []]}
                for j in scope.get("joins") or []
            ]
            scope["group_by_columns"] = [fix_ref(c) for c in scope.get("group_by_columns") or []]
            scope["ctes"] = [fix_scope(cte) for cte in scope.get("ctes") or []]
            return scope

        return fix_scope(extraction)

    def _merge_graph(
        self,
        sub: nx.MultiDiGraph,
        edge_extra: dict[str, Any],
        *,
        target_prefix: str,
        is_temp: bool,
    ) -> None:
        for node, attrs in sub.nodes(data=True):
            payload = dict(attrs)
            if is_temp and node.startswith(f"{target_prefix}."):
                payload["is_temp"] = True
            if node in self._graph.nodes:
                existing = self._graph.nodes[node]
                merged_type = existing.get("node_type")
                # A column written by one statement and read by another stays
                # an output: it is a real relation, not a bare source.
                if "output_column" in {merged_type, payload.get("node_type")}:
                    merged_type = "output_column"
                    payload["is_intermediate"] = True
                existing.update({k: v for k, v in payload.items() if v not in (None, "")})
                if merged_type:
                    existing["node_type"] = merged_type
            else:
                self._graph.add_node(node, **payload)

        for source, target, data in sub.edges(data=True):
            payload = dict(data)
            payload.update(edge_extra)
            if self._has_equivalent_edge(source, target, payload):
                continue
            self._graph.add_edge(source, target, **payload)

    def _has_equivalent_edge(self, source: str, target: str, payload: dict[str, Any]) -> bool:
        if not self._graph.has_edge(source, target):
            return False
        edge_type = payload.get("edge_type")
        for existing in self._graph.get_edge_data(source, target).values():
            if existing.get("edge_type") == edge_type and existing.get(
                "control_flow"
            ) == payload.get("control_flow"):
                return True
        return False

    def _break_cycles(self) -> list[str]:
        """Drop edges that would make the merged graph cyclic (e.g. ``INSERT INTO t SELECT FROM t``)."""
        warnings: list[str] = []
        while self._graph.number_of_edges() and not nx.is_directed_acyclic_graph(self._graph):
            try:
                cycle = nx.find_cycle(self._graph)
            except nx.NetworkXNoCycle:  # pragma: no cover - guarded by the while condition
                break
            source, target, key = cycle[0]
            edge_type = self._graph.edges[source, target, key].get("edge_type", "")
            self._graph.remove_edge(source, target, key)
            warnings.append(f"Removed cyclic edge: {source} -> {target} ({edge_type})")
        return warnings

    # ------------------------------------------------------------------
    # Dynamic SQL
    # ------------------------------------------------------------------

    def _resolve_dynamic(self, stmt: PlpgsqlStmt) -> str | None:
        """Best-effort recovery of SQL hidden inside a dynamic ``EXECUTE``.

        Only the statically known parts are used: ``format()`` placeholders are
        replaced with a neutral identifier so the surrounding query shape can
        still be parsed. Returns ``None`` when nothing parsable remains.
        """
        match = _FORMAT_LITERAL_RE.search(stmt.sql)
        if not match:
            return None
        literal = match.group(1)[1:-1].replace("''", "'")
        candidate = _FORMAT_PLACEHOLDER_RE.sub(DYNAMIC_PLACEHOLDER, literal).replace("%%", "%")
        if not candidate.strip():
            return None
        if sqlglot is None:
            return None
        try:
            sqlglot.parse_one(candidate, read=self.dialect)
        except Exception:
            return None
        return candidate

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_query(sql: str) -> bool:
        return bool(re.match(r"^\s*\(?\s*(SELECT|WITH)\b", sql or "", re.IGNORECASE))

    def _record_statement(self, stmt: PlpgsqlStmt, *, target: str, resolved: bool) -> None:
        self._statements.append(
            {
                "kind": stmt.kind,
                "sql": stmt.sql,
                "target": target,
                "line_start": stmt.line_start,
                "line_end": stmt.line_end,
                "is_dynamic": stmt.is_dynamic,
                "control_flow": stmt.context,
                "resolved": resolved,
            }
        )

    def _unresolved_from(self, stmt: PlpgsqlStmt, reason: str, detail: str) -> UnresolvedItem:
        return UnresolvedItem(
            reason=reason,
            detail=detail,
            sql_fragment=stmt.sql,
            kind=stmt.kind,
            line_start=stmt.line_start,
            line_end=stmt.line_end,
            function=self._visiting[-1] if self._visiting else "",
        )

    def _to_node_link(self) -> dict[str, Any]:
        try:
            return json_graph.node_link_data(self._graph, edges="links")
        except TypeError:  # pragma: no cover - networkx < 3.4
            return json_graph.node_link_data(self._graph)

    def _build_metadata(self, sql: str, function_name: str) -> dict[str, Any]:
        return {
            "source_sql_hash": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "spec_version": "2.1",
            "implementation_profile": "column_level_v2",
            "pipeline_stage": "plpgsql",
            "function": function_name,
            "is_dag": nx.is_directed_acyclic_graph(self._graph),
            "unresolved_count": len(self._unresolved),
            "limitations": [
                "udf_inputs_only",
                "unnest_best_effort",
                "structs_best_effort",
                "plpgsql_branches_merged",
                "plpgsql_dynamic_sql_best_effort",
            ],
        }

    def _empty_result(self, sql: str, *, error: str) -> dict[str, Any]:
        self._graph = nx.MultiDiGraph()
        self._unresolved = []
        return {
            "function": "",
            "functions": [],
            "graph": {"nodes": [], "links": [], "metadata": self._build_metadata(sql, "")},
            "metadata": self._build_metadata(sql, ""),
            "pipeline_stage": "plpgsql",
            "statements": [],
            "unresolved": [],
            "temp_tables": [],
            "variables": [],
            "table_lineage": [],
            "warnings": [],
            "error": error,
        }


def extract_plpgsql_lineage(
    create_function_sql: str,
    *,
    schema_registry: SchemaRegistry | None = None,
    dialect: str = "postgres",
    max_depth: int = 5,
) -> dict[str, Any]:
    """Convenience wrapper around :class:`PlpgsqlLineageExtractor`."""
    extractor = PlpgsqlLineageExtractor(
        schema_registry=schema_registry, dialect=dialect, max_depth=max_depth
    )
    return extractor.extract(create_function_sql)


def contains_plpgsql_function(sql: str) -> bool:
    """True when ``sql`` declares a PL/pgSQL routine (used for pipeline routing)."""
    return is_plpgsql_function(sql)
