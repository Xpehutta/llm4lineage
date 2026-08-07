"""Split PL/pgSQL function bodies into individual SQL statements.

sqlglot cannot parse procedural PL/pgSQL syntax, so this module performs a
lexical pass that is aware of PostgreSQL quoting rules (dollar quoting,
doubled single quotes, ``E''`` backslash escapes, nested block comments) and
yields the plain SQL statements embedded in a function body. Each extracted
statement can then be handed to the existing sqlglot-based parsers.

The splitter deliberately does not try to understand control flow semantics.
Branches of ``IF``/``CASE`` and loop bodies are all emitted, which keeps the
resulting lineage conservative: every statement that *could* execute is
reported.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

__all__ = [
    "PlpgsqlStmt",
    "extract_function_def",
    "find_function_defs",
    "is_plpgsql_function",
    "split_function_body",
]


#: Statement kinds that map onto a SQL statement we can hand to sqlglot.
LINEAGE_KINDS = frozenset(
    {
        "select",
        "insert",
        "update",
        "delete",
        "execute",
        "perform",
        "create_temp",
        "create_table",
        "call",
        "assign",
    }
)


@dataclass
class PlpgsqlStmt:
    """A single statement extracted from a PL/pgSQL function body.

    ``line_start``/``line_end`` are 1-based and relative to the body text that
    was passed to :func:`split_function_body`.
    """

    kind: str
    sql: str
    line_start: int
    line_end: int
    is_dynamic: bool = False
    dynamic_reason: str = ""
    #: Target variable for ``kind == "assign"`` and ``SELECT ... INTO`` statements.
    into: Optional[str] = None
    #: Control-flow keywords the statement was nested under, outermost first.
    context: List[str] = field(default_factory=list)

    @property
    def is_lineage_bearing(self) -> bool:
        """True when the statement can contribute nodes/edges to a graph."""
        return self.kind in LINEAGE_KINDS


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

_DOLLAR_TAG_RE = re.compile(r"\$([A-Za-z_\u0080-\uffff][A-Za-z0-9_\u0080-\uffff]*)?\$")


def _skip_line_comment(text: str, i: int) -> int:
    end = text.find("\n", i)
    return len(text) if end == -1 else end + 1


def _skip_block_comment(text: str, i: int) -> int:
    """Skip a ``/* ... */`` comment. PostgreSQL allows these to nest."""
    n = len(text)
    depth = 1
    i += 2
    while i < n and depth:
        if text.startswith("/*", i):
            depth += 1
            i += 2
        elif text.startswith("*/", i):
            depth -= 1
            i += 2
        else:
            i += 1
    return i


def _skip_single_quoted(text: str, i: int, *, escape_backslash: bool) -> int:
    """Skip a ``'...'`` literal, honouring ``''`` doubling."""
    n = len(text)
    i += 1
    while i < n:
        ch = text[i]
        if escape_backslash and ch == "\\" and i + 1 < n:
            i += 2
            continue
        if ch == "'":
            if i + 1 < n and text[i + 1] == "'":
                i += 2
                continue
            return i + 1
        i += 1
    return n


def _skip_double_quoted(text: str, i: int) -> int:
    """Skip a ``"..."`` quoted identifier, honouring ``""`` doubling."""
    n = len(text)
    i += 1
    while i < n:
        if text[i] == '"':
            if i + 1 < n and text[i + 1] == '"':
                i += 2
                continue
            return i + 1
        i += 1
    return n


def _skip_dollar_quoted(text: str, i: int) -> Tuple[int, bool]:
    """Skip a ``$tag$ ... $tag$`` literal.

    Returns ``(next_index, matched)``. ``matched`` is False when the ``$`` is
    not actually opening a dollar-quoted string (e.g. the ``$1`` of a
    positional parameter), in which case the caller should just advance by one.
    """
    match = _DOLLAR_TAG_RE.match(text, i)
    if not match:
        return i + 1, False
    tag = match.group(0)
    close = text.find(tag, match.end())
    if close == -1:
        return len(text), True
    return close + len(tag), True


def _is_escape_string_start(text: str, i: int) -> bool:
    """True when the quote at ``i`` belongs to an ``E'...'`` escape string."""
    j = i - 1
    if j < 0 or text[j] not in "Ee":
        return False
    # The E must not be part of a longer identifier such as `TRUE'`.
    return j == 0 or not (text[j - 1].isalnum() or text[j - 1] == "_")


def _line_starts(text: str) -> List[int]:
    starts = [0]
    for idx, ch in enumerate(text):
        if ch == "\n":
            starts.append(idx + 1)
    return starts


def _line_of(offset: int, starts: List[int]) -> int:
    return bisect_right(starts, offset)


def iter_statement_spans(text: str) -> Iterator[Tuple[int, int]]:
    """Yield ``(start, end)`` offsets of statements split on top-level ``;``."""
    n = len(text)
    i = 0
    start = 0
    while i < n:
        ch = text[i]
        if ch == "-" and text.startswith("--", i):
            i = _skip_line_comment(text, i)
        elif ch == "/" and text.startswith("/*", i):
            i = _skip_block_comment(text, i)
        elif ch == "'":
            i = _skip_single_quoted(text, i, escape_backslash=_is_escape_string_start(text, i))
        elif ch == '"':
            i = _skip_double_quoted(text, i)
        elif ch == "$":
            i, _ = _skip_dollar_quoted(text, i)
        elif ch == ";":
            yield start, i
            i += 1
            start = i
        else:
            i += 1
    if text[start:].strip():
        yield start, n


def _blank_comments(text: str) -> str:
    """Blank out comments while preserving length, so offsets stay usable.

    Comment characters become spaces and newlines are kept, which lets the
    caller map any index in the result straight back onto the source text.
    """
    out = list(text)
    n = len(text)
    i = 0

    def blank(from_i: int, to_i: int) -> None:
        for k in range(from_i, to_i):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        ch = text[i]
        if ch == "-" and text.startswith("--", i):
            end = _skip_line_comment(text, i)
            blank(i, end)
            i = end
        elif ch == "/" and text.startswith("/*", i):
            end = _skip_block_comment(text, i)
            blank(i, end)
            i = end
        elif ch == "'":
            i = _skip_single_quoted(text, i, escape_backslash=_is_escape_string_start(text, i))
        elif ch == '"':
            i = _skip_double_quoted(text, i)
        elif ch == "$":
            i, _ = _skip_dollar_quoted(text, i)
        else:
            i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Control-flow stripping
# ---------------------------------------------------------------------------

_LABEL_RE = re.compile(r"<<\s*\w+\s*>>", re.IGNORECASE)
_SIMPLE_PREFIXES = (
    re.compile(r"BEGIN\b", re.IGNORECASE),
    re.compile(r"DECLARE\b", re.IGNORECASE),
    re.compile(r"ELSE\b(?!\s*IF\b)", re.IGNORECASE),
    re.compile(r"LOOP\b", re.IGNORECASE),
    re.compile(r"EXCEPTION\b", re.IGNORECASE),
    re.compile(r"END\s+(?:IF|LOOP|CASE)\b", re.IGNORECASE),
    re.compile(r"END\b", re.IGNORECASE),
)
_IF_RE = re.compile(r"(?:ELSIF|ELSEIF|IF)\b.*?\bTHEN\b", re.IGNORECASE | re.DOTALL)
_WHEN_RE = re.compile(r"WHEN\b.*?\bTHEN\b", re.IGNORECASE | re.DOTALL)
_CASE_RE = re.compile(r"CASE\b(?!\s*WHEN\b.*\bEND\b)", re.IGNORECASE)
_WHILE_RE = re.compile(r"WHILE\b.*?\bLOOP\b", re.IGNORECASE | re.DOTALL)
_FOR_RE = re.compile(
    r"(?:FOR|FOREACH)\s+.*?\s+IN\s+(?P<query>.*?)\s+LOOP\b",
    re.IGNORECASE | re.DOTALL,
)
#: Statements that never carry lineage and can be dropped without loss.
_NOISE_RE = re.compile(
    r"^(?:RAISE|RETURN\s*$|RETURN\s+(?!QUERY\b)|NULL\s*$|COMMIT|ROLLBACK|EXIT|CONTINUE|"
    r"GET\s+DIAGNOSTICS|SET\s+|RESET\s+|ANALYZE\b|VACUUM\b|GRANT\b|REVOKE\b|"
    r"DROP\b|TRUNCATE\b|ALTER\b|CREATE\s+INDEX\b|COMMENT\s+ON\b)",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(r"^\S+\s*\.\.\s*\S+$", re.IGNORECASE)


def _strip_control_prefix(sql: str) -> Tuple[int, List[Tuple[str, int]], List[str]]:
    """Peel control-flow headers off the front of a fragment.

    Returns ``(offset, nested, context)`` where ``offset`` is the index in
    ``sql`` at which the real statement begins, ``nested`` holds ``(sql,
    offset)`` pairs for SQL embedded in the headers themselves (the source
    query of a ``FOR ... IN <query> LOOP``) and ``context`` records the
    keywords that were removed. Offsets are preserved so callers can map the
    statement back onto the original source lines.
    """
    nested: List[Tuple[str, int]] = []
    context: List[str] = []
    pos = 0
    n = len(sql)
    changed = True

    while changed and pos < n:
        changed = False
        while pos < n and sql[pos].isspace():
            pos += 1

        label = _LABEL_RE.match(sql, pos)
        if label:
            pos = label.end()
            changed = True
            continue

        for_match = _FOR_RE.match(sql, pos)
        if for_match:
            query = (for_match.group("query") or "").strip()
            if query and not _RANGE_RE.match(query):
                nested.append((query, for_match.start("query")))
            context.append("FOR")
            pos = for_match.end()
            changed = True
            continue

        for pattern, label_name in (
            (_WHILE_RE, "WHILE"),
            (_IF_RE, "IF"),
            (_WHEN_RE, "WHEN"),
            (_CASE_RE, "CASE"),
        ):
            match = pattern.match(sql, pos)
            if match:
                context.append(label_name)
                pos = match.end()
                changed = True
                break
        if changed:
            continue

        for pattern in _SIMPLE_PREFIXES:
            match = pattern.match(sql, pos)
            if match:
                context.append(match.group(0).upper())
                pos = match.end()
                changed = True
                break

    while pos < n and sql[pos].isspace():
        pos += 1
    return pos, nested, context


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

_ASSIGN_RE = re.compile(
    r"^(?P<var>(?:\"[^\"]+\"|[A-Za-z_]\w*)(?:\.(?:\"[^\"]+\"|[A-Za-z_]\w*))*)\s*(?::=|(?<![<>!:])=(?!=))",
)
_RETURN_QUERY_RE = re.compile(r"^RETURN\s+QUERY\s+", re.IGNORECASE)
_CREATE_TABLE_RE = re.compile(
    r"^CREATE\s+(?:(?P<temp>GLOBAL\s+TEMP(?:ORARY)?|LOCAL\s+TEMP(?:ORARY)?|TEMP(?:ORARY)?)\s+)?"
    r"(?:UNLOGGED\s+)?TABLE\b",
    re.IGNORECASE,
)
_INTO_RE = re.compile(
    r"\bINTO\s+(?:STRICT\s+)?(?P<var>(?:\"[^\"]+\"|[A-Za-z_]\w*)(?:\s*,\s*(?:\"[^\"]+\"|[A-Za-z_]\w*))*)",
    re.IGNORECASE,
)
_USING_TAIL_RE = re.compile(r"\s+USING\s+.*$", re.IGNORECASE | re.DOTALL)

_KIND_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("insert", re.compile(r"^INSERT\s+INTO\b", re.IGNORECASE)),
    ("update", re.compile(r"^UPDATE\b", re.IGNORECASE)),
    ("delete", re.compile(r"^DELETE\s+FROM\b", re.IGNORECASE)),
    ("execute", re.compile(r"^EXECUTE\b", re.IGNORECASE)),
    ("perform", re.compile(r"^PERFORM\b", re.IGNORECASE)),
    ("call", re.compile(r"^CALL\b", re.IGNORECASE)),
    ("select", re.compile(r"^(?:SELECT|WITH|TABLE|VALUES)\b", re.IGNORECASE)),
    ("insert", re.compile(r"^MERGE\b", re.IGNORECASE)),
)

#: A DECLARE-section entry such as ``v_cnt integer := 0`` or ``rec RECORD``.
_DECLARE_RE = re.compile(
    r"^(?P<var>[A-Za-z_]\w*)\s+(?:CONSTANT\s+)?"
    r"(?P<type>[A-Za-z_][\w.]*(?:\s+VARYING)?(?:\s*\([^)]*\))?(?:%TYPE|%ROWTYPE)?(?:\s*\[\s*\])*)"
    r"(?:\s+NOT\s+NULL)?"
    r"(?:\s*(?::=|\bDEFAULT\b)\s*(?P<init>.+))?$",
    re.IGNORECASE | re.DOTALL,
)
_STRING_LITERAL_RE = re.compile(r"^'((?:[^']|'')*)'$")
_DOLLAR_LITERAL_RE = re.compile(r"^\$([A-Za-z_]\w*)?\$(.*)\$\1\$$", re.DOTALL)


def _analyse_execute(payload: str) -> Tuple[str, bool, str]:
    """Resolve the SQL carried by an ``EXECUTE`` statement.

    Returns ``(sql, is_dynamic, reason)``. A statically known string literal is
    unwrapped so it can be parsed; anything assembled at runtime is flagged.
    """
    payload = _USING_TAIL_RE.sub("", payload).strip()

    literal = _STRING_LITERAL_RE.match(payload)
    if literal:
        return literal.group(1).replace("''", "'"), False, ""

    dollar = _DOLLAR_LITERAL_RE.match(payload)
    if dollar:
        return dollar.group(2), False, ""

    lowered = payload.lower()
    if lowered.startswith("format("):
        reason = "EXECUTE format(...) is assembled at runtime"
    elif "||" in payload:
        reason = "EXECUTE builds SQL by string concatenation"
    elif _STRING_LITERAL_RE.match(payload.split("||")[0].strip()):
        reason = "EXECUTE builds SQL by string concatenation"
    else:
        reason = "EXECUTE runs SQL held in a variable"
    return payload, True, reason


def _classify(sql: str, *, in_declare: bool = False) -> Tuple[str, str, bool, str, Optional[str]]:
    """Return ``(kind, sql, is_dynamic, dynamic_reason, into_variable)``."""
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        return "empty", "", False, "", None

    return_query = _RETURN_QUERY_RE.match(sql)
    if return_query:
        sql = sql[return_query.end() :].strip()
        if not sql:
            return "empty", "", False, "", None

    if _NOISE_RE.match(sql):
        return "noise", sql, False, "", None

    assign = _ASSIGN_RE.match(sql)
    if assign and not _CREATE_TABLE_RE.match(sql):
        expression = sql[assign.end() :].strip()
        return "assign", expression, False, "", assign.group("var")

    create = _CREATE_TABLE_RE.match(sql)
    if create:
        return ("create_temp" if create.group("temp") else "create_table"), sql, False, "", None

    for kind, pattern in _KIND_PATTERNS:
        match = pattern.match(sql)
        if not match:
            continue
        if kind == "execute":
            payload = sql[match.end() :].strip()
            resolved, dynamic, reason = _analyse_execute(payload)
            return "execute", resolved, dynamic, reason, None
        into = None
        if kind in {"select", "insert"}:
            into_match = _INTO_RE.search(sql)
            if into_match and kind == "select":
                into = into_match.group("var")
        return kind, sql, False, "", into

    if in_declare:
        declaration = _DECLARE_RE.match(sql)
        if declaration:
            return "declare", (declaration.group("init") or "").strip(), False, "", declaration.group("var")

    return "unknown", sql, False, "", None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_function_body(body: str) -> List[PlpgsqlStmt]:
    """Split a PL/pgSQL function body into the SQL statements it executes.

    Control-flow scaffolding is removed and every branch is emitted, so the
    result is a conservative superset of what a single invocation would run.
    Statements that carry no lineage (``RAISE``, ``RETURN``, ``COMMIT``, …) are
    dropped; anything unrecognised is kept with ``kind == "unknown"`` so that
    nothing disappears silently.
    """
    if not body or not body.strip():
        return []

    starts = _line_starts(body)
    blanked = _blank_comments(body)
    statements: List[PlpgsqlStmt] = []

    state = {"in_declare": False}

    def emit(raw_sql: str, abs_offset: int, span_end: int, context: List[str]) -> None:
        kind, sql, dynamic, reason, into = _classify(raw_sql, in_declare=state["in_declare"])
        if kind in {"empty", "noise"}:
            return
        statements.append(
            PlpgsqlStmt(
                kind=kind,
                sql=sql,
                line_start=_line_of(abs_offset, starts),
                line_end=_line_of(max(span_end - 1, abs_offset), starts),
                is_dynamic=dynamic,
                dynamic_reason=reason,
                into=into,
                context=list(context),
            )
        )

    for span_start, span_end in iter_statement_spans(body):
        fragment = blanked[span_start:span_end]
        if not fragment.strip():
            continue

        last = span_start + len(fragment.rstrip())
        offset, nested, context = _strip_control_prefix(fragment)

        # The DECLARE section runs until the matching BEGIN; knowing which one
        # we are in is what tells `v_cnt integer := 0` apart from a statement.
        for keyword in context:
            if keyword == "DECLARE":
                state["in_declare"] = True
            elif keyword == "BEGIN":
                state["in_declare"] = False

        for nested_sql, nested_offset in nested:
            emit(nested_sql, span_start + nested_offset, last, context)

        remainder = fragment[offset:].strip()
        if remainder:
            emit(remainder, span_start + offset, last, context)

    return statements


_CREATE_FUNCTION_RE = re.compile(
    r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:FUNCTION|PROCEDURE)\s+"
    r"(?P<name>(?:\"[^\"]+\"|[A-Za-z_]\w*)(?:\s*\.\s*(?:\"[^\"]+\"|[A-Za-z_]\w*))*)",
    re.IGNORECASE,
)
_LANGUAGE_PLPGSQL_RE = re.compile(r"\bLANGUAGE\s+'?plpgsql'?\b", re.IGNORECASE)


def _iter_dollar_quoted_spans(text: str) -> Iterator[Tuple[int, int, int]]:
    """Yield ``(open_end, close_start, tag_len)`` for each dollar-quoted block."""
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch == "-" and text.startswith("--", i):
            i = _skip_line_comment(text, i)
        elif ch == "/" and text.startswith("/*", i):
            i = _skip_block_comment(text, i)
        elif ch == "'":
            i = _skip_single_quoted(text, i, escape_backslash=_is_escape_string_start(text, i))
        elif ch == '"':
            i = _skip_double_quoted(text, i)
        elif ch == "$":
            match = _DOLLAR_TAG_RE.match(text, i)
            if not match:
                i += 1
                continue
            tag = match.group(0)
            close = text.find(tag, match.end())
            if close == -1:
                return
            yield match.end(), close, len(tag)
            i = close + len(tag)
        else:
            i += 1


def is_plpgsql_function(sql: str) -> bool:
    """True when ``sql`` declares at least one ``LANGUAGE plpgsql`` routine."""
    if not sql:
        return False
    stripped = _blank_comments(sql)
    return bool(_CREATE_FUNCTION_RE.search(stripped) and _LANGUAGE_PLPGSQL_RE.search(stripped))


def find_function_defs(sql: str) -> List[Tuple[str, str]]:
    """Return ``(function_name, body)`` for every PL/pgSQL routine in ``sql``."""
    if not sql:
        return []

    results: List[Tuple[str, str]] = []
    for match in _CREATE_FUNCTION_RE.finditer(sql):
        name = re.sub(r"\s*\.\s*", ".", match.group("name")).replace('"', "")
        tail = sql[match.end() :]
        if not _LANGUAGE_PLPGSQL_RE.search(tail.split(";")[0] or tail):
            # `LANGUAGE plpgsql` may follow the body; fall back to a wider look.
            if not _LANGUAGE_PLPGSQL_RE.search(tail):
                continue

        body = None
        for open_end, close_start, _tag_len in _iter_dollar_quoted_spans(tail):
            body = tail[open_end:close_start]
            break
        if body is None:
            quoted = re.search(r"\bAS\s+'((?:[^']|'')*)'", tail, re.IGNORECASE | re.DOTALL)
            if quoted:
                body = quoted.group(1).replace("''", "'")
        if body is None:
            continue
        results.append((name.lower(), body))

    return results


def extract_function_def(create_function_sql: str) -> Tuple[str, str]:
    """Return ``(function_name, body)`` for the first PL/pgSQL routine found.

    Raises:
        ValueError: when no ``LANGUAGE plpgsql`` function with a readable body
            is present. Failing loudly keeps unparsed input from silently
            producing an empty lineage graph.
    """
    defs = find_function_defs(create_function_sql)
    if not defs:
        raise ValueError(
            "No PL/pgSQL function definition found: expected "
            "`CREATE [OR REPLACE] FUNCTION ... AS $$ ... $$ LANGUAGE plpgsql`"
        )
    return defs[0]
