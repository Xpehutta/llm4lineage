import unittest

from Classes.plpgsql_splitter import (
    PlpgsqlStmt,
    extract_function_def,
    find_function_defs,
    is_plpgsql_function,
    split_function_body,
)

FULL_FUNCTION = """
CREATE OR REPLACE FUNCTION analytics.refresh_sales(p_day date)
RETURNS void AS $$
DECLARE
    v_cnt integer := 0;
    v_sql text;
BEGIN
    CREATE TEMP TABLE tmp_orders AS
    SELECT o.id, o.amount FROM sales.orders o WHERE o.day = p_day;

    IF p_day IS NULL THEN
        INSERT INTO analytics.errors (msg) SELECT 'null day';
    ELSIF p_day > current_date THEN
        INSERT INTO analytics.future (id) SELECT id FROM tmp_orders;
    ELSE
        UPDATE analytics.sales SET amount = t.amount
        FROM tmp_orders t WHERE analytics.sales.id = t.id;
    END IF;

    FOR rec IN SELECT id FROM sales.customers LOOP
        DELETE FROM analytics.stale WHERE id = rec.id;
    END LOOP;

    v_sql := 'SELECT 1';
    EXECUTE format('INSERT INTO %I SELECT * FROM tmp_orders', v_target);
    EXECUTE 'SELECT count(*) FROM sales.orders';

    RAISE NOTICE 'done %', v_cnt;
    RETURN QUERY SELECT id FROM analytics.sales;
END;
$$ LANGUAGE plpgsql;
"""


def kinds(statements):
    return [stmt.kind for stmt in statements]


def by_kind(statements, kind):
    return [stmt for stmt in statements if stmt.kind == kind]


class TestExtractFunctionDef(unittest.TestCase):
    def test_extracts_name_and_body(self):
        name, body = extract_function_def(FULL_FUNCTION)
        self.assertEqual(name, "analytics.refresh_sales")
        self.assertIn("CREATE TEMP TABLE tmp_orders", body)
        self.assertNotIn("LANGUAGE plpgsql", body)

    def test_custom_dollar_tag(self):
        sql = """
        CREATE FUNCTION public.f() RETURNS void AS $body$
        BEGIN
            INSERT INTO t SELECT 1;
        END;
        $body$ LANGUAGE plpgsql;
        """
        name, body = extract_function_def(sql)
        self.assertEqual(name, "public.f")
        self.assertIn("INSERT INTO t", body)

    def test_quoted_identifier_name(self):
        sql = 'CREATE FUNCTION analytics."My Func"() RETURNS void AS $$BEGIN END;$$ LANGUAGE plpgsql;'
        name, _ = extract_function_def(sql)
        self.assertEqual(name, "analytics.my func")

    def test_raises_when_absent(self):
        with self.assertRaises(ValueError):
            extract_function_def("SELECT 1")

    def test_sql_language_function_is_not_plpgsql(self):
        sql = "CREATE FUNCTION f() RETURNS int AS $$ SELECT 1 $$ LANGUAGE sql;"
        self.assertFalse(is_plpgsql_function(sql))
        with self.assertRaises(ValueError):
            extract_function_def(sql)

    def test_is_plpgsql_function(self):
        self.assertTrue(is_plpgsql_function(FULL_FUNCTION))
        self.assertFalse(is_plpgsql_function("INSERT INTO t SELECT 1"))

    def test_find_multiple_functions(self):
        sql = FULL_FUNCTION + """
        CREATE FUNCTION public.second() RETURNS void AS $b$
        BEGIN
            INSERT INTO other SELECT 1;
        END;
        $b$ LANGUAGE plpgsql;
        """
        defs = find_function_defs(sql)
        self.assertEqual([name for name, _ in defs], ["analytics.refresh_sales", "public.second"])


class TestSplitFunctionBody(unittest.TestCase):
    def setUp(self):
        _, body = extract_function_def(FULL_FUNCTION)
        self.body = body
        self.statements = split_function_body(body)

    def test_extracts_every_branch(self):
        found = kinds(self.statements)
        for expected in ("create_temp", "insert", "update", "delete", "execute", "select"):
            self.assertIn(expected, found, f"missing {expected} in {found}")

    def test_all_if_branches_are_kept(self):
        inserts = [stmt.sql for stmt in by_kind(self.statements, "insert")]
        self.assertTrue(any("analytics.errors" in sql for sql in inserts))
        self.assertTrue(any("analytics.future" in sql for sql in inserts))
        self.assertEqual(len(by_kind(self.statements, "update")), 1)

    def test_for_loop_query_and_body_both_extracted(self):
        selects = [stmt.sql for stmt in by_kind(self.statements, "select")]
        self.assertTrue(any("sales.customers" in sql for sql in selects))
        deletes = by_kind(self.statements, "delete")
        self.assertEqual(len(deletes), 1)
        self.assertIn("analytics.stale", deletes[0].sql)

    def test_create_temp_detected(self):
        temps = by_kind(self.statements, "create_temp")
        self.assertEqual(len(temps), 1)
        self.assertIn("tmp_orders", temps[0].sql)

    def test_return_query_becomes_select(self):
        selects = [stmt.sql for stmt in by_kind(self.statements, "select")]
        self.assertTrue(any(sql.upper().startswith("SELECT ID FROM ANALYTICS.SALES") for sql in selects))

    def test_noise_statements_dropped(self):
        self.assertNotIn("noise", kinds(self.statements))
        self.assertFalse(any("RAISE" in stmt.sql.upper() for stmt in self.statements))

    def test_assignment_recorded(self):
        assigns = by_kind(self.statements, "assign")
        self.assertTrue(any(stmt.into == "v_sql" for stmt in assigns))

    def test_line_numbers_are_ordered_and_in_range(self):
        total_lines = self.body.count("\n") + 1
        previous = 0
        for stmt in self.statements:
            self.assertGreaterEqual(stmt.line_start, 1)
            self.assertLessEqual(stmt.line_end, total_lines)
            self.assertLessEqual(stmt.line_start, stmt.line_end)
            self.assertGreaterEqual(stmt.line_start, previous)
            previous = stmt.line_start

    def test_line_numbers_point_at_the_real_source(self):
        lines = self.body.splitlines()
        temp = by_kind(self.statements, "create_temp")[0]
        self.assertIn("CREATE TEMP TABLE", lines[temp.line_start - 1])


class TestDynamicExecute(unittest.TestCase):
    def test_static_execute_is_unwrapped(self):
        stmts = split_function_body("BEGIN EXECUTE 'SELECT * FROM t'; END;")
        execs = by_kind(stmts, "execute")
        self.assertEqual(len(execs), 1)
        self.assertFalse(execs[0].is_dynamic)
        self.assertEqual(execs[0].sql, "SELECT * FROM t")
        self.assertEqual(execs[0].dynamic_reason, "")

    def test_static_execute_unescapes_doubled_quotes(self):
        stmts = split_function_body("BEGIN EXECUTE 'SELECT ''a'' FROM t'; END;")
        self.assertEqual(by_kind(stmts, "execute")[0].sql, "SELECT 'a' FROM t")

    def test_format_execute_is_dynamic(self):
        stmts = split_function_body("BEGIN EXECUTE format('SELECT * FROM %I', v); END;")
        stmt = by_kind(stmts, "execute")[0]
        self.assertTrue(stmt.is_dynamic)
        self.assertIn("format", stmt.dynamic_reason)

    def test_variable_execute_is_dynamic(self):
        stmts = split_function_body("BEGIN EXECUTE v_sql; END;")
        stmt = by_kind(stmts, "execute")[0]
        self.assertTrue(stmt.is_dynamic)
        self.assertTrue(stmt.dynamic_reason)

    def test_concatenated_execute_is_dynamic(self):
        stmts = split_function_body("BEGIN EXECUTE 'SELECT * FROM ' || v_tbl; END;")
        stmt = by_kind(stmts, "execute")[0]
        self.assertTrue(stmt.is_dynamic)
        self.assertIn("concatenation", stmt.dynamic_reason)

    def test_using_clause_is_stripped(self):
        stmts = split_function_body("BEGIN EXECUTE 'SELECT * FROM t WHERE id = $1' USING v_id; END;")
        stmt = by_kind(stmts, "execute")[0]
        self.assertFalse(stmt.is_dynamic)
        self.assertEqual(stmt.sql, "SELECT * FROM t WHERE id = $1")


class TestLexer(unittest.TestCase):
    def test_semicolon_inside_string_does_not_split(self):
        stmts = split_function_body("BEGIN INSERT INTO t (c) VALUES ('a;b'); END;")
        inserts = by_kind(stmts, "insert")
        self.assertEqual(len(inserts), 1)
        self.assertIn("a;b", inserts[0].sql)

    def test_semicolon_inside_dollar_quote_does_not_split(self):
        stmts = split_function_body("BEGIN PERFORM $tag$ a; b $tag$; END;")
        self.assertEqual(len(by_kind(stmts, "perform")), 1)

    def test_semicolon_inside_comment_does_not_split(self):
        body = """
        BEGIN
            -- a comment with ; inside
            INSERT INTO t SELECT 1;
        END;
        """
        self.assertEqual(len(by_kind(split_function_body(body), "insert")), 1)

    def test_nested_block_comment(self):
        body = "BEGIN /* outer /* inner ; */ still comment ; */ INSERT INTO t SELECT 1; END;"
        self.assertEqual(len(by_kind(split_function_body(body), "insert")), 1)

    def test_doubled_quote_inside_literal(self):
        stmts = split_function_body("BEGIN INSERT INTO t (c) VALUES ('it''s; fine'); END;")
        self.assertEqual(len(by_kind(stmts, "insert")), 1)

    def test_escape_string_backslash_quote(self):
        stmts = split_function_body(r"BEGIN INSERT INTO t (c) VALUES (E'a\';b'); END;")
        self.assertEqual(len(by_kind(stmts, "insert")), 1)

    def test_quoted_identifier_with_semicolon(self):
        stmts = split_function_body('BEGIN INSERT INTO "weird;name" SELECT 1; END;')
        self.assertEqual(len(by_kind(stmts, "insert")), 1)

    def test_positional_parameter_is_not_dollar_quote(self):
        body = "BEGIN UPDATE t SET a = $1 WHERE b = $2; INSERT INTO u SELECT 1; END;"
        stmts = split_function_body(body)
        self.assertEqual(len(by_kind(stmts, "update")), 1)
        self.assertEqual(len(by_kind(stmts, "insert")), 1)

    def test_empty_body(self):
        self.assertEqual(split_function_body(""), [])
        self.assertEqual(split_function_body("   \n  "), [])

    def test_unknown_statement_is_preserved(self):
        stmts = split_function_body("BEGIN FOOBAR baz; END;")
        unknown = by_kind(stmts, "unknown")
        self.assertEqual(len(unknown), 1)
        self.assertEqual(unknown[0].sql, "FOOBAR baz")


class TestStatementFlags(unittest.TestCase):
    def test_is_lineage_bearing(self):
        self.assertTrue(PlpgsqlStmt("insert", "INSERT INTO t SELECT 1", 1, 1).is_lineage_bearing)
        self.assertFalse(PlpgsqlStmt("unknown", "FOOBAR", 1, 1).is_lineage_bearing)

    def test_select_into_variable_recorded(self):
        stmts = split_function_body("BEGIN SELECT count(*) INTO v_cnt FROM t; END;")
        stmt = by_kind(stmts, "select")[0]
        self.assertEqual(stmt.into, "v_cnt")

    def test_context_records_enclosing_control_flow(self):
        stmts = split_function_body("BEGIN IF x THEN INSERT INTO t SELECT 1; END IF; END;")
        insert = by_kind(stmts, "insert")[0]
        self.assertIn("IF", insert.context)


if __name__ == "__main__":
    unittest.main()
