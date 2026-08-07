import unittest
from pathlib import Path

from Classes.plpgsql_lineage import (
    DYNAMIC_CONFIDENCE,
    DYNAMIC_PROVENANCE,
    PlpgsqlLineageExtractor,
    contains_plpgsql_function,
    extract_plpgsql_lineage,
)
from Classes.sql2graph_classes import SQL2GraphBuilder

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "plpgsql_functions.sql"


def load_function(name: str) -> str:
    """Return the fixture text starting at ``name`` so it is treated as primary."""
    text = FIXTURE.read_text(encoding="utf-8")
    marker = f"CREATE OR REPLACE FUNCTION {name}"
    index = text.index(marker)
    return text[index:]


def derived_edges(result):
    return [
        (link["source"], link["target"], link)
        for link in result["graph"]["links"]
        if link.get("edge_type") == "DERIVED_FROM"
    ]


def edge_between(result, source, target):
    for src, tgt, link in derived_edges(result):
        if src == source and tgt == target:
            return link
    return None


class TestGraphContract(unittest.TestCase):
    """Every fixture must produce a graph the shared validator accepts."""

    def test_all_fixtures_validate(self):
        for name in (
            "analytics.build_daily_summary",
            "analytics.route_customer",
            "staging.load_partition",
        ):
            with self.subTest(function=name):
                result = extract_plpgsql_lineage(load_function(name))
                self.assertEqual(result["pipeline_stage"], "plpgsql")
                self.assertEqual(result["metadata"]["pipeline_stage"], "plpgsql")
                self.assertTrue(result["metadata"]["is_dag"])
                self.assertEqual(result["warnings"], [])

    def test_node_and_edge_types_are_known(self):
        result = extract_plpgsql_lineage(load_function("analytics.build_daily_summary"))
        for node in result["graph"]["nodes"]:
            node_type = node.get("node_type")
            if node_type:
                self.assertIn(node_type, SQL2GraphBuilder.ALL_NODE_TYPES)
        for link in result["graph"]["links"]:
            edge_type = link.get("edge_type")
            if edge_type:
                self.assertIn(edge_type, SQL2GraphBuilder.ALL_EDGE_TYPES)

    def test_every_edge_carries_confidence_and_provenance(self):
        result = extract_plpgsql_lineage(load_function("staging.load_partition"))
        self.assertTrue(result["graph"]["links"])
        for link in result["graph"]["links"]:
            self.assertIn("confidence", link)
            self.assertIn("provenance", link)


class TestTempTableChaining(unittest.TestCase):
    def setUp(self):
        self.result = extract_plpgsql_lineage(load_function("analytics.build_daily_summary"))

    def test_temp_table_recorded(self):
        self.assertIn("tmp_daily_orders", self.result["temp_tables"])

    def test_lineage_chains_through_temp_table(self):
        self.assertIsNotNone(
            edge_between(self.result, "sales.orders.amount", "tmp_daily_orders.amount"),
            "source table should feed the temp table",
        )
        self.assertIsNotNone(
            edge_between(self.result, "tmp_daily_orders.amount", "analytics.daily_summary.total_amount"),
            "temp table should feed the final target",
        )

    def test_temp_columns_marked(self):
        temp_nodes = [
            node
            for node in self.result["graph"]["nodes"]
            if str(node["id"]).startswith("tmp_daily_orders.")
        ]
        self.assertTrue(temp_nodes)
        self.assertTrue(all(node.get("is_temp") for node in temp_nodes))

    def test_variable_recorded(self):
        self.assertIn("v_rows", self.result["variables"])

    def test_select_into_does_not_create_a_table(self):
        targets = {stmt["target"] for stmt in self.result["statements"]}
        self.assertIn("var.v_rows", targets)
        self.assertNotIn("v_rows", self.result["temp_tables"])


class TestBranchMerging(unittest.TestCase):
    def setUp(self):
        self.result = extract_plpgsql_lineage(load_function("analytics.route_customer"))

    def test_all_branches_present(self):
        self.assertIsNotNone(edge_between(self.result, "crm.customers.id", "analytics.new_customers.id"))
        self.assertIsNotNone(
            edge_between(self.result, "crm.customers.id", "analytics.churned_customers.id")
        )
        self.assertIsNotNone(
            edge_between(self.result, "crm.customers.status", "analytics.customer_state.status"),
            "the ELSE branch UPDATE must be resolved at column level",
        )

    def test_conditional_edges_are_flagged(self):
        link = edge_between(self.result, "crm.customers.id", "analytics.churned_customers.id")
        self.assertTrue(link.get("conditional"))
        self.assertIn("IF", link.get("control_flow", ""))

    def test_loop_query_captured(self):
        sources = {src for src, _, _ in derived_edges(self.result)}
        self.assertTrue(any(src.startswith("crm.blacklist.") for src in sources))

    def test_update_is_column_resolved(self):
        update_statements = [s for s in self.result["statements"] if s["kind"] == "update"]
        self.assertEqual(len(update_statements), 1)
        self.assertTrue(update_statements[0]["resolved"])
        self.assertEqual(update_statements[0]["target"], "analytics.customer_state")


class TestDynamicSql(unittest.TestCase):
    def setUp(self):
        self.result = extract_plpgsql_lineage(load_function("staging.load_partition"))

    def test_dynamic_statements_reported_as_unresolved(self):
        reasons = [item["reason"] for item in self.result["unresolved"]]
        self.assertEqual(reasons.count("dynamic_execute"), 2)

    def test_unresolved_items_carry_the_offending_fragment(self):
        for item in self.result["unresolved"]:
            self.assertTrue(item["sql_fragment"])
            self.assertTrue(item["detail"])
            self.assertGreaterEqual(item["line_start"], 1)

    def test_recovered_dynamic_edges_are_low_confidence(self):
        dynamic_links = [
            link
            for link in self.result["graph"]["links"]
            if link.get("provenance") == DYNAMIC_PROVENANCE
        ]
        self.assertTrue(dynamic_links)
        for link in dynamic_links:
            self.assertEqual(link["confidence"], DYNAMIC_CONFIDENCE)
            self.assertEqual(link["transform_type"], "dynamic")
            self.assertTrue(link["sql_fragment"])

    def test_static_source_recovered_from_format_string(self):
        sources = {src for src, _, _ in derived_edges(self.result)}
        self.assertTrue(any(src.startswith("staging.raw_events.") for src in sources))

    def test_runtime_target_is_not_invented(self):
        targets = [entry["target"] for entry in self.result["table_lineage"]]
        self.assertNotIn("dynamic_placeholder", targets)
        self.assertIn("", targets)

    def test_static_execute_is_fully_resolved(self):
        table_targets = {entry["target"] for entry in self.result["table_lineage"]}
        self.assertIn("staging.audit", table_targets)


class TestRecursionGuard(unittest.TestCase):
    SELF_RECURSIVE = """
    CREATE FUNCTION public.walk(p_id int) RETURNS void AS $$
    BEGIN
        INSERT INTO public.visited (id) SELECT id FROM public.nodes WHERE id = p_id;
        PERFORM public.walk(p_id + 1);
    END;
    $$ LANGUAGE plpgsql;
    """

    MUTUAL_RECURSION = """
    CREATE FUNCTION public.ping() RETURNS void AS $$
    BEGIN
        INSERT INTO public.a (id) SELECT id FROM public.src;
        PERFORM public.pong();
    END;
    $$ LANGUAGE plpgsql;

    CREATE FUNCTION public.pong() RETURNS void AS $$
    BEGIN
        INSERT INTO public.b (id) SELECT id FROM public.a;
        PERFORM public.ping();
    END;
    $$ LANGUAGE plpgsql;
    """

    def test_self_recursion_terminates(self):
        result = extract_plpgsql_lineage(self.SELF_RECURSIVE)
        self.assertIn("recursive_call", [item["reason"] for item in result["unresolved"]])
        self.assertIsNotNone(edge_between(result, "public.nodes.id", "public.visited.id"))

    def test_mutual_recursion_terminates(self):
        result = extract_plpgsql_lineage(self.MUTUAL_RECURSION)
        self.assertIn("recursive_call", [item["reason"] for item in result["unresolved"]])
        self.assertIsNotNone(edge_between(result, "public.src.id", "public.a.id"))
        self.assertIsNotNone(
            edge_between(result, "public.a.id", "public.b.id"),
            "the callee's lineage should be inlined into the caller's graph",
        )

    def test_depth_limit_is_respected(self):
        extractor = PlpgsqlLineageExtractor(max_depth=0)
        result = extractor.extract(self.MUTUAL_RECURSION)
        reasons = {item["reason"] for item in result["unresolved"]}
        self.assertIn("max_depth_exceeded", reasons)


class TestRouting(unittest.TestCase):
    def test_contains_plpgsql_function(self):
        self.assertTrue(contains_plpgsql_function(load_function("staging.load_partition")))
        self.assertFalse(contains_plpgsql_function("INSERT INTO t SELECT 1"))

    def test_plain_sql_reports_error_instead_of_empty_graph(self):
        result = extract_plpgsql_lineage("SELECT 1")
        self.assertIn("error", result)
        self.assertEqual(result["graph"]["nodes"], [])

    def test_unknown_statement_is_reported(self):
        sql = """
        CREATE FUNCTION public.odd() RETURNS void AS $$
        BEGIN
            FLUMMOX the_widget;
        END;
        $$ LANGUAGE plpgsql;
        """
        result = extract_plpgsql_lineage(sql)
        self.assertIn("unsupported_statement", [item["reason"] for item in result["unresolved"]])


if __name__ == "__main__":
    unittest.main()
