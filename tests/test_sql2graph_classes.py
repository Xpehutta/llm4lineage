import json
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx

from Classes.sql2graph_classes import (
    SQL2GraphBuilder,
    SQL2GraphLLMExtractor,
    SQL2GraphParser,
    SQL2GraphValidator,
    SQL2GraphVisualizer,
)


class _DummyChatAdapter:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = 0

    def invoke_messages(self, _messages):
        payload = self.payloads[self.calls]
        self.calls += 1
        return payload


class _DummyLegacyAdapter:
    """Simulates older adapter that only has invoke()."""

    class _Resp:
        def __init__(self, content: str):
            self.content = content

    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = 0

    def invoke(self, _payload):
        payload = self.payloads[self.calls]
        self.calls += 1
        return self._Resp(payload)


class _DummyExtractor:
    def __init__(self, payload, *, enable_refinement: bool = True):
        self.payload = payload
        self.enable_refinement = enable_refinement

    def extract(self, **_kwargs):
        return self.payload

    def verify(self, **_kwargs):
        return self.payload

    def enhance(self, **_kwargs):
        return self.payload

    def verify_and_enhance(self, **_kwargs):
        return self.payload


SAMPLE_EXTRACTION = {
    "ctes": [
        {
            "alias": "r",
            "output_columns": [
                {
                    "alias": "customer_id",
                    "expression": "customer_id",
                    "dependencies": [{"table_alias": "orders", "column": "customer_id"}],
                    "aggregate": False,
                    "window_function": False,
                },
                {
                    "alias": "total",
                    "expression": "SUM(amount)",
                    "dependencies": [
                        {"table_alias": "orders", "column": "amount"},
                        {"table_alias": "orders", "column": "customer_id"},
                    ],
                    "aggregate": True,
                    "window_function": False,
                },
            ],
            "filters": [
                {
                    "clause": "WHERE",
                    "condition": "order_date > '2025-01-01'",
                    "columns_used": [{"table_alias": "orders", "column": "order_date"}],
                }
            ],
            "joins": [],
            "group_by_columns": [{"table_alias": "orders", "column": "customer_id"}],
            "ctes": [],
        }
    ],
    "output_columns": [
        {
            "alias": "name",
            "expression": "c.name",
            "dependencies": [{"table_alias": "c", "column": "name"}],
            "aggregate": False,
            "window_function": False,
        },
        {
            "alias": "total",
            "expression": "r.total",
            "dependencies": [{"table_alias": "r", "column": "total"}],
            "aggregate": False,
            "window_function": False,
        },
    ],
    "filters": [
        {
            "clause": "WHERE",
            "condition": "c.active = true",
            "columns_used": [{"table_alias": "c", "column": "active"}],
        }
    ],
    "joins": [
        {
            "type": "INNER",
            "left_alias": "c",
            "right_alias": "r",
            "condition": "c.id = r.customer_id",
            "join_columns": [
                {"table_alias": "c", "column": "id"},
                {"table_alias": "r", "column": "customer_id"},
            ],
        }
    ],
    "group_by_columns": [],
}


class TestSQL2GraphParser(unittest.TestCase):
    def test_parser_fallback_or_structure(self):
        parser = SQL2GraphParser()
        sql = "SELECT c.name FROM customers c WHERE c.active = true"
        simplified = parser.simplify(sql)

        if parser.sqlglot_available:
            self.assertTrue(simplified.get("parser_used"))
            self.assertIn("select", simplified)
            self.assertIn("where", simplified)
        else:
            self.assertFalse(simplified.get("parser_used"))
            self.assertEqual(simplified.get("raw_sql"), sql)

    def test_simplify_returns_fallback_on_parse_error(self):
        parser = SQL2GraphParser()
        if not parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        simplified = parser.simplify("SELECT FROM WHERE")
        self.assertFalse(simplified.get("parser_used"))
        self.assertEqual(simplified.get("raw_sql"), "SELECT FROM WHERE")
        self.assertIn("parse_error", simplified)

    def test_parser_extracts_deterministic_filters_and_joins(self):
        parser = SQL2GraphParser()
        if not parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        sql = (
            "SELECT c.name FROM customers c "
            "JOIN orders o ON c.id = o.customer_id "
            "WHERE c.active = true"
        )
        simplified = parser.simplify(sql)
        self.assertTrue(simplified["deterministic_filters"])
        self.assertEqual(simplified["deterministic_filters"][0]["clause"], "WHERE")
        self.assertIn("c.active", simplified["deterministic_filters"][0]["condition"])
        self.assertTrue(simplified["deterministic_joins"])
        self.assertEqual(len(simplified["deterministic_joins"][0]["join_columns"]), 2)

    def test_parser_detects_insert_statement_context(self):
        parser = SQL2GraphParser()
        if not parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        sql = """
        INSERT INTO analytics.sales_summary (category, total)
        SELECT p.category, SUM(s.amount)
        FROM products.raw_data p
        JOIN sales.transactions s ON p.product_id = s.product_id
        GROUP BY p.category
        """
        simplified = parser.simplify(sql)
        self.assertEqual(simplified["statement_type"], "insert")
        self.assertEqual(simplified["target_table"], "analytics.sales_summary")
        self.assertTrue(simplified["parser_used"])

    def test_parser_extracts_subgraph_blocks_for_cte_join_and_union(self):
        parser = SQL2GraphParser()
        if not parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        sql = """
        WITH cte_a AS (
            SELECT a.id, a.name FROM table_a a
        )
        SELECT c.id FROM cte_a c
        JOIN table_b b ON c.id = b.id
        UNION
        SELECT x.id FROM table_x x
        """
        simplified = parser.simplify(sql)
        block_types = [block["type"] for block in simplified.get("subgraph_blocks", [])]
        self.assertIn("cte", block_types)
        self.assertIn("subjoin", block_types)
        self.assertIn("union_block", block_types)


class TestSQL2GraphBuilderAndValidator(unittest.TestCase):
    def test_build_graph_is_directed_acyclic(self):
        builder = SQL2GraphBuilder()
        graph = builder.build(SAMPLE_EXTRACTION)
        builder.ensure_acyclic()
        self.assertTrue(nx.is_directed_acyclic_graph(graph))

    def test_build_graph_contains_expected_nodes_and_edges(self):
        builder = SQL2GraphBuilder()
        graph = builder.build(SAMPLE_EXTRACTION)

        self.assertIn("output.name", graph.nodes)
        self.assertIn("c.name", graph.nodes)
        self.assertIn("r.total", graph.nodes)

        derived_edges = [
            (u, v, d.get("edge_type")) for u, v, d in graph.edges(data=True) if d.get("edge_type") == "DERIVED_FROM"
        ]
        self.assertIn(("c.name", "output.name", "DERIVED_FROM"), derived_edges)
        self.assertIn(("r.total", "output.total", "DERIVED_FROM"), derived_edges)

        filter_edges = [d.get("edge_type") for _, _, d in graph.edges(data=True)]
        self.assertIn("FILTERED_BY", filter_edges)
        self.assertIn("USES_COLUMN", filter_edges)
        self.assertIn("JOINS_ON", filter_edges)

        node_link = builder.to_node_link()
        self.assertIn("links", node_link)
        self.assertGreater(len(node_link["links"]), 0)

    def test_validator_reports_unknown_columns_with_schema(self):
        builder = SQL2GraphBuilder()
        graph = builder.build(SAMPLE_EXTRACTION)

        schema = {
            "tables": [
                {"name": "customers", "alias": "c", "columns": [{"name": "id"}, {"name": "name"}, {"name": "active"}]},
                {"name": "recent_orders", "alias": "r", "columns": [{"name": "customer_id"}]},
                {"name": "orders", "alias": "orders", "columns": [{"name": "customer_id"}, {"name": "amount"}]},
            ]
        }
        warnings = SQL2GraphValidator.validate_graph(graph, schema=schema)
        self.assertTrue(any("Unknown column reference: r.total" in warning for warning in warnings))

    def test_validator_accepts_valid_extraction(self):
        ok, message = SQL2GraphValidator.validate_extraction(SAMPLE_EXTRACTION)
        self.assertTrue(ok)
        self.assertEqual(message, "valid")

    def test_visualizer_loads_both_links_and_edges_shapes(self):
        builder = SQL2GraphBuilder()
        _ = builder.build(SAMPLE_EXTRACTION)
        node_link = builder.to_node_link()

        graph_from_links = SQL2GraphVisualizer.graph_from_node_link(node_link)
        self.assertGreater(graph_from_links.number_of_edges(), 0)

        as_edges = dict(node_link)
        as_edges["edges"] = as_edges.pop("links")
        graph_from_edges = SQL2GraphVisualizer.graph_from_node_link(as_edges)
        self.assertGreater(graph_from_edges.number_of_edges(), 0)

    def test_interactive_html_contains_vis_graph_payload(self):
        builder = SQL2GraphBuilder()
        _ = builder.build(SAMPLE_EXTRACTION)
        node_link = builder.to_node_link()
        html_doc = SQL2GraphVisualizer.to_interactive_html(node_link, title="Test graph")

        self.assertIn("vis-network", html_doc)
        self.assertIn("nodeDetails", html_doc)
        self.assertIn("edgeDetails", html_doc)
        self.assertIn("output.total", html_doc)
        self.assertIn("Search nodes", html_doc)

    def test_interactive_html_rejects_empty_graph(self):
        with self.assertRaises(ValueError):
            SQL2GraphVisualizer.to_interactive_html({"nodes": [], "links": []})

    def test_build_plotly_figure(self):
        try:
            import plotly.graph_objects as go
        except ImportError:
            self.skipTest("plotly not installed")
        builder = SQL2GraphBuilder()
        _ = builder.build(SAMPLE_EXTRACTION)
        graph = SQL2GraphVisualizer.graph_from_node_link(builder.to_node_link())
        fig, node_ids = SQL2GraphVisualizer._build_plotly_figure(graph, "Test graph")
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(node_ids), 0)
        self.assertGreater(len(fig.data), 1)

    def test_collect_lineage_nodes_traverses_derived_from_chain(self):
        graph = nx.MultiDiGraph()
        graph.add_node("source.a", node_type="source_column")
        graph.add_node("mid.b", node_type="source_column")
        graph.add_node("output.x", node_type="output_column")
        graph.add_edge("source.a", "mid.b", edge_type="DERIVED_FROM")
        graph.add_edge("mid.b", "output.x", edge_type="DERIVED_FROM")

        lineage = SQL2GraphVisualizer.collect_lineage_nodes(graph, "mid.b")
        self.assertEqual(lineage, {"source.a", "mid.b", "output.x"})

        isolated = SQL2GraphVisualizer.collect_lineage_nodes(graph, "output.x")
        self.assertEqual(isolated, {"source.a", "mid.b", "output.x"})

    def test_normalize_scope_payload_fills_missing_condition_and_join_columns(self):
        raw = {
            "output_columns": [
                {"alias": "x", "expression": "a.id", "dependencies": [{"table_alias": "a", "column": "id"}]}
            ],
            "filters": [{"clause": "a.flag = 1", "columns_used": [{"table_alias": "a", "column": "flag"}]}],
            "joins": [{"type": "INNER", "left_alias": "a", "right_alias": "b", "condition": "a.id = b.id"}],
            "ctes": [],
        }
        normalized = SQL2GraphLLMExtractor._normalize_scope_payload(raw)
        self.assertEqual(normalized["filters"][0]["clause"], "WHERE")
        self.assertEqual(normalized["filters"][0]["condition"], "a.flag = 1")
        self.assertEqual(len(normalized["joins"][0]["join_columns"]), 2)
        self.assertEqual(normalized["joins"][0]["join_columns"][0]["table_alias"], "a")

    def test_normalize_scope_payload_coerces_alternative_llm_shapes(self):
        """Regression: left/right_column join dicts and string group_by/dependency refs."""
        from Classes.sql2graph_classes import SQL2GraphExtraction

        raw = {
            "output_columns": [
                {
                    "alias": "meas_val",
                    "expression": "SUM(a.meas_val)",
                    "dependencies": ["a_agr_cred_coa_period.meas_val"],
                    "aggregate": True,
                }
            ],
            "filters": [
                {"clause": "WHERE", "condition": "a.actual_flg = 1", "columns_used": ["a.actual_flg"]}
            ],
            "joins": [
                {
                    "type": "INNER",
                    "left_alias": "a",
                    "right_alias": "b",
                    "condition": "a.agr_cred_id = b.agr_cred_id",
                    "join_columns": [{"left_column": "agr_cred_id", "right_column": "agr_cred_id"}],
                }
            ],
            "group_by_columns": ["a_agr_cred_coa_period.meas_cd", "meas_dt"],
            "ctes": [],
        }
        normalized = SQL2GraphLLMExtractor._normalize_scope_payload(raw)

        join_columns = normalized["joins"][0]["join_columns"]
        self.assertEqual(len(join_columns), 2)
        self.assertEqual(join_columns[0], {"table_alias": "a", "column": "agr_cred_id"})
        self.assertEqual(join_columns[1], {"table_alias": "b", "column": "agr_cred_id"})

        self.assertEqual(
            normalized["group_by_columns"],
            [
                {"table_alias": "a_agr_cred_coa_period", "column": "meas_cd"},
                {"table_alias": None, "column": "meas_dt"},
            ],
        )
        self.assertEqual(
            normalized["output_columns"][0]["dependencies"],
            [{"table_alias": "a_agr_cred_coa_period", "column": "meas_val"}],
        )
        self.assertEqual(
            normalized["filters"][0]["columns_used"],
            [{"table_alias": "a", "column": "actual_flg"}],
        )

        # The coerced payload must now pass schema validation.
        SQL2GraphExtraction.model_validate(normalized)

    def test_verify_and_enhance_calls_llm_twice_when_refinement_enabled(self):
        payload = {
            "ctes": [],
            "output_columns": [
                {
                    "alias": "x",
                    "expression": "a.id",
                    "dependencies": [{"table_alias": "a", "column": "id"}],
                    "aggregate": False,
                    "window_function": False,
                }
            ],
            "filters": [],
            "joins": [],
            "group_by_columns": [],
        }
        refined_payload = {
            "ctes": [],
            "output_columns": [
                {
                    "alias": "x",
                    "expression": "a.id + b.id",
                    "dependencies": [
                        {"table_alias": "a", "column": "id"},
                        {"table_alias": "b", "column": "id"},
                    ],
                    "aggregate": False,
                    "window_function": False,
                }
            ],
            "filters": [],
            "joins": [],
            "group_by_columns": [],
        }

        extractor = SQL2GraphLLMExtractor.__new__(SQL2GraphLLMExtractor)
        extractor.max_retries = 1
        extractor.enable_refinement = True
        extractor.verification_system_prompt = "system"
        extractor.enhancement_system_prompt = "system"
        extractor.chat_adapter = _DummyChatAdapter(
            [json.dumps(payload), json.dumps(refined_payload)]
        )
        extractor.chat_model = None
        extractor.structured_llm = None

        result = extractor.verify_and_enhance(sql="SELECT 1", deterministic_draft=payload)
        self.assertEqual(extractor.chat_adapter.calls, 2)
        self.assertEqual(result["output_columns"][0]["expression"], "a.id + b.id")

    def test_verify_only_calls_llm_once_when_refinement_disabled(self):
        payload = {
            "ctes": [],
            "output_columns": [
                {
                    "alias": "x",
                    "expression": "a.id",
                    "dependencies": [{"table_alias": "a", "column": "id"}],
                    "aggregate": False,
                    "window_function": False,
                }
            ],
            "filters": [],
            "joins": [],
            "group_by_columns": [],
        }

        extractor = SQL2GraphLLMExtractor.__new__(SQL2GraphLLMExtractor)
        extractor.max_retries = 1
        extractor.enable_refinement = False
        extractor.verification_system_prompt = "system"
        extractor.enhancement_system_prompt = "system"
        extractor.chat_adapter = _DummyChatAdapter([json.dumps(payload)])
        extractor.chat_model = None
        extractor.structured_llm = None

        result = extractor.verify_and_enhance(sql="SELECT 1", deterministic_draft=payload)
        self.assertEqual(extractor.chat_adapter.calls, 1)
        self.assertEqual(result["output_columns"][0]["expression"], "a.id")

    def test_enhance_retries_transient_provider_errors(self):
        payload = {
            "ctes": [],
            "output_columns": [
                {
                    "alias": "x",
                    "expression": "a.id",
                    "dependencies": [{"table_alias": "a", "column": "id"}],
                    "aggregate": False,
                    "window_function": False,
                }
            ],
            "filters": [],
            "joins": [],
            "group_by_columns": [],
        }

        class _FlakyEnhanceAdapter(_DummyChatAdapter):
            def invoke_messages(self, messages):
                if self.calls == 0:
                    self.calls += 1
                    raise RuntimeError(
                        "Bad request: {'message': 'This model is busy, please try again later.', "
                        "'type': 'server_error', 'code': 'completion_error'}"
                    )
                return super().invoke_messages(messages)

        extractor = SQL2GraphLLMExtractor.__new__(SQL2GraphLLMExtractor)
        extractor.max_retries = 3
        extractor.enable_refinement = True
        extractor.verification_system_prompt = "system"
        extractor.enhancement_system_prompt = "system"
        extractor.chat_adapter = _FlakyEnhanceAdapter([json.dumps(payload), json.dumps(payload)])
        extractor.chat_model = None
        extractor.structured_llm = None

        with patch("Classes.sql2graph_classes.time.sleep", return_value=None):
            result = extractor.enhance(sql="SELECT 1", verified_payload=payload)

        self.assertEqual(extractor.chat_adapter.calls, 2)
        self.assertEqual(result["output_columns"][0]["expression"], "a.id")

    def test_enhance_marks_busy_error_as_transient(self):
        extractor = SQL2GraphLLMExtractor.__new__(SQL2GraphLLMExtractor)
        extractor.max_retries = 1
        extractor.enable_refinement = True
        extractor.enhancement_system_prompt = "system"

        class _BusyAdapter(_DummyChatAdapter):
            def invoke_messages(self, messages):
                self.calls += 1
                raise RuntimeError("This model is busy, please try again later.")

        extractor.chat_adapter = _BusyAdapter([])
        extractor.chat_model = None
        extractor.structured_llm = None

        result = extractor.enhance(
            sql="SELECT 1",
            verified_payload={
                "ctes": [],
                "output_columns": [],
                "filters": [],
                "joins": [],
                "group_by_columns": [],
            },
        )
        self.assertIn("error", result)
        self.assertTrue(result.get("transient"))

    def test_extract_supports_legacy_adapter_without_invoke_messages(self):
        payload = {
            "ctes": [],
            "output_columns": [
                {
                    "alias": "x",
                    "expression": "a.id",
                    "dependencies": [{"table_alias": "a", "column": "id"}],
                    "aggregate": False,
                    "window_function": False,
                }
            ],
            "filters": [],
            "joins": [],
            "group_by_columns": [],
        }
        extractor = SQL2GraphLLMExtractor.__new__(SQL2GraphLLMExtractor)
        extractor.max_retries = 1
        extractor.enable_refinement = False
        extractor.cache = None
        extractor.prompt_version = "v2.1"
        extractor.model = "test-model"
        extractor.system_prompt = "system"
        extractor.verification_system_prompt = "system"
        extractor.chat_adapter = _DummyLegacyAdapter([json.dumps(payload)])
        extractor.chat_model = None
        extractor.structured_llm = None

        result = extractor.extract(sql="SELECT 1")
        self.assertEqual(extractor.chat_adapter.calls, 1)
        self.assertEqual(result["output_columns"][0]["alias"], "x")

    def test_pipeline_keeps_llm_filter_output_without_deterministic_override(self):
        parser = SQL2GraphParser()
        if not parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        sql = (
            "SELECT c.name FROM customers c "
            "JOIN orders o ON c.id = o.customer_id "
            "WHERE c.active = true"
        )
        fictional_payload = {
            "ctes": [],
            "output_columns": [
                {
                    "alias": "name",
                    "expression": "c.name",
                    "dependencies": [{"table_alias": "c", "column": "name"}],
                    "aggregate": False,
                    "window_function": False,
                }
            ],
            "filters": [
                {
                    "clause": "WHERE",
                    "condition": "fictional_condition = 1",
                    "columns_used": [{"table_alias": "f", "column": "fake"}],
                }
            ],
            "joins": [],
            "group_by_columns": [],
        }
        from Classes.sql2graph_classes import SQL2GraphPipeline

        pipeline = SQL2GraphPipeline(llm_extractor=_DummyExtractor(fictional_payload), parser=parser)
        result = pipeline.run(sql=sql, include_visualization=False)
        self.assertNotIn("error", result)
        self.assertEqual(result["extraction"]["filters"][0]["clause"], "WHERE")
        self.assertEqual(result["extraction"]["filters"][0]["condition"], "fictional_condition = 1")

    def test_pipeline_connects_aliased_cte_outputs_to_main_query(self):
        """Spec appendix: output.total must trace back to orders.amount through the aliased CTE."""
        import networkx as nx

        from Classes.sql2graph_classes import SQL2GraphPipeline

        parser = SQL2GraphParser()
        if not parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        sql = """
        WITH recent_orders AS (
            SELECT customer_id, SUM(amount) AS total
            FROM orders
            WHERE order_date > '2025-01-01'
            GROUP BY customer_id
        )
        SELECT c.name, r.total
        FROM customers c
        JOIN recent_orders r ON c.id = r.customer_id
        WHERE c.active = true
        """
        extraction = json.loads(json.dumps(SAMPLE_EXTRACTION))
        extraction["ctes"][0]["alias"] = "recent_orders"

        pipeline = SQL2GraphPipeline(llm_extractor=_DummyExtractor(extraction), parser=parser)
        result = pipeline.run(sql=sql, include_visualization=False)
        self.assertNotIn("error", result)

        graph = SQL2GraphVisualizer.graph_from_node_link(result["graph"])
        self.assertIn("recent_orders.total", graph.nodes)
        self.assertIn("output.total", graph.nodes)
        self.assertIn("orders.amount", graph.nodes)
        total_col = next(
            col for col in result["extraction"]["output_columns"] if col["alias"] == "total"
        )
        self.assertEqual(total_col["dependencies"], [{"table_alias": "orders", "column": "amount"}])
        self.assertTrue(nx.has_path(graph, "orders.amount", "output.total"))

    def test_materialize_transitive_derived_from_adds_shortcut_edges(self):
        builder = SQL2GraphBuilder()
        builder.build(SAMPLE_EXTRACTION)
        added = builder.materialize_transitive_derived_from()
        self.assertGreater(added, 0)
        self.assertTrue(builder.graph.has_edge("orders.amount", "output.total"))

    def test_pipeline_includes_metadata_and_transitive_lineage(self):
        from Classes.sql2graph_classes import SQL2GraphPipeline

        parser = SQL2GraphParser()
        if not parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        sql = """
        WITH recent_orders AS (
            SELECT customer_id, SUM(amount) AS total
            FROM orders
            GROUP BY customer_id
        )
        SELECT c.name, r.total
        FROM customers c
        JOIN recent_orders r ON c.id = r.customer_id
        """
        extraction = json.loads(json.dumps(SAMPLE_EXTRACTION))
        extraction["ctes"][0]["alias"] = "recent_orders"

        pipeline = SQL2GraphPipeline(llm_extractor=_DummyExtractor(extraction), parser=parser)
        result = pipeline.run(sql=sql, include_visualization=False)

        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["spec_version"], "2.1")
        self.assertIn("source_sql_hash", result["graph"]["metadata"])
        self.assertTrue(result["metadata"].get("is_dag", result["graph"]["metadata"].get("is_dag")))

        graph = SQL2GraphVisualizer.graph_from_node_link(result["graph"])
        self.assertTrue(graph.has_edge("orders.amount", "output.total"))

    def test_pipeline_returns_subgraphs_payload(self):
        from Classes.sql2graph_classes import SQL2GraphPipeline

        parser = SQL2GraphParser()
        if not parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        sql = """
        WITH r AS (
            SELECT customer_id, SUM(amount) AS total
            FROM orders
            GROUP BY customer_id
        )
        SELECT c.name, r.total
        FROM customers c
        JOIN r ON c.id = r.customer_id
        """
        pipeline = SQL2GraphPipeline(llm_extractor=_DummyExtractor(SAMPLE_EXTRACTION), parser=parser)
        result = pipeline.run(sql=sql, include_visualization=False)

        self.assertIn("subgraphs", result)
        self.assertTrue(result["subgraphs"])
        cte_blocks = [block for block in result["subgraphs"] if block["type"] == "cte"]
        self.assertTrue(cte_blocks)
        self.assertIn("graph", cte_blocks[0])

    def test_build_deterministic_extraction_from_simplify(self):
        parser = SQL2GraphParser()
        if not parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        sql = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
        simplified = parser.simplify(sql)
        draft = parser.build_deterministic_extraction(simplified)

        self.assertEqual(len(draft["output_columns"]), 2)
        aliases = {col["alias"] for col in draft["output_columns"]}
        self.assertEqual(aliases, {"name", "total"})
        self.assertTrue(any(col["dependencies"] for col in draft["output_columns"]))

    def test_materialize_output_dependencies_strips_llm_cte_passthrough(self):
        parser = SQL2GraphParser()
        if not parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        sql = Path(__file__).resolve().parents[1].joinpath("data/DDLs_10.txt").read_text().split(";")[0].strip()
        simplified = parser.simplify(sql, dialect="postgres")
        deterministic = parser.build_deterministic_extraction(simplified, dialect="postgres")
        llm_like = json.loads(json.dumps(deterministic))
        attr = next(col for col in llm_like["output_columns"] if col["alias"] == "attr_name")
        attr["dependencies"] = [{"table_alias": "pprb_attr_val", "column": "attr_name"}]
        end_dt = next(col for col in llm_like["output_columns"] if col["alias"] == "end_dt")
        end_dt["dependencies"] = [
            {"table_alias": "a_agr_collat_mkt_period", "column": "end_dt"},
            {"table_alias": "mkt_prd", "column": "end_dt"},
            {"table_alias": "a", "column": "end_dt"},
        ]

        resolved = parser.overlay_deterministic_column_lineage(llm_like, deterministic)
        resolved = parser._materialize_output_dependencies(resolved, simplified)
        attr = next(col for col in resolved["output_columns"] if col["alias"] == "attr_name")
        self.assertEqual(attr["dependencies"], [])
        end_dt = next(col for col in resolved["output_columns"] if col["alias"] == "end_dt")
        physical_tables = {dep.get("table_alias") for dep in end_dt["dependencies"]}
        self.assertEqual(
            physical_tables,
            {
                "s_grnplm_vd_t_bvd_db_dmcl.a_agr_collat_mkt_period",
                "s_grnplm_vd_t_bvd_db_dmcl.a_agr_collat_qlty_period",
            },
        )

    def test_pipeline_overlays_sqlglot_lineage_after_llm(self):
        from Classes.sql2graph_classes import SQL2GraphPipeline

        parser = SQL2GraphParser()
        if not parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        sql = Path(__file__).resolve().parents[1].joinpath("data/DDLs_10.txt").read_text().split(";")[0].strip()
        deterministic = parser.build_deterministic_extraction(
            parser.simplify(sql, dialect="postgres"),
            dialect="postgres",
        )
        llm_payload = json.loads(json.dumps(deterministic))
        end_dt = next(col for col in llm_payload["output_columns"] if col["alias"] == "end_dt")
        end_dt["dependencies"] = [
            {"table_alias": "a_agr_collat_mkt_period", "column": "end_dt"},
            {"table_alias": "mkt_prd", "column": "end_dt"},
            {"table_alias": "a", "column": "end_dt"},
        ]
        pipeline = SQL2GraphPipeline(llm_extractor=_DummyExtractor(llm_payload), parser=parser)
        result = pipeline.run(sql=sql, dialect="postgres", use_llm_verify=True, use_llm_enhance=True)
        end_dt = next(col for col in result["extraction"]["output_columns"] if col["alias"] == "end_dt")
        physical_tables = {dep.get("table_alias") for dep in end_dt["dependencies"]}
        self.assertEqual(
            physical_tables,
            {
                "s_grnplm_vd_t_bvd_db_dmcl.a_agr_collat_mkt_period",
                "s_grnplm_vd_t_bvd_db_dmcl.a_agr_collat_qlty_period",
            },
        )
        parser = SQL2GraphParser()
        if not parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        sql = Path(__file__).resolve().parents[1].joinpath("data/DDLs_10.txt").read_text().split(";")[0].strip()
        extraction = parser.build_deterministic_extraction(parser.simplify(sql, dialect="postgres"), dialect="postgres")
        attr = next(col for col in extraction["output_columns"] if col["alias"] == "attr_name")
        self.assertEqual(attr["dependencies"], [])

    def test_pipeline_sqlglot_first_without_llm(self):
        from Classes.sql2graph_classes import SQL2GraphPipeline

        parser = SQL2GraphParser()
        if not parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        sql = "SELECT a, b FROM t"
        pipeline = SQL2GraphPipeline(parser=parser)
        result = pipeline.run(sql=sql, use_llm_verify=False, use_llm_enhance=False)

        self.assertEqual(result["pipeline_stage"], "deterministic")
        self.assertIn("deterministic_extraction", result)
        self.assertEqual(len(result["deterministic_extraction"]["output_columns"]), 2)
        steps = result.get("pipeline_steps") or {}
        self.assertEqual(steps["chunking"]["status"], "completed")
        self.assertEqual(steps["parsing"]["status"], "completed")
        self.assertEqual(steps["verifying"]["status"], "skipped")
        self.assertEqual(steps["enhancing"]["status"], "skipped")
        self.assertEqual(steps["combining"]["status"], "completed")
        self.assertIn("chunks", result)

    def test_diff_extraction_reports_column_field_changes(self):
        from Classes.sql2graph_classes import SQL2GraphPipeline

        before = {
            "output_columns": [
                {"alias": "a", "expression": "x", "dependencies": []},
            ],
            "filters": [],
            "joins": [],
            "ctes": [],
        }
        after = {
            "output_columns": [
                {"alias": "a", "expression": "y", "dependencies": [{"table_alias": "t", "column": "x"}]},
            ],
            "filters": [{"clause": "WHERE"}],
            "joins": [],
            "ctes": [],
        }
        diff = SQL2GraphPipeline.diff_extraction(before, after)
        self.assertEqual(diff["change_count"], 3)
        fields = {change.get("field") for change in diff["changes"] if change.get("field")}
        self.assertIn("expression", fields)
        self.assertIn("dependencies", fields)

    def test_pipeline_verify_only_skips_enhance(self):
        from Classes.sql2graph_classes import SQL2GraphPipeline

        parser = SQL2GraphParser()
        if not parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        class _TrackingExtractor:
            enable_refinement = True

            def __init__(self):
                self.verify_calls = 0
                self.enhance_calls = 0

            def verify(self, deterministic_draft=None, **_kwargs):
                self.verify_calls += 1
                return deterministic_draft

            def enhance(self, verified_payload=None, **_kwargs):
                self.enhance_calls += 1
                return verified_payload

        sql = "SELECT a, b FROM t"
        extractor = _TrackingExtractor()
        pipeline = SQL2GraphPipeline(llm_extractor=extractor, parser=parser)
        result = pipeline.run(sql=sql, use_llm_verify=True, use_llm_enhance=False)
        self.assertEqual(extractor.verify_calls, 1)
        self.assertEqual(extractor.enhance_calls, 0)
        self.assertEqual(result["pipeline_steps"]["enhancing"]["status"], "skipped")

    def test_pipeline_enhance_without_verify_runs_on_deterministic(self):
        from Classes.sql2graph_classes import SQL2GraphPipeline

        parser = SQL2GraphParser()
        if not parser.sqlglot_available:
            self.skipTest("sqlglot not installed in runtime")

        class _TrackingExtractor:
            enable_refinement = True

            def __init__(self):
                self.verify_calls = 0
                self.enhance_calls = 0

            def verify(self, deterministic_draft=None, **_kwargs):
                self.verify_calls += 1
                return deterministic_draft

            def enhance(self, verified_payload=None, **_kwargs):
                self.enhance_calls += 1
                return verified_payload

        sql = "SELECT a, b FROM t"
        extractor = _TrackingExtractor()
        pipeline = SQL2GraphPipeline(llm_extractor=extractor, parser=parser)
        result = pipeline.run(sql=sql, use_llm_verify=False, use_llm_enhance=True)
        self.assertEqual(extractor.verify_calls, 0)
        self.assertEqual(extractor.enhance_calls, 1)
        self.assertEqual(result["pipeline_steps"]["verifying"]["status"], "skipped")
        self.assertEqual(result["pipeline_steps"]["enhancing"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
