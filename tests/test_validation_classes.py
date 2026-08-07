import unittest

from Classes.validation_classes import SQLLineageValidator


class DummyExtractor:
    def __init__(self, result):
        self.result = result

    def extract(self, _sql_query):
        return self.result


class TestSQLLineageValidator(unittest.TestCase):
    def test_validate_target_name(self):
        self.assertEqual(SQLLineageValidator.validate_target_name("schema.table")[0], True)
        self.assertEqual(SQLLineageValidator.validate_target_name("table_only")[0], False)
        self.assertEqual(SQLLineageValidator.validate_target_name("")[0], False)

    def test_validate_unique_sources_case_insensitive(self):
        ok, _ = SQLLineageValidator.validate_unique_sources(["a.b", "c.d"])
        self.assertTrue(ok)

        ok, message = SQLLineageValidator.validate_unique_sources(["a.b", "A.B"])
        self.assertFalse(ok)
        self.assertIn("Duplicate sources found", message)

    def test_validate_no_derived_tables(self):
        ok, _ = SQLLineageValidator.validate_no_derived_tables(["schema.table"])
        self.assertTrue(ok)

        ok, message = SQLLineageValidator.validate_no_derived_tables(["t1"])
        self.assertFalse(ok)
        self.assertIn("Derived table detected", message)

    def test_validate_no_derived_tables_rejects_target_in_sources_case_insensitive(self):
        ok, message = SQLLineageValidator.validate_no_derived_tables(
            ["RAW.ORDERS", "analytics.sales"],
            target="Analytics.Sales",
        )
        self.assertFalse(ok)
        self.assertIn("should not appear in sources", message)

    def test_precision_recall_f1(self):
        expected = {"sources": ["a.b", "c.d"]}
        actual = {"sources": ["a.b", "x.y"]}
        precision, recall, f1 = SQLLineageValidator.calculate_precision_recall_f1(expected, actual)
        self.assertAlmostEqual(precision, 0.5)
        self.assertAlmostEqual(recall, 0.5)
        self.assertAlmostEqual(f1, 0.5)

    def test_run_comprehensive_validation_success_with_metrics(self):
        extractor = DummyExtractor({"target": "analytics.sales", "sources": ["raw.orders"]})
        expected = {"target": "analytics.sales", "sources": ["raw.orders"]}
        result = SQLLineageValidator.run_comprehensive_validation(extractor, "SELECT 1", expected)

        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("metrics", result)
        self.assertEqual(result["metrics"]["f1_score"], 1.0)

    def test_run_comprehensive_validation_fails_on_format(self):
        extractor = DummyExtractor("bad result")
        result = SQLLineageValidator.run_comprehensive_validation(extractor, "SELECT 1")

        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["validation_type"], "format")


    def test_calculate_edge_f1_identical_graphs(self):
        graph = {
            "links": [
                {"source": "a.x", "target": "output.y", "edge_type": "DERIVED_FROM"},
                {"source": "b.z", "target": "output.y", "edge_type": "DERIVED_FROM"},
            ]
        }
        metrics = SQLLineageValidator.calculate_edge_f1(graph, graph)
        self.assertEqual(metrics["f1"], 1.0)


class TestValidateOutputFormat(unittest.TestCase):
    def test_accepts_a_well_formed_result(self):
        self.assertEqual(
            SQLLineageValidator.validate_output_format({"target": "a.b", "sources": []}),
            (True, "Valid format"),
        )

    def test_rejects_a_non_dict(self):
        ok, message = SQLLineageValidator.validate_output_format(["a.b"])
        self.assertFalse(ok)
        self.assertEqual(message, "Result should be a dictionary")

    def test_rejects_a_missing_target(self):
        ok, message = SQLLineageValidator.validate_output_format({"sources": []})
        self.assertFalse(ok)
        self.assertEqual(message, "Missing 'target' field")

    def test_rejects_a_missing_sources(self):
        ok, message = SQLLineageValidator.validate_output_format({"target": "a.b"})
        self.assertFalse(ok)
        self.assertEqual(message, "Missing 'sources' field")

    def test_rejects_a_non_string_target(self):
        ok, message = SQLLineageValidator.validate_output_format({"target": 1, "sources": []})
        self.assertFalse(ok)
        self.assertEqual(message, "'target' should be a string")

    def test_rejects_non_list_sources(self):
        ok, message = SQLLineageValidator.validate_output_format(
            {"target": "a.b", "sources": "raw.orders"}
        )
        self.assertFalse(ok)
        self.assertEqual(message, "'sources' should be a list")


class TestValidateNames(unittest.TestCase):
    def test_target_with_an_empty_schema_is_rejected(self):
        ok, message = SQLLineageValidator.validate_target_name(".table")
        self.assertFalse(ok)
        self.assertIn("missing schema or table name", message)

    def test_empty_source_list_is_valid(self):
        self.assertEqual(
            SQLLineageValidator.validate_source_names([]),
            (True, "No sources (valid case)"),
        )

    def test_every_malformed_source_is_reported(self):
        ok, message = SQLLineageValidator.validate_source_names(
            [42, "", "unqualified", "raw.", "raw.orders"]
        )
        self.assertFalse(ok)
        self.assertIn("Source 0 is not a string", message)
        self.assertIn("Source 1 is empty", message)
        self.assertIn("Source 'unqualified' should be fully qualified", message)
        self.assertIn("Source 'raw.' has missing schema or table name", message)
        self.assertEqual(message.count(";"), 3)

    def test_fully_qualified_names_pass(self):
        self.assertEqual(
            SQLLineageValidator.validate_fully_qualified_names(["a.b", "c.d"]),
            (True, "All names are fully qualified"),
        )

    def test_unqualified_and_empty_names_are_reported(self):
        ok, message = SQLLineageValidator.validate_fully_qualified_names(["a.b", "c", ""])
        self.assertFalse(ok)
        self.assertIn("Name 'c' is not fully qualified", message)
        self.assertIn("Name '' is not fully qualified", message)

    def test_alias_suffixed_source_is_reported_once(self):
        ok, message = SQLLineageValidator.validate_no_derived_tables(["orders_alias"])
        self.assertFalse(ok)
        self.assertEqual(message, "Derived table detected: orders_alias")


class TestMetrics(unittest.TestCase):
    def test_two_empty_source_sets_are_a_perfect_match(self):
        self.assertEqual(
            SQLLineageValidator.calculate_precision_recall_f1({"sources": []}, {"sources": []}),
            (1.0, 1.0, 1.0),
        )

    def test_no_overlap_scores_zero(self):
        precision, recall, f1 = SQLLineageValidator.calculate_precision_recall_f1(
            {"sources": ["a.b"]}, {"sources": ["c.d"]}
        )
        self.assertEqual((precision, recall, f1), (0, 0, 0))

    def test_edge_f1_on_two_empty_graphs_is_one(self):
        self.assertEqual(
            SQLLineageValidator.calculate_edge_f1({}, {}),
            {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        )

    def test_edge_f1_counts_true_and_false_positives(self):
        expected = {"links": [{"source": "a", "target": "b", "edge_type": "DERIVED_FROM"}]}
        actual = {
            "edges": [
                {"source": "a", "target": "b", "type": "DERIVED_FROM"},
                {"source": "a", "target": "c", "type": "DERIVED_FROM"},
            ]
        }
        metrics = SQLLineageValidator.calculate_edge_f1(expected, actual)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 0))
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 1.0)

    def test_edge_f1_ignores_links_missing_an_endpoint_or_type(self):
        graph = {
            "links": [
                {"source": "a", "target": "b", "edge_type": "DERIVED_FROM"},
                {"source": "", "target": "b", "edge_type": "DERIVED_FROM"},
                {"source": "a", "target": "c"},
            ]
        }
        metrics = SQLLineageValidator.calculate_edge_f1(graph, graph)
        self.assertEqual(metrics["tp"], 1)


class TestRunComprehensiveValidation(unittest.TestCase):
    """Each validation stage short-circuits with its own ``validation_type``."""

    def _validate(self, result):
        return SQLLineageValidator.run_comprehensive_validation(DummyExtractor(result), "SELECT 1")

    def test_target_stage_failure(self):
        outcome = self._validate({"target": "unqualified", "sources": []})
        self.assertEqual(outcome["validation_type"], "target")
        self.assertEqual(outcome["status"], "FAILED")

    def test_sources_stage_failure(self):
        outcome = self._validate({"target": "a.b", "sources": ["unqualified"]})
        self.assertEqual(outcome["validation_type"], "sources")

    def test_derived_tables_stage_failure(self):
        outcome = self._validate({"target": "a.b", "sources": ["x.t1_alias"]})
        self.assertEqual(outcome["validation_type"], "derived_tables")

    def test_uniqueness_stage_failure(self):
        outcome = self._validate({"target": "a.b", "sources": ["raw.orders", "RAW.ORDERS"]})
        self.assertEqual(outcome["validation_type"], "uniqueness")

    def test_success_without_expected_result_has_no_metrics(self):
        outcome = self._validate({"target": "a.b", "sources": ["raw.orders"]})
        self.assertEqual(outcome["status"], "SUCCESS")
        self.assertNotIn("metrics", outcome)


if __name__ == "__main__":
    unittest.main()
