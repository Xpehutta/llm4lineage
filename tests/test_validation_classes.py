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


if __name__ == "__main__":
    unittest.main()
