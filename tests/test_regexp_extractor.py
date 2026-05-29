import unittest

from Classes.regexp_extractor import RegexSQLExtractor


class TestRegexSQLExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = RegexSQLExtractor()

    def test_extract_target_and_sources(self):
        sql = """
        INSERT INTO s_grnplm_dm.sales.target_table
        SELECT o.id, c.name
        FROM s_grnplm_src.raw.orders o
        JOIN s_grnplm_src.raw.customers c ON o.customer_id = c.id
        """
        result = self.extractor.extract(sql)

        self.assertEqual(result["target"], "s_grnplm_dm.sales.target_table")
        self.assertEqual(
            result["sources"],
            ["s_grnplm_src.raw.customers", "s_grnplm_src.raw.orders"],
        )

    def test_clean_sql_removes_literals_comments_and_casts(self):
        sql = """
        -- comment
        SELECT 'literal'::text, col::bigint
        FROM s_grnplm_src.raw.orders
        /* block comment */
        """
        cleaned = self.extractor._clean_sql(sql)
        self.assertNotIn("literal", cleaned)
        self.assertNotIn("-- comment", cleaned)
        self.assertNotIn("/* block comment */", cleaned)
        self.assertNotIn("::text", cleaned)
        self.assertNotIn("::bigint", cleaned)

    def test_extract_target_from_update(self):
        sql = "UPDATE s_grnplm_dm.sales.target_table SET x = 1"
        target = self.extractor._extract_target(sql)
        self.assertEqual(target, "s_grnplm_dm.sales.target_table")

    def test_invalid_object_detection(self):
        self.assertTrue(self.extractor._is_valid_object("s_grnplm_src.raw.orders"))
        self.assertFalse(self.extractor._is_valid_object("s_grnplm_src.raw.orders.*"))


if __name__ == "__main__":
    unittest.main()
