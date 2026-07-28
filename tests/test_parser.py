"""Tests for SQLParser."""

import unittest

import sqlglot

from Classes.pipeline.core.parser import SQLParser
from Classes.pipeline.exceptions import ParsingError


class TestSQLParser(unittest.TestCase):
    def setUp(self):
        self.parser = SQLParser(dialect="spark")

    def test_parse_simple_select(self):
        tree = self.parser.parse("SELECT a, b FROM t")
        self.assertIsInstance(tree, sqlglot.exp.Expression)

    def test_parse_join(self):
        sql = (
            "SELECT u.name FROM users u "
            "JOIN orders o ON u.id = o.user_id"
        )
        tree = self.parser.parse(sql)
        self.assertIsNotNone(tree)

    def test_parse_cte(self):
        sql = (
            "WITH r AS (SELECT id FROM t) "
            "SELECT * FROM r"
        )
        tree = self.parser.parse(sql)
        self.assertIsNotNone(tree)

    def test_parse_subquery(self):
        sql = "SELECT (SELECT MAX(v) FROM t) AS m FROM u"
        tree = self.parser.parse(sql)
        self.assertIsNotNone(tree)

    def test_invalid_sql_raises(self):
        with self.assertRaises(ParsingError):
            self.parser.parse("SELECT FROM")

    def test_empty_sql_raises(self):
        with self.assertRaises(ParsingError):
            self.parser.parse("   ")

    def test_dialect_variation(self):
        parser = SQLParser(dialect="postgres")
        tree = parser.parse("SELECT 1 AS one")
        self.assertIsNotNone(tree)


if __name__ == "__main__":
    unittest.main()
