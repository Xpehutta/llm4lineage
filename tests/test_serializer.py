"""Tests for ASTSerializer."""

import unittest

from Classes.pipeline.core.parser import SQLParser
from Classes.pipeline.core.serializer import ASTSerializer


class TestASTSerializer(unittest.TestCase):
    def setUp(self):
        self.parser = SQLParser()
        self.serializer = ASTSerializer()

    def test_serialize_structure(self):
        tree = self.parser.parse("SELECT a FROM t")
        result = self.serializer.serialize(tree)
        self.assertIn("type", result)
        self.assertIn("properties", result)
        self.assertIn("children", result)

    def test_preserves_boolean_distinct(self):
        tree = self.parser.parse("SELECT DISTINCT a FROM t")
        result = self.serializer.serialize(tree)
        select_nodes = [
            node for node in self._walk_nodes(result) if node["type"] == "Select"
        ]
        self.assertTrue(select_nodes)
        distinct = select_nodes[0]["properties"].get("distinct")
        self.assertIsInstance(distinct, bool)

    def test_table_convenience_properties(self):
        tree = self.parser.parse("SELECT t.a FROM my_table t")
        result = self.serializer.serialize(tree)
        tables = [
            node for node in self._walk_nodes(result) if node["type"] == "Table"
        ]
        self.assertTrue(tables)
        self.assertEqual(tables[0]["properties"]["name"], "my_table")

    def test_max_depth_truncation(self):
        serializer = ASTSerializer(max_depth=1)
        tree = self.parser.parse(
            "SELECT a FROM (SELECT b FROM (SELECT c FROM t) x) y"
        )
        result = serializer.serialize(tree)
        truncated = [
            node
            for node in self._walk_nodes(result)
            if node["properties"].get("_truncated")
        ]
        self.assertTrue(truncated)

    def test_list_valued_args(self):
        tree = self.parser.parse("SELECT a, b, c FROM t")
        result = self.serializer.serialize(tree)
        self.assertIsInstance(result["children"], list)

    def _walk_nodes(self, node: dict) -> list:
        nodes = [node]
        for child in node.get("children", []):
            nodes.extend(self._walk_nodes(child))
        return nodes


if __name__ == "__main__":
    unittest.main()
