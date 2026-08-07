"""Tests for ASTSerializer."""

import unittest
from unittest.mock import patch

from sqlglot import exp

from Classes.pipeline.core.parser import SQLParser
from Classes.pipeline.core.serializer import ASTSerializer
from Classes.pipeline.exceptions import SerializationError


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

    def test_mixed_list_args_split_into_children_and_properties(self):
        """Expressions in a list arg become children; primitives become properties."""
        tree = self.parser.parse("SELECT a FROM t")
        tree.args["mixed"] = [exp.Literal.string("kept"), "flag", None, 7]

        result = self.serializer.serialize(tree)

        self.assertEqual(result["properties"]["mixed"], ["flag", 7])
        self.assertIn("Literal", [child["type"] for child in result["children"]])

    def test_list_args_with_no_primitives_add_no_property(self):
        tree = self.parser.parse("SELECT a FROM t")
        tree.args["expressions_only"] = [exp.Literal.string("kept")]

        result = self.serializer.serialize(tree)

        self.assertNotIn("expressions_only", result["properties"])

    def test_column_convenience_properties(self):
        tree = self.parser.parse("SELECT t.a FROM my_table t")
        result = self.serializer.serialize(tree)
        columns = [node for node in self._walk_nodes(result) if node["type"] == "Column"]
        self.assertTrue(columns)
        self.assertEqual(columns[0]["properties"]["name"], "a")
        self.assertEqual(columns[0]["properties"]["table"], "t")

    def test_unexpected_failures_are_wrapped_in_a_serialization_error(self):
        tree = self.parser.parse("SELECT a FROM t")
        with patch.object(ASTSerializer, "_make_node_dict", side_effect=RuntimeError("boom")):
            with self.assertRaises(SerializationError) as ctx:
                self.serializer.serialize(tree)
        self.assertIn("AST serialization failed: boom", str(ctx.exception))

    def test_an_existing_serialization_error_is_not_double_wrapped(self):
        tree = self.parser.parse("SELECT a FROM t")
        original = SerializationError("already specific")
        with patch.object(ASTSerializer, "_make_node_dict", side_effect=original):
            with self.assertRaises(SerializationError) as ctx:
                self.serializer.serialize(tree)
        self.assertIs(ctx.exception, original)

    def _walk_nodes(self, node: dict) -> list:
        nodes = [node]
        for child in node.get("children", []):
            nodes.extend(self._walk_nodes(child))
        return nodes


class TestToJsonSafe(unittest.TestCase):
    """Arg values must survive `json.dumps`, whatever sqlglot puts in them."""

    def test_primitives_pass_through_unchanged(self):
        for value in (None, True, 3, 1.5, "text"):
            self.assertEqual(ASTSerializer._to_json_safe(value), value)

    def test_sequences_are_converted_element_wise(self):
        self.assertEqual(ASTSerializer._to_json_safe(["a", 1]), ["a", 1])
        self.assertEqual(ASTSerializer._to_json_safe(("a", 1)), ["a", 1])

    def test_dict_keys_are_stringified(self):
        self.assertEqual(ASTSerializer._to_json_safe({1: "a"}), {"1": "a"})

    def test_nested_structures_are_converted_recursively(self):
        self.assertEqual(
            ASTSerializer._to_json_safe({"k": [{"inner": object.__name__}]}),
            {"k": [{"inner": "object"}]},
        )

    def test_arbitrary_objects_fall_back_to_their_string_form(self):
        class Opaque:
            def __str__(self):
                return "opaque-value"

        self.assertEqual(ASTSerializer._to_json_safe(Opaque()), "opaque-value")


if __name__ == "__main__":
    unittest.main()
