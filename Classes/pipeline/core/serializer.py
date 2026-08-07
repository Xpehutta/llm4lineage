"""Serialise a sqlglot AST to a nested dictionary."""

import logging
from typing import Any

from sqlglot import exp

from Classes.pipeline.exceptions import SerializationError

logger = logging.getLogger(__name__)


class ASTSerializer:
    """Serialise a sqlglot AST to a nested dictionary."""

    def __init__(self, max_depth: int = 50):
        self.max_depth = max_depth

    def serialize(self, tree: exp.Expression) -> dict[str, Any]:
        """Return a JSON-safe dict representation of *tree*.

        Raises:
            SerializationError: If traversal exceeds *max_depth*.
        """
        try:
            return self._serialize_iterative(tree)
        except SerializationError:
            raise
        except Exception as exc:
            raise SerializationError(
                f"AST serialization failed: {exc}"
            ) from exc

    def _serialize_iterative(self, root: exp.Expression) -> dict[str, Any]:
        """Stack-based DFS to avoid RecursionError on deep ASTs."""
        root_dict = self._make_node_dict(root)
        stack: list[tuple[exp.Expression, dict[str, Any], int]] = [(root, root_dict, 0)]

        while stack:
            node, node_dict, depth = stack.pop()

            if depth > self.max_depth:
                node_dict["properties"]["_truncated"] = True
                logger.warning(
                    "AST depth %d exceeded max_depth=%d; truncating subtree.",
                    depth,
                    self.max_depth,
                )
                continue

            for key, value in node.args.items():
                if key in node_dict["properties"]:
                    continue

                if isinstance(value, exp.Expression):
                    child_dict = self._make_node_dict(value)
                    node_dict["children"].append(child_dict)
                    stack.append((value, child_dict, depth + 1))
                elif isinstance(value, list):
                    primitives = []
                    for item in value:
                        if isinstance(item, exp.Expression):
                            child_dict = self._make_node_dict(item)
                            node_dict["children"].append(child_dict)
                            stack.append((item, child_dict, depth + 1))
                        elif item is not None:
                            primitives.append(self._to_json_safe(item))
                    if primitives:
                        node_dict["properties"][key] = primitives
                elif value is not None:
                    node_dict["properties"][key] = self._to_json_safe(value)

        return root_dict

    @staticmethod
    def _make_node_dict(node: exp.Expression) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": type(node).__name__,
            "properties": {},
            "children": [],
        }

        if isinstance(node, exp.Table):
            result["properties"]["name"] = node.name
            result["properties"]["alias"] = node.alias_or_name
        elif isinstance(node, exp.Column):
            result["properties"]["name"] = node.name
            result["properties"]["table"] = node.table
        elif isinstance(node, exp.Select):
            result["properties"]["distinct"] = bool(node.args.get("distinct"))

        return result

    @staticmethod
    def _to_json_safe(value: Any) -> Any:
        """Coerce sqlglot arg values to JSON-serializable primitives."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [ASTSerializer._to_json_safe(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): ASTSerializer._to_json_safe(item)
                for key, item in value.items()
            }
        return str(value)
