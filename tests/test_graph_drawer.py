import json
import unittest
from pathlib import Path

from Classes.graph_drawer import JsonLineageDrawer

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "data" / "sqlglot_ddls10_first_snapshot.json"

CHUNK_PAYLOAD = {
    "chunks": [
        {"id": "cte_a", "name": "cte_a", "chunk_type": "cte", "sql": "SELECT 1"},
        {"id": "target_t", "name": "target_t", "chunk_type": "target", "sql": "target_t"},
    ],
    "links": [
        {"source": "cte_a", "target": "target_t", "link_type": "INSERT", "condition": ""},
    ],
}


SAMPLE_GRAPH = {
    "graph": {
        "nodes": [
            {"id": "orders.amount", "node_type": "source_column", "table_alias": "orders", "column": "amount"},
            {"id": "output.total", "node_type": "output_column", "alias": "total"},
        ],
        "links": [
            {"source": "orders.amount", "target": "output.total", "edge_type": "DERIVED_FROM"},
        ],
    }
}


class TestJsonLineageDrawer(unittest.TestCase):
    def test_from_sql2graph_result(self):
        drawer = JsonLineageDrawer.from_sql2graph_result(SAMPLE_GRAPH)
        summary = drawer.summary()
        self.assertEqual(summary["node_count"], 2)
        self.assertIn("source_column", summary["node_types"])
        self.assertIn("output_column", summary["node_types"])

    def test_normalize_chunk_payload(self):
        node_link = JsonLineageDrawer.normalize_to_node_link(CHUNK_PAYLOAD)
        self.assertEqual(len(node_link["nodes"]), 2)
        self.assertEqual(node_link["links"][0]["edge_type"], "INSERT")
        self.assertEqual(node_link["nodes"][0]["node_type"], "chunk")

    def test_from_snapshot_file(self):
        if not SNAPSHOT.exists():
            self.skipTest("snapshot file missing")
        drawer = JsonLineageDrawer.from_path(SNAPSHOT)
        summary = drawer.summary()
        self.assertGreater(summary["node_count"], 0)
        self.assertIn("pprb_attr_val", summary["node_ids"])

    def test_table_lineage_list(self):
        node_link = JsonLineageDrawer.normalize_to_node_link(
            [{"target": "t2", "sources": ["t0", "t1"]}]
        )
        self.assertEqual(len(node_link["nodes"]), 3)
        self.assertEqual(len(node_link["links"]), 2)


if __name__ == "__main__":
    unittest.main()
