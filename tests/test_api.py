"""REST API smoke tests (FastAPI TestClient)."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from Web.api.main import app
from Web.api.store import STORE


def _sample_graph() -> dict:
    return {
        "directed": True,
        "multigraph": True,
        "graph": {},
        "nodes": [
            {
                "id": "orders.amount",
                "node_type": "source_column",
                "physical_table": "orders",
                "column": "amount",
            },
            {"id": "output.total", "node_type": "output_column", "alias": "total"},
        ],
        "links": [
            {
                "key": 0,
                "source": "orders.amount",
                "target": "output.total",
                "edge_type": "DERIVED_FROM",
                "confidence": 1.0,
                "provenance": "deterministic",
                "verified": True,
            }
        ],
    }


class TestLineageApi(unittest.TestCase):
    def setUp(self) -> None:
        STORE.graphs.clear()
        STORE.column_meta.clear()
        self.client = TestClient(app)
        self.client.post(
            "/lineage/sales_summary",
            json={
                "graph": _sample_graph(),
                "column_meta": {
                    "total": {
                        "is_pii": False,
                        "owner": "analytics",
                        "description": "sum of amounts",
                    },
                    "ssn": {"is_pii": True, "owner": "compliance", "description": "tax id"},
                },
            },
        )

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_impact_returns_downstream_chain(self) -> None:
        response = self.client.get("/impact/sales_summary/orders.amount?direction=down")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["target"], "orders.amount")
        downstream_nodes = [hit["node"] for hit in body["downstream"]]
        self.assertIn("output.total", downstream_nodes)

    def test_lineage_dot_format(self) -> None:
        response = self.client.get("/lineage/sales_summary?format=dot")
        self.assertEqual(response.status_code, 200)
        text = response.text
        self.assertIn("digraph lineage", text)
        self.assertIn("orders.amount", text)
        self.assertIn("->", text)

    def test_lineage_mermaid_format(self) -> None:
        response = self.client.get("/lineage/sales_summary?format=mermaid")
        self.assertEqual(response.status_code, 200)
        self.assertIn("flowchart LR", response.text)

    def test_coverage(self) -> None:
        response = self.client.get("/coverage")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["objects"], 1)
        self.assertEqual(body["edges"], 1)
        self.assertEqual(body["verified_edges"], 1)

    def test_pii(self) -> None:
        response = self.client.get("/pii")
        self.assertEqual(response.status_code, 200)
        columns = response.json()["columns"]
        self.assertEqual(len(columns), 1)
        self.assertEqual(columns[0]["column"], "ssn")

    def test_unknown_object_404(self) -> None:
        response = self.client.get("/lineage/missing")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
