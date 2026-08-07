"""DOT / Mermaid / HTML visualization helpers for SQL2Graph."""
from __future__ import annotations

import html
import json
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph


class SQL2GraphVisualizer:
    """Render SQL2Graph node-link JSON output."""

    EDGE_COLORS = {
        "DERIVED_FROM": "#1f77b4",
        "FILTERED_BY": "#ff7f0e",
        "USES_COLUMN": "#2ca02c",
        "JOINS_ON": "#d62728",
        "GROUPED_BY": "#9467bd",
        "ROW_FLOW_IN": "#17becf",
        "ROW_FLOW_OUT": "#17becf",
        "VALUE_FLOW": "#bcbd22",
        "AGGREGATES_ON": "#8c564b",
        "WINDOW_OVER": "#e377c2",
        "CHUNK_LINK": "#444444",
        "CONTAINS": "#999999",
        "JOIN": "#d62728",
        "INSERT": "#2ca02c",
        "UNION": "#1f77b4",
        "UNION ALL": "#1f77b4",
    }

    LINEAGE_EDGE_TYPES = frozenset(
        {
            "DERIVED_FROM",
            "FILTERED_BY",
            "USES_COLUMN",
            "GROUPED_BY",
            "JOINS_ON",
            "ROW_FLOW_IN",
            "ROW_FLOW_OUT",
            "VALUE_FLOW",
            "AGGREGATES_ON",
            "WINDOW_OVER",
            "CHUNK_LINK",
            "CONTAINS",
            "JOIN",
            "INSERT",
            "UNION",
            "UNION ALL",
            "UNION DISTINCT",
            "INTERSECT",
            "EXCEPT",
        }
    )

    HIGHLIGHT_SELECTED_COLOR = "#FF5722"
    HIGHLIGHT_LINEAGE_COLOR = "#FFC107"
    HIGHLIGHT_DIMMED_COLOR = "#E8E8E8"

    NODE_COLORS = {
        "source_column": "#90EE90",
        "output_column": "#ADD8E6",
        "filter": "#F6D186",
        "join": "#F08080",
        "chunk": "#ADD8E6",
        "union": "#87CEFA",
        "aggregate": "#DDA0DD",
        "window": "#FFB6C1",
        "transformation": "#F0E68C",
        "rowset": "#B0C4DE",
    }

    CHUNK_TYPE_COLORS = {
        "target": "#FFB6C1",
        "cte": "#DDA0DD",
        "query": "#ADD8E6",
    }

    NODE_SHAPES = {
        "source_column": "dot",
        "output_column": "box",
        "filter": "diamond",
        "join": "triangle",
    }

    NODE_TYPE_LABELS = {
        "source_column": "Source column",
        "output_column": "Output column",
        "filter": "Filter",
        "join": "Join",
    }

    @staticmethod
    def graph_from_node_link(graph_json: dict[str, Any]) -> nx.MultiDiGraph:
        # Support both historic "links" and newer "edges".
        if "links" in graph_json:
            try:
                graph = json_graph.node_link_graph(graph_json, edges="links")
            except TypeError:
                graph = json_graph.node_link_graph(graph_json)
        elif "edges" in graph_json and "links" not in graph_json:
            normalized = dict(graph_json)
            normalized["links"] = normalized.get("edges", [])
            graph = json_graph.node_link_graph(normalized)
        else:
            graph = json_graph.node_link_graph(graph_json)

        if isinstance(graph, nx.MultiDiGraph):
            return graph

        directed = nx.MultiDiGraph()
        directed.add_nodes_from(graph.nodes(data=True))
        for source, target, _key, data in graph.edges(keys=True, data=True):
            directed.add_edge(source, target, **data)
        return directed

    @staticmethod
    def _hierarchical_layout(graph: nx.MultiDiGraph) -> dict[Any, tuple[float, float]]:
        """Layer nodes by topological order for DAG visualization."""
        if not nx.is_directed_acyclic_graph(graph):
            return nx.spring_layout(graph, seed=42, k=1.4)

        layers: dict[Any, int] = {}
        for node in nx.topological_sort(graph):
            preds = list(graph.predecessors(node))
            layers[node] = 0 if not preds else max(layers[p] for p in preds) + 1

        by_layer: dict[int, list[Any]] = {}
        for node, layer in layers.items():
            by_layer.setdefault(layer, []).append(node)

        pos: dict[Any, tuple[float, float]] = {}
        max_layer = max(layers.values()) if layers else 0
        for layer, nodes in by_layer.items():
            y = 1.0 - (layer / max_layer) if max_layer else 0.5
            spacing = 1.0 / (len(nodes) + 1)
            for index, node in enumerate(sorted(nodes)):
                pos[node] = ((index + 1) * spacing - 0.5, y)
        return pos

    @classmethod
    def draw(
        cls,
        graph_json: dict[str, Any],
        figsize: tuple[int, int] = (16, 10),
        with_labels: bool = True,
        layout: str = "spring",
        title: str = "SQL2Graph Column Lineage",
    ):
        graph = cls.graph_from_node_link(graph_json)
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot visualize empty graph.")

        if layout in {"hierarchical", "dag"}:
            pos = cls._hierarchical_layout(graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(graph)
        elif layout == "shell":
            pos = nx.shell_layout(graph)
        elif nx.is_directed_acyclic_graph(graph):
            pos = cls._hierarchical_layout(graph)
        else:
            pos = nx.spring_layout(graph, seed=42, k=1.4)

        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize)

        node_colors = [
            cls.NODE_COLORS.get(graph.nodes[node].get("node_type", ""), "#CCCCCC")
            for node in graph.nodes()
        ]
        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=node_colors,
            node_size=1300,
            edgecolors="black",
            linewidths=0.8,
        )

        grouped_edges: dict[str, list[tuple[str, str, int]]] = {}
        for source, target, key, attrs in graph.edges(keys=True, data=True):
            edge_type = attrs.get("edge_type", "OTHER")
            grouped_edges.setdefault(edge_type, []).append((source, target, key))

        for edge_type, triples in grouped_edges.items():
            color = cls.EDGE_COLORS.get(edge_type, "#7f7f7f")
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=[(source, target) for source, target, _ in triples],
                edge_color=color,
                width=1.8,
                alpha=0.8,
                arrows=True,
                arrowsize=14,
                connectionstyle="arc3,rad=0.08",
            )

        if with_labels:
            labels = {}
            for node, attrs in graph.nodes(data=True):
                alias = attrs.get("alias")
                if alias:
                    labels[node] = alias
                else:
                    labels[node] = node if len(node) < 36 else f"{node[:33]}..."
            nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)

        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        return graph

    @classmethod
    def _node_display_label(cls, node_id: str, attrs: dict[str, Any]) -> str:
        label = attrs.get("label")
        if label:
            return str(label)
        alias = attrs.get("alias")
        if alias:
            return str(alias)
        if len(node_id) <= 28:
            return node_id
        return f"{node_id[:25]}..."

    @classmethod
    def _node_hover_title(cls, node_id: str, attrs: dict[str, Any]) -> str:
        parts = [
            f"<b>{html.escape(cls._node_display_label(node_id, attrs))}</b>",
            f"Type: {html.escape(cls.NODE_TYPE_LABELS.get(attrs.get('node_type', ''), attrs.get('node_type', 'unknown')))}",
            f"ID: {html.escape(node_id)}",
        ]
        for key in ("table_alias", "column", "expression", "table"):
            value = attrs.get(key)
            if value:
                parts.append(f"{key.replace('_', ' ').title()}: {html.escape(str(value))}")
        return "<br>".join(parts)

    @staticmethod
    def _format_detail_block(title: str, rows: list[tuple[str, str]]) -> str:
        if not rows:
            return ""
        body = "".join(
            f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
            for label, value in rows
            if value
        )
        if not body:
            return ""
        return f"<h4>{html.escape(title)}</h4><table class='detail-table'>{body}</table>"

    @classmethod
    def _node_detail_html(cls, graph: nx.MultiDiGraph, node_id: str) -> str:
        if node_id not in graph.nodes:
            return "<p>Node not found.</p>"
        attrs = dict(graph.nodes[node_id])
        node_type = attrs.get("node_type", "")
        rows = [
            ("Label", cls._node_display_label(node_id, attrs)),
            ("Type", cls.NODE_TYPE_LABELS.get(node_type, node_type or "unknown")),
            ("ID", node_id),
            ("Chunk type", str(attrs.get("chunk_type") or "")),
            ("SQL", str(attrs.get("sql") or "")[:1200]),
            ("Table alias", str(attrs.get("table_alias") or "")),
            ("Column", str(attrs.get("column") or "")),
            ("Table", str(attrs.get("table") or "")),
            ("Expression", str(attrs.get("expression") or "")),
        ]
        incoming = []
        for source, _, edge_attrs in graph.in_edges(node_id, data=True):
            incoming.append(
                f"{source} <span class='edge-tag'>{edge_attrs.get('edge_type', 'EDGE')}</span>"
            )
        outgoing = []
        for _, target, edge_attrs in graph.out_edges(node_id, data=True):
            outgoing.append(
                f"{target} <span class='edge-tag'>{edge_attrs.get('edge_type', 'EDGE')}</span>"
            )
        detail = cls._format_detail_block("Node", rows)
        if incoming:
            detail += "<h4>Incoming lineage</h4><ul>" + "".join(
                f"<li>{item}</li>" for item in incoming
            ) + "</ul>"
        if outgoing:
            detail += "<h4>Outgoing lineage</h4><ul>" + "".join(
                f"<li>{item}</li>" for item in outgoing
            ) + "</ul>"
        return detail or "<p>No details available.</p>"

    @classmethod
    def _edge_detail_html(cls, graph: nx.MultiDiGraph, edge_id: str) -> str:
        for source, target, key, attrs in graph.edges(keys=True, data=True):
            current_id = f"{source}->{target}:{key}"
            if current_id != edge_id:
                continue
            rows = [
                ("Type", str(attrs.get("edge_type") or "")),
                ("From", source),
                ("To", target),
            ]
            for key_name, value in attrs.items():
                if key_name == "edge_type" or value in (None, ""):
                    continue
                rows.append((key_name.replace("_", " ").title(), str(value)))
            return cls._format_detail_block("Edge", rows) or "<p>No edge details available.</p>"
        return "<p>Edge not found.</p>"

    @classmethod
    def to_interactive_html(
        cls,
        graph_json: dict[str, Any],
        height: str = "780px",
        title: str = "SQL2Graph Column Lineage",
    ) -> str:
        """Build a self-contained interactive HTML view (vis.js) with click details."""
        graph = cls.graph_from_node_link(graph_json)
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot visualize empty graph.")

        is_dag = nx.is_directed_acyclic_graph(graph)
        vis_nodes: list[dict[str, Any]] = []
        node_details: dict[str, str] = {}
        for node_id, attrs in graph.nodes(data=True):
            node_type = attrs.get("node_type", "")
            vis_nodes.append(
                {
                    "id": node_id,
                    "label": cls._node_display_label(node_id, attrs),
                    "title": cls._node_hover_title(node_id, attrs),
                    "group": node_type or "other",
                    "shape": cls.NODE_SHAPES.get(node_type, "dot"),
                    "color": {
                        "background": cls.NODE_COLORS.get(node_type, "#CCCCCC"),
                        "border": "#2f2f2f",
                        "highlight": {"background": "#fff3bf", "border": "#e67700"},
                    },
                    "font": {"size": 14, "face": "Inter, Arial, sans-serif"},
                    "margin": 10,
                }
            )
            node_details[node_id] = cls._node_detail_html(graph, node_id)

        vis_edges: list[dict[str, Any]] = []
        edge_details: dict[str, str] = {}
        for source, target, key, attrs in graph.edges(keys=True, data=True):
            edge_type = attrs.get("edge_type", "EDGE")
            edge_id = f"{source}->{target}:{key}"
            vis_edges.append(
                {
                    "id": edge_id,
                    "from": source,
                    "to": target,
                    "label": edge_type.replace("_", " "),
                    "title": html.escape(edge_type),
                    "arrows": "to",
                    "color": {"color": cls.EDGE_COLORS.get(edge_type, "#7f7f7f"), "highlight": "#111"},
                    "width": 2,
                    "smooth": {"type": "curvedCW", "roundness": 0.12},
                    "font": {"size": 11, "align": "middle", "strokeWidth": 0},
                }
            )
            edge_details[edge_id] = cls._edge_detail_html(graph, edge_id)

        node_legend = "".join(
            f"<span class='legend-item'><i style='background:{color}'></i>{html.escape(cls.NODE_TYPE_LABELS.get(node_type, node_type))}</span>"
            for node_type, color in cls.NODE_COLORS.items()
        )
        edge_legend = "".join(
            f"<span class='legend-item'><i style='background:{color}'></i>{html.escape(edge_type.replace('_', ' '))}</span>"
            for edge_type, color in cls.EDGE_COLORS.items()
        )

        physics_options = (
            {
                "enabled": True,
                "hierarchicalRepulsion": {
                    "nodeDistance": 140,
                    "centralGravity": 0.0,
                    "springLength": 120,
                    "springConstant": 0.01,
                },
                "solver": "hierarchicalRepulsion",
            }
            if is_dag
            else {
                "enabled": True,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -40,
                    "centralGravity": 0.01,
                    "springLength": 120,
                    "avoidOverlap": 1,
                },
                "stabilization": {"iterations": 150},
            }
        )

        layout_options = (
            {
                "hierarchical": {
                    "enabled": True,
                    "direction": "UD",
                    "sortMethod": "directed",
                    "levelSeparation": 170,
                    "nodeSpacing": 180,
                    "treeSpacing": 220,
                }
            }
            if is_dag
            else {}
        )

        payload = {
            "nodes": vis_nodes,
            "edges": vis_edges,
            "nodeDetails": node_details,
            "edgeDetails": edge_details,
            "options": {
                "layout": layout_options,
                "physics": physics_options,
                "interaction": {
                    "hover": True,
                    "multiselect": True,
                    "navigationButtons": True,
                    "keyboard": True,
                    "tooltipDelay": 120,
                },
                "nodes": {"borderWidth": 1.5, "shadow": True},
                "edges": {"shadow": False, "selectionWidth": 2},
            },
            "groups": {
                node_type: {
                    "color": {"background": color, "border": "#2f2f2f"},
                    "shape": cls.NODE_SHAPES.get(node_type, "dot"),
                }
                for node_type, color in cls.NODE_COLORS.items()
            },
        }

        stats = (
            f"{graph.number_of_nodes()} nodes · {graph.number_of_edges()} edges · "
            f"{'DAG' if is_dag else 'cyclic'}"
        )
        payload_json = json.dumps(payload, ensure_ascii=False)
        payload_json = payload_json.replace("</", "<\\/")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, Arial, sans-serif;
    }}
    body {{
      margin: 0;
      background: #f7f8fb;
      color: #1f2937;
    }}
    .toolbar {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      padding: 12px 16px;
      background: #ffffff;
      border-bottom: 1px solid #e5e7eb;
    }}
    .toolbar input, .toolbar select, .toolbar button {{
      font: inherit;
      padding: 8px 10px;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      background: #fff;
    }}
    .toolbar button {{
      cursor: pointer;
      background: #eef2ff;
    }}
    .stats {{
      margin-left: auto;
      color: #6b7280;
      font-size: 13px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 0;
      height: calc({height} - 56px);
      min-height: 520px;
    }}
    #network {{
      background: #ffffff;
      border-right: 1px solid #e5e7eb;
    }}
    #detail-panel {{
      background: #fcfcfd;
      padding: 16px;
      overflow: auto;
    }}
    #detail-panel h3 {{
      margin: 0 0 8px;
      font-size: 18px;
    }}
    #detail-content {{
      font-size: 14px;
      line-height: 1.45;
    }}
    .detail-table {{
      width: 100%;
      border-collapse: collapse;
      margin: 8px 0 14px;
      font-size: 13px;
    }}
    .detail-table th {{
      text-align: left;
      vertical-align: top;
      width: 38%;
      color: #6b7280;
      padding: 6px 8px 6px 0;
      font-weight: 600;
    }}
    .detail-table td {{
      padding: 6px 0;
      word-break: break-word;
    }}
    .edge-tag {{
      display: inline-block;
      margin-left: 6px;
      padding: 1px 6px;
      border-radius: 999px;
      background: #eef2ff;
      color: #3730a3;
      font-size: 11px;
    }}
    .legend {{
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      padding: 10px 16px 14px;
      background: #ffffff;
      border-top: 1px solid #e5e7eb;
      font-size: 12px;
    }}
    .legend-item i {{
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 3px;
      margin-right: 6px;
      vertical-align: -2px;
    }}
    .hint {{
      color: #6b7280;
      font-size: 13px;
      margin-bottom: 12px;
    }}
    ul {{
      margin: 0 0 12px 18px;
      padding: 0;
    }}
  </style>
</head>
<body>
  <div class="toolbar">
    <input id="search" type="search" placeholder="Search nodes..." aria-label="Search nodes" />
    <select id="type-filter" aria-label="Filter by node type">
      <option value="">All node types</option>
      <option value="source_column">Source columns</option>
      <option value="output_column">Output columns</option>
      <option value="filter">Filters</option>
      <option value="join">Joins</option>
    </select>
    <button id="fit-btn" type="button">Fit view</button>
    <button id="reset-btn" type="button">Reset selection</button>
    <span class="stats">{html.escape(stats)}</span>
  </div>
  <div class="layout">
    <div id="network"></div>
    <aside id="detail-panel">
      <h3>{html.escape(title)}</h3>
      <p class="hint">Click a node or edge to inspect lineage. Drag nodes, scroll to zoom, use arrow keys to pan.</p>
      <div id="detail-content">Select a node or edge to see details here.</div>
    </aside>
  </div>
  <div class="legend">
    <div>{node_legend}</div>
    <div>{edge_legend}</div>
  </div>
  <script>
    const payload = {payload_json};
    const nodes = new vis.DataSet(payload.nodes);
    const edges = new vis.DataSet(payload.edges);
    const container = document.getElementById("network");
    const detailContent = document.getElementById("detail-content");
    const network = new vis.Network(container, {{ nodes, edges }}, payload.options);
    network.setOptions({{ groups: payload.groups }});

    function setDetail(html) {{
      detailContent.innerHTML = html;
    }}

    function highlightNodes(matchingIds) {{
      const matchSet = new Set(matchingIds);
      const updates = payload.nodes.map((node) => {{
        if (matchSet.size === 0) {{
          return {{ id: node.id, hidden: false, opacity: 1 }};
        }}
        const matched = matchSet.has(node.id);
        return {{
          id: node.id,
          hidden: !matched,
          opacity: matched ? 1 : 0.15,
        }};
      }});
      nodes.update(updates);
      const edgeUpdates = payload.edges.map((edge) => {{
        if (matchSet.size === 0) {{
          return {{ id: edge.id, hidden: false }};
        }}
        const matched = matchSet.has(edge.from) || matchSet.has(edge.to);
        return {{ id: edge.id, hidden: !matched }};
      }});
      edges.update(edgeUpdates);
      if (matchingIds.length > 0) {{
        network.fit({{ nodes: matchingIds, animation: true }});
      }}
    }}

    network.on("click", (params) => {{
      if (params.nodes.length > 0) {{
        const nodeId = params.nodes[0];
        setDetail(payload.nodeDetails[nodeId] || "<p>No details available.</p>");
        return;
      }}
      if (params.edges.length > 0) {{
        const edgeId = params.edges[0];
        setDetail(payload.edgeDetails[edgeId] || "<p>No details available.</p>");
        return;
      }}
      setDetail("<p>Select a node or edge to see details here.</p>");
    }});

    network.on("doubleClick", (params) => {{
      if (params.nodes.length === 0) {{
        return;
      }}
      const nodeId = params.nodes[0];
      const connected = network.getConnectedNodes(nodeId);
      highlightNodes([nodeId, ...connected]);
      setDetail(payload.nodeDetails[nodeId] || "<p>No details available.</p>");
    }});

    document.getElementById("search").addEventListener("input", (event) => {{
      const query = event.target.value.trim().toLowerCase();
      if (!query) {{
        highlightNodes([]);
        return;
      }}
      const matches = payload.nodes
        .filter((node) => {{
          return String(node.id).toLowerCase().includes(query)
            || String(node.label).toLowerCase().includes(query)
            || String(node.group || "").toLowerCase().includes(query);
        }})
        .map((node) => node.id);
      highlightNodes(matches);
      if (matches.length === 1) {{
        setDetail(payload.nodeDetails[matches[0]] || "<p>No details available.</p>");
      }}
    }});

    document.getElementById("type-filter").addEventListener("change", (event) => {{
      const selected = event.target.value;
      if (!selected) {{
        highlightNodes([]);
        return;
      }}
      const matches = payload.nodes
        .filter((node) => node.group === selected)
        .map((node) => node.id);
      highlightNodes(matches);
    }});

    document.getElementById("fit-btn").addEventListener("click", () => network.fit({{ animation: true }}));
    document.getElementById("reset-btn").addEventListener("click", () => {{
      document.getElementById("search").value = "";
      document.getElementById("type-filter").value = "";
      highlightNodes([]);
      network.unselectAll();
      setDetail("<p>Select a node or edge to see details here.</p>");
    }});

    network.once("stabilizationIterationsDone", () => network.fit({{ animation: true }}));
    if (!payload.options.physics.enabled) {{
      network.fit({{ animation: false }});
    }}
  </script>
</body>
</html>"""

    @staticmethod
    def _parse_height(height: str) -> int:
        value = str(height).strip().lower().removesuffix("px")
        try:
            return max(400, int(float(value)))
        except ValueError:
            return 780

    @classmethod
    def _edge_lineage_type(cls, edge_data: dict[str, Any]) -> str:
        edge_type = str(edge_data.get("edge_type") or edge_data.get("link_type") or "").strip().upper()
        if edge_type == "UNIONALL":
            return "UNION ALL"
        return edge_type

    @classmethod
    def _iter_lineage_neighbors(cls, graph: nx.MultiDiGraph, node: str) -> list[str]:
        neighbors: list[str] = []
        for pred, _target, _key, edge_data in graph.in_edges(node, keys=True, data=True):
            if cls._edge_lineage_type(edge_data) in cls.LINEAGE_EDGE_TYPES:
                neighbors.append(pred)
        for _source, succ, _key, edge_data in graph.out_edges(node, keys=True, data=True):
            if cls._edge_lineage_type(edge_data) in cls.LINEAGE_EDGE_TYPES:
                neighbors.append(succ)
        return neighbors

    @classmethod
    def collect_lineage_nodes(cls, graph: nx.MultiDiGraph, start: str) -> set[str]:
        """Collect all nodes connected to ``start`` via lineage edge types."""
        if start not in graph:
            return set()

        visited: set[str] = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor in cls._iter_lineage_neighbors(graph, node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return visited

    @classmethod
    def _node_marker_color(cls, attrs: dict[str, Any]) -> str:
        if attrs.get("node_type") == "chunk":
            return cls.CHUNK_TYPE_COLORS.get(attrs.get("chunk_type", "query"), "#ADD8E6")
        return cls.NODE_COLORS.get(attrs.get("node_type", ""), "#CCCCCC")

    @classmethod
    def _build_plotly_figure(
        cls,
        graph: nx.MultiDiGraph,
        title: str,
    ):
        """Build a Plotly figure for notebook-native pan/zoom/click interaction."""
        import plotly.graph_objects as go

        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot visualize empty graph.")

        pos = cls._hierarchical_layout(graph)
        node_ids = list(graph.nodes())

        edge_traces: list[Any] = []
        seen_edge_types: set = set()
        for _source, _target, _key, attrs in graph.edges(keys=True, data=True):
            edge_type = cls._edge_lineage_type(attrs) or "OTHER"
            seen_edge_types.add(edge_type)

        for edge_type in sorted(seen_edge_types):
            edge_x: list[float | None] = []
            edge_y: list[float | None] = []
            for source, target, _key, attrs in graph.edges(keys=True, data=True):
                if (cls._edge_lineage_type(attrs) or "OTHER") != edge_type:
                    continue
                x0, y0 = pos[source]
                x1, y1 = pos[target]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            if not edge_x:
                continue
            edge_traces.append(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line=dict(width=2, color=cls.EDGE_COLORS.get(edge_type, "#7f7f7f")),
                    hoverinfo="skip",
                    name=edge_type.replace("_", " "),
                    legendgroup="edges",
                )
            )

        node_x = [pos[node_id][0] for node_id in node_ids]
        node_y = [pos[node_id][1] for node_id in node_ids]
        node_labels = [cls._node_display_label(node_id, graph.nodes[node_id]) for node_id in node_ids]
        node_hover = []
        for node_id in node_ids:
            attrs = graph.nodes[node_id]
            hover_lines = [
                cls._node_display_label(node_id, attrs),
                f"Type: {cls.NODE_TYPE_LABELS.get(attrs.get('node_type', ''), attrs.get('node_type', 'unknown'))}",
                f"ID: {node_id}",
            ]
            for key in ("table_alias", "column", "expression", "table"):
                value = attrs.get(key)
                if value:
                    hover_lines.append(f"{key.replace('_', ' ').title()}: {value}")
            node_hover.append("<br>".join(hover_lines))

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_labels,
            textposition="top center",
            textfont=dict(size=10),
            hovertext=node_hover,
            hoverinfo="text",
            marker=dict(
                size=[
                    26
                    if graph.nodes[node_id].get("node_type") == "chunk"
                    else (22 if graph.nodes[node_id].get("node_type") == "output_column" else 16)
                    for node_id in node_ids
                ],
                color=[cls._node_marker_color(graph.nodes[n]) for n in node_ids],
                line=dict(width=1.5, color="#333333"),
            ),
            name="Nodes",
            showlegend=False,
        )

        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title=title,
            hovermode="closest",
            dragmode="pan",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
            plot_bgcolor="#ffffff",
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        return fig, node_ids

    @classmethod
    def _display_plotly_interactive(
        cls,
        graph_json: dict[str, Any],
        title: str,
        height: str = "780px",
    ) -> nx.MultiDiGraph:
        """Pan/zoom/click graph using Plotly FigureWidget (works reliably in Jupyter)."""
        try:
            import ipywidgets as widgets
            import plotly.graph_objects as go
            from IPython.display import display
        except ImportError as exc:
            raise RuntimeError(
                "Interactive Plotly view requires plotly and ipywidgets. "
                "Install with: uv pip install plotly ipywidgets"
            ) from exc

        graph = cls.graph_from_node_link(graph_json)
        fig, node_ids = cls._build_plotly_figure(graph, title)
        try:
            fig_widget = go.FigureWidget(fig)
        except ImportError as exc:
            raise RuntimeError(
                "Plotly FigureWidget requires anywidget. "
                "Install with: uv sync  (or: uv pip install anywidget), then restart the kernel."
            ) from exc
        fig_widget.update_layout(height=cls._parse_height(height))

        detail = widgets.HTML(
            value=(
                "<p><b>Click a node</b> to highlight its full upstream/downstream lineage. "
                "Drag to pan, scroll to zoom, hover for quick info.</p>"
            ),
            layout=widgets.Layout(
                width="100%",
                min_height="120px",
                border="1px solid #e5e7eb",
                padding="12px",
                overflow="auto",
            ),
        )
        node_trace_idx = len(fig_widget.data) - 1

        default_colors = [cls._node_marker_color(graph.nodes[node_id]) for node_id in node_ids]
        default_sizes = [
            26
            if graph.nodes[node_id].get("node_type") == "chunk"
            else (22 if graph.nodes[node_id].get("node_type") == "output_column" else 16)
            for node_id in node_ids
        ]
        selected: dict[str, int | None] = {"index": None}

        def apply_highlight(selected_idx: int | None) -> None:
            colors = list(default_colors)
            sizes = list(default_sizes)
            if selected_idx is not None:
                node_id = node_ids[selected_idx]
                lineage_nodes = cls.collect_lineage_nodes(graph, node_id)
                for index, current_id in enumerate(node_ids):
                    if current_id not in lineage_nodes:
                        colors[index] = cls.HIGHLIGHT_DIMMED_COLOR
                        sizes[index] = max(10, default_sizes[index] - 4)
                        continue
                    if current_id == node_id:
                        colors[index] = cls.HIGHLIGHT_SELECTED_COLOR
                        sizes[index] = 30
                    else:
                        colors[index] = cls.HIGHLIGHT_LINEAGE_COLOR
                        sizes[index] = max(default_sizes[index], 20)
            with fig_widget.batch_update():
                fig_widget.data[node_trace_idx].marker.color = tuple(colors)
                fig_widget.data[node_trace_idx].marker.size = tuple(sizes)

        def on_click(trace, points, _selector) -> None:
            if not points.point_inds:
                return
            selected_idx = points.point_inds[0]
            selected["index"] = selected_idx
            apply_highlight(selected_idx)
            node_id = node_ids[selected_idx]
            lineage_nodes = cls.collect_lineage_nodes(graph, node_id)
            detail.value = (
                cls._node_detail_html(graph, node_id)
                + f"<p><i>Highlighted lineage: {len(lineage_nodes)} node(s)</i></p>"
            )

        fig_widget.data[node_trace_idx].on_click(on_click)

        reset_btn = widgets.Button(description="Clear selection", layout=widgets.Layout(width="140px"))
        reset_btn.on_click(
            lambda _btn: (
                selected.update(index=None),
                apply_highlight(None),
                detail.__setattr__(
                    "value",
                    "<p><b>Click a node</b> to highlight its full lineage. "
                    "Drag to pan, scroll to zoom.</p>",
                ),
            )
        )
        display(widgets.VBox([fig_widget, widgets.HBox([reset_btn]), detail]))
        return graph

    @staticmethod
    def _display_interactive_html(html_doc: str, width: str = "100%", height: str = "780px") -> None:
        """Embed interactive HTML in Jupyter (IPython IFrame has no srcdoc support)."""
        try:
            from IPython.display import HTML, display
        except ImportError as exc:
            raise RuntimeError(
                "Interactive visualization requires IPython (Jupyter). "
                "Use to_interactive_html() and open the HTML in a browser."
            ) from exc

        srcdoc = html.escape(html_doc, quote=True)
        display(
            HTML(
                f'<iframe width="{width}" height="{height}" '
                f'srcdoc="{srcdoc}" '
                f'sandbox="allow-scripts allow-same-origin" '
                f'frameborder="0" style="border:0;width:100%;"></iframe>'
            )
        )

    @classmethod
    def show_interactive(
        cls,
        graph_json: dict[str, Any],
        height: str = "780px",
        title: str = "SQL2Graph Column Lineage",
        backend: str = "plotly",
    ) -> nx.MultiDiGraph:
        """Display an interactive graph in Jupyter (click nodes for details)."""
        if backend == "html":
            graph = cls.graph_from_node_link(graph_json)
            html_doc = cls.to_interactive_html(graph_json, height=height, title=title)
            cls._display_interactive_html(html_doc, height=height)
            return graph
        try:
            return cls._display_plotly_interactive(graph_json, title=title, height=height)
        except (ImportError, RuntimeError) as exc:
            import warnings

            warnings.warn(
                f"Plotly widget backend unavailable ({exc}); falling back to HTML viewer.",
                stacklevel=2,
            )
            graph = cls.graph_from_node_link(graph_json)
            html_doc = cls.to_interactive_html(graph_json, height=height, title=title)
            cls._display_interactive_html(html_doc, height=height)
            return graph

    @classmethod
    def explore(
        cls,
        result: dict[str, Any],
        height: str = "780px",
        backend: str = "plotly",
    ) -> None:
        """Notebook explorer: switch between full graph and subgraphs interactively."""
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError as exc:
            raise RuntimeError("explore() requires ipywidgets (installed with Jupyter).") from exc

        if "error" in result:
            raise ValueError(result.get("error", "Pipeline result contains an error."))

        graph_options: list[tuple[str, dict[str, Any]]] = [("Full graph", result["graph"])]
        for index, subgraph in enumerate(result.get("subgraphs", [])):
            label = f"[{index}] {subgraph.get('type')} / {subgraph.get('name')}"
            graph_options.append((label, subgraph.get("graph") or {"nodes": [], "links": []}))

        graph_output = widgets.Output(
            layout=widgets.Layout(width="100%", overflow="visible"),
        )
        dropdown = widgets.Dropdown(
            options=[(label, idx) for idx, (label, _) in enumerate(graph_options)],
            value=0,
            description="View:",
            layout=widgets.Layout(width="70%"),
        )
        summary = widgets.HTML(
            value=(
                "<p><b>Interactive lineage explorer</b> — pick a graph, click nodes for details, "
                "drag to pan, scroll to zoom.</p>"
            )
        )

        def refresh(_=None) -> None:
            label, graph_json = graph_options[dropdown.value]
            with graph_output:
                graph_output.clear_output(wait=True)
                if not graph_json.get("nodes"):
                    display(widgets.HTML(f"<p><b>{html.escape(label)}</b> — no mapped nodes in this subgraph yet.</p>"))
                    return
                if backend == "html":
                    cls._display_interactive_html(
                        cls.to_interactive_html(graph_json, height=height, title=label),
                        height=height,
                    )
                else:
                    cls._display_plotly_interactive(graph_json, title=label, height=height)

        dropdown.observe(refresh, names="value")
        refresh()
        display(widgets.VBox([summary, dropdown, graph_output]))
