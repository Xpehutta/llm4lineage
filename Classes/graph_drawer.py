import networkx as nx
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams


class DataLineageDAG:
    def __init__(self):
        self.G = nx.DiGraph()
        self.original_node_colors = {}
        self.pure_sources = set()
        self.pure_targets = set()
        self.intermediates = set()

    def build_dag(self, lineage_data):
        """Build DAG from lineage data."""
        if isinstance(lineage_data, dict):
            lineage_data = [lineage_data]

        all_sources = set()
        all_targets = set()

        for entry in lineage_data:
            target = entry['target']
            sources = entry['sources']
            all_targets.add(target)
            all_sources.update(sources)

            for source in sources:
                self.G.add_edge(source, target)

        self.pure_sources = all_sources - all_targets
        self.pure_targets = all_targets - all_sources
        self.intermediates = all_sources & all_targets

        # Assign colors
        for node in self.G.nodes():
            if node in self.pure_sources:
                self.original_node_colors[node] = '#90EE90'  # lightgreen
            elif node in self.pure_targets:
                self.original_node_colors[node] = '#F08080'  # lightcoral
            else:
                self.original_node_colors[node] = '#ADD8E6'  # lightblue

        return self

    def visualize(self, title="Data Lineage DAG", exclude_intermediates=False,
                  target=None, depth=None, figsize=(14, 10),
                  save_path=None, dpi=300, layout='hierarchical'):
        """
        Visualize lineage with clear directional edges.

        Parameters:
        - title: Plot title
        - exclude_intermediates: Collapse intermediate nodes
        - target: Focus on this target table
        - depth: Max depth for upstream traversal
        - figsize: Figure size (width, height)
        - save_path: Path to save the figure
        - dpi: Resolution for saved figure
        - layout: 'hierarchical' (top-to-bottom) or 'spring' (force-directed)
        """
        # Create a copy of the graph for visualization
        G = self.G.copy()
        node_colors = dict(self.original_node_colors)

        # Focus on target subgraph
        if target is not None:
            if target not in G:
                raise ValueError(f"Target table '{target}' not found in DAG")

            # Get upstream nodes
            upstream_nodes = self._get_upstream_nodes(target, depth)
            G = G.subgraph(upstream_nodes).copy()
            # Highlight focus target
            node_colors[target] = '#FFD700'  # gold

        # Collapse intermediates if requested
        if exclude_intermediates:
            self._collapse_intermediates(G, node_colors, target)

        # Create matplotlib figure
        plt.figure(figsize=figsize)

        # Generate title
        plot_title = self._generate_title(title, target, depth, exclude_intermediates)

        # Get positions based on layout choice
        if layout == 'hierarchical':
            pos = self._get_hierarchical_layout(G, target)
        else:
            pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

        # Draw edges FIRST with clear arrows
        self._draw_edges(G, pos)

        # Draw nodes SECOND (on top of edges)
        self._draw_nodes(G, pos, node_colors)

        # Draw labels
        self._draw_labels(G, pos)

        # Set title
        plt.title(plot_title, fontsize=18, pad=25, fontweight='bold')

        # Add legend
        self._add_legend(G, target, exclude_intermediates)

        # Remove axes and add grid for better readability
        plt.axis('off')
        plt.grid(False)

        # Adjust layout
        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"Figure saved to: {save_path}")

        plt.show()

        return plt.gcf()

    def _get_upstream_nodes(self, target, max_depth=None):
        """Get all upstream nodes for a target."""
        visited = set()
        queue = deque([(target, 0)])

        while queue:
            node, current_depth = queue.popleft()
            if node not in visited:
                visited.add(node)
                # Traverse upstream if depth allows
                if max_depth is None or current_depth < max_depth:
                    for predecessor in self.G.predecessors(node):
                        queue.append((predecessor, current_depth + 1))

        return visited

    def _collapse_intermediates(self, G, node_colors, target):
        """Collapse intermediate nodes by connecting predecessors to successors."""
        # Identify intermediates in current graph
        intermediates_in_graph = [n for n in G.nodes()
                                  if n in self.intermediates
                                  and n != target]  # Don't collapse focus target

        for node in intermediates_in_graph:
            predecessors = list(G.predecessors(node))
            successors = list(G.successors(node))

            # Connect predecessors to successors
            for pred in predecessors:
                for succ in successors:
                    # Avoid self-loops and duplicate edges
                    if pred != succ and not G.has_edge(pred, succ):
                        G.add_edge(pred, succ)

            # Remove intermediate node
            G.remove_node(node)
            if node in node_colors:
                del node_colors[node]

    def _get_hierarchical_layout(self, G, target=None):
        """Create hierarchical (top-to-bottom) layout for DAG."""
        if not G.nodes():
            return {}

        # Use multipartite layout for hierarchical arrangement
        # Assign layers based on longest path from sources
        if target:
            # For target-focused view, use layers based on distance from target
            layers = {}
            for node in G.nodes():
                try:
                    # Get shortest path length from node to target
                    path_length = nx.shortest_path_length(G, node, target)
                    layers[node] = path_length
                except nx.NetworkXNoPath:
                    # If no path to target (shouldn't happen in subgraph), use default
                    layers[node] = 0
        else:
            # For full graph, use layers based on longest path from sources
            layers = {}
            for node in G.nodes():
                if G.in_degree(node) == 0:  # Source node
                    layers[node] = 0
                else:
                    # Find longest path from any source to this node
                    max_len = 0
                    for source in [n for n in G.nodes() if G.in_degree(n) == 0]:
                        try:
                            path_length = nx.shortest_path_length(G, source, node)
                            max_len = max(max_len, path_length)
                        except nx.NetworkXNoPath:
                            continue
                    layers[node] = max_len

        # Create multipartite layout
        pos = {}
        layer_nodes = {}

        # Group nodes by layer
        for node, layer in layers.items():
            if layer not in layer_nodes:
                layer_nodes[layer] = []
            layer_nodes[layer].append(node)

        # Position nodes
        max_layer = max(layers.values()) if layers else 0
        for layer, nodes in layer_nodes.items():
            # Reverse the layer order so sources are at top, targets at bottom
            y = 1 - (layer / max_layer) if max_layer > 0 else 0.5
            x_positions = []
            spacing = 1.0 / (len(nodes) + 1)
            for i, node in enumerate(sorted(nodes)):
                x = (i + 1) * spacing - 0.5
                x_positions.append(x)
                pos[node] = (x, y)

        return pos

    def _draw_edges(self, G, pos):
        """Draw directed edges with clear arrows."""
        # Draw edges with arrows
        nx.draw_networkx_edges(
            G, pos,
            arrowstyle='-|>',  # Clear arrow style
            arrowsize=25,  # Larger arrow heads
            edge_color='#666666',  # Dark gray edges
            width=2.5,  # Thicker edges
            alpha=0.8,
            connectionstyle='arc3,rad=0.1',  # Slightly curved edges
            min_source_margin=15,  # Space between node and arrow start
            min_target_margin=15  # Space between node and arrow end
        )

    def _draw_nodes(self, G, pos, node_colors):
        """Draw nodes with appropriate styling."""
        for node in G.nodes():
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=[node],
                node_color=[node_colors.get(node, '#ADD8E6')],
                node_shape='s',  # square
                node_size=3500,
                edgecolors='black',
                linewidths=2,
                alpha=0.9
            )

    def _draw_labels(self, G, pos):
        """Draw node labels with escaped special characters."""
        labels = {}
        for node in G.nodes():
            # Escape special characters that mathtext might interpret
            escaped_label = node
            # Option 1: Replace $ with escaped version
            escaped_label = escaped_label.replace('$', r'\$')
            # Option 2: Or wrap entire label in verbatim mode
            # escaped_label = r'\verb|' + node + '|'
            labels[node] = escaped_label

        nx.draw_networkx_labels(
            G, pos,
            labels=labels,  # Use the escaped labels
            font_size=11,
            font_weight='bold',
            font_family='sans-serif',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                edgecolor='none',
                alpha=0.7
            )
        )

    def _add_legend(self, G, target, exclude_intermediates):
        """Add legend to the plot."""
        legend_entries = self._generate_legend(G, target, exclude_intermediates)
        legend_patches = []

        for label, color in legend_entries:
            patch = mpatches.Patch(color=color, label=label, alpha=0.9)
            legend_patches.append(patch)

        if legend_patches:
            # Add a box around the legend for better visibility
            plt.legend(
                handles=legend_patches,
                loc='upper left',
                bbox_to_anchor=(1.02, 1),
                fontsize=12,
                frameon=True,
                fancybox=True,
                framealpha=0.9,
                edgecolor='gray',
                title='Legend',
                title_fontsize='13'
            )

    def _generate_title(self, base_title, target, depth, exclude_intermediates):
        """Generate descriptive title."""
        title = base_title

        if target:
            title = f"Data Lineage: {target}"
            if depth is not None:
                title += f" (Depth={depth})"

        if exclude_intermediates:
            title += " - Intermediates Collapsed"

        return title

    def _generate_legend(self, graph, target, exclude_intermediates):
        """Create legend entries based on visible nodes."""
        legend = []

        # Color definitions
        colors = {
            'source': '#90EE90',  # lightgreen
            'target': '#F08080',  # lightcoral
            'intermediate': '#ADD8E6',  # lightblue
            'focus': '#FFD700'  # gold
        }

        # Add focus target entry if applicable
        if target and target in graph.nodes:
            legend.append(('Focus Target', colors['focus']))

        # Check what types of nodes exist in the graph
        has_sources = any(n in self.pure_sources and n != target for n in graph.nodes)
        has_targets = any(n in self.pure_targets and n != target for n in graph.nodes)
        has_intermediates = any(n in self.intermediates and n != target for n in graph.nodes)

        if has_sources:
            legend.append(('Source Table', colors['source']))
        if has_targets:
            legend.append(('Target Table', colors['target']))
        if has_intermediates and not exclude_intermediates:
            legend.append(('Intermediate Table', colors['intermediate']))

        return legend

    def print_lineage_summary(self):
        """Print a summary of the lineage DAG."""
        print("=" * 60)
        print("DATA LINEAGE SUMMARY")
        print("=" * 60)

        print(f"\nTotal Tables: {self.G.number_of_nodes()}")
        print(f"Total Relationships: {self.G.number_of_edges()}")
        print(f"Sources (no incoming edges): {len(self.pure_sources)}")
        print(f"Targets (no outgoing edges): {len(self.pure_targets)}")
        print(f"Intermediate Tables: {len(self.intermediates)}")

        if self.pure_sources:
            print(f"\nSource Tables:")
            for source in sorted(self.pure_sources):
                print(f"  • {source}")

        if self.pure_targets:
            print(f"\nTarget Tables:")
            for target in sorted(self.pure_targets):
                print(f"  • {target}")

        # Check for cycles (shouldn't exist in proper lineage)
        try:
            cycles = list(nx.find_cycle(self.G, orientation='original'))
            if cycles:
                print("\n⚠️  WARNING: Cycles detected in lineage!")
                for cycle in cycles[:3]:  # Show first 3 cycles
                    print(f"  Cycle: {cycle}")
        except nx.NetworkXNoCycle:
            print("\n✓ Lineage is acyclic (valid DAG)")

    def get_lineage_paths(self, source, target, max_paths=5):
        """Get all paths from source to target."""
        try:
            paths = list(nx.all_simple_paths(self.G, source, target, cutoff=10))
            return paths[:max_paths]
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound as e:
            print(f"Error: {e}")
            return []

    def export_to_dot(self, filename="lineage.dot"):
        """Export the DAG to Graphviz DOT format."""
        dot_string = "digraph DataLineage {\n"
        dot_string += "  rankdir=TB;\n"
        dot_string += "  node [shape=box, style=filled];\n\n"

        # Add nodes with colors
        for node in self.G.nodes():
            color = self.original_node_colors.get(node, 'lightblue')
            dot_string += f'  "{node}" [fillcolor="{color}"];\n'

        dot_string += "\n"

        # Add edges
        for source, target in self.G.edges():
            dot_string += f'  "{source}" -> "{target}";\n'

        dot_string += "}\n"

        with open(filename, 'w') as f:
            f.write(dot_string)

        print(f"DOT file exported to: {filename}")
        return dot_string