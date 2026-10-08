import torch
import networkx as nx
import matplotlib.pyplot as plt
import io
from dataclasses import dataclass

import math


@dataclass(frozen=True)
class RenderInfo:
    graph: nx.Graph
    x_errors: torch.Tensor
    z_errors: torch.Tensor
    data: object  # Assuming data is a PyTorch Geometric Data object
    node_to_index: dict
    pos: dict = None
    figsize: tuple = (10, 8)
    edge_groups: list = None
    node_groups: list = None
    labels: dict = None
    legend: bool = True
    title: str = None
    save_path: str = None


def _get_subgraph_of_indices(graph, indices):
    # Return the subgraph of the Tanner graph containing only the specified qubit indices and their neighboring checks.
    nodes_to_include = set()
    for idx in indices:
        qubit_node = f"q{idx}"
        nodes_to_include.add(qubit_node)
        neighbors = graph.neighbors(qubit_node)
        nodes_to_include.update(neighbors)

    # Filter on only q and z check nodes to simplify visualization
    nodes_to_include = {n for n in nodes_to_include if graph.nodes[n]["node_type"] in ("qubit", "z_check")}

    return graph.subgraph(nodes_to_include)


def _get_render_groups(graph, x_errors, z_errors, data, node_to_index):
    # Separate node lists
    qubits = [n for n in graph.nodes if graph.nodes[n]["node_type"] == "qubit"] #and x_errors[int(n[1:])] == 1]
    x_checks = [n for n in graph.nodes if graph.nodes[n]["node_type"] == "x_check"] #and data.x[node_to_index[n], 3] == 1]
    z_checks = [n for n in graph.nodes if graph.nodes[n]["node_type"] == "z_check"] #and data.x[node_to_index[n], 4] == 1]
    qubit_colors = ["orange" if x_errors[int(n[1:])] == 1 else "black" for n in qubits]
    x_check_colors = ["red" if data.x[node_to_index[n], 3] == 1 else "lightcoral" for n in x_checks]
    z_check_colors = ["blue" if data.x[node_to_index[n], 4] == 1 else "lightblue" for n in z_checks]

    # Color edges based on whether they are connected to an error qubit or not
    error_edges = []
    normal_edges = []

    def is_error_qubit(node):
        return graph.nodes[node]["node_type"] == "qubit" and x_errors[int(node[1:])] == 1

    for u, v in graph.edges:
        # Check if either endpoint is a qubit with error
        if is_error_qubit(u) or is_error_qubit(v):
            error_edges.append((u, v))
        else:
            normal_edges.append((u, v))

    node_groups = [
        {"nodelist": qubits, "node_color": qubit_colors, "node_shape": "o", "node_size": 200, "label": "Data qubits"},
        {"nodelist": x_checks, "node_color": x_check_colors, "node_shape": "s", "node_size": 300, "label": "X checks"},
        {"nodelist": z_checks, "node_color": z_check_colors, "node_shape": "^", "node_size": 300, "label": "Z checks"}
    ]

    edge_groups = [
        {"edgelist": normal_edges, "alpha": 0.3},
        {"edgelist": error_edges, "edge_color": "red", "width": 2.0}
    ]

    return node_groups, edge_groups


def _draw_graph(render_info: RenderInfo):
    fig = plt.figure(figsize=render_info.figsize)

    if render_info.edge_groups:
        for kwargs in render_info.edge_groups:
            nx.draw_networkx_edges(render_info.graph, render_info.pos, **kwargs)
    else:
        nx.draw_networkx_edges(render_info.graph, render_info.pos, alpha=0.3)

    if render_info.node_groups:
        for kwargs in render_info.node_groups:
            nx.draw_networkx_nodes(render_info.graph, render_info.pos, **kwargs)

    if render_info.labels:
        nx.draw_networkx_labels(render_info.graph, render_info.pos, labels=render_info.labels, font_size=8)

    if render_info.legend:
        plt.legend(scatterpoints=1)

    if render_info.title:
        plt.title(render_info.title)

    plt.axis('off')

    if not render_info.save_path:
        plt.show()
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)

    plt.close(fig)


def render_subgraph(graph, data, node_to_index, indices=None, x_errors=None, z_errors=None, with_labels=False, overlap=None):

    subgraph = _get_subgraph_of_indices(graph, torch.where(x_errors == 1)[0].tolist()) if indices is None else _get_subgraph_of_indices(graph, indices)

    qubits = [n for n in subgraph.nodes if subgraph.nodes[n]["node_type"] == "qubit"]
    x_checks = [n for n in subgraph.nodes if subgraph.nodes[n]["node_type"] == "x_check" and data.x[node_to_index[n], 3] == 1]
    x_checks_syndrome = [n for n in subgraph.nodes if subgraph.nodes[n]["node_type"] == "x_check" and data.x[node_to_index[n], 3] == 0]
    z_checks = [n for n in subgraph.nodes if subgraph.nodes[n]["node_type"] == "z_check" and data.x[node_to_index[n], 4] == 1]
    z_checks_syndrome = [n for n in subgraph.nodes if subgraph.nodes[n]["node_type"] == "z_check" and data.x[node_to_index[n], 4] == 0]

    # Example key-value pair: "q17" -> "17"
    qubit_labels = {n: str(int(n[1:])) for n in subgraph.nodes if subgraph.nodes[n]["node_type"] == "qubit"} if with_labels else None

    node_groups = [
        {"nodelist": qubits, "node_color": "orange", "node_shape": "o", "node_size": 200, "label": "Data qubits"},
        {"nodelist": z_checks, "node_color": "red", "node_shape": "s", "node_size": 300, "label": "Z checks"},
        {"nodelist": z_checks_syndrome, "node_color": "lightcoral", "node_shape": "s", "node_size": 300, "label": "Z checks (no syndrome)"}
    ]

    render_info = RenderInfo(
        graph=subgraph,
        x_errors=x_errors,
        z_errors=z_errors,
        data=data,
        node_to_index=node_to_index,
        pos=nx.spring_layout(subgraph, seed=42),
        figsize=(8, 6),
        edge_groups=None,
        node_groups=node_groups,
        labels=qubit_labels,
        legend=True,
        title=f"Pattern {overlap}" if overlap else None,
        save_path=None
    )

    _draw_graph(render_info)


def render_ldpc(graph, x_errors, z_errors, data, node_to_index):

    node_groups, edge_groups = _get_render_groups(graph, x_errors, z_errors, data, node_to_index)

    render_info = RenderInfo(
        graph=graph,
        x_errors=x_errors,
        z_errors=z_errors,
        data=data,
        node_to_index=node_to_index,
        pos=nx.spring_layout(graph, seed=42),
        figsize=(10, 8),
        edge_groups=edge_groups,
        node_groups=node_groups,
        labels=None,
        legend=True,
        title="LDPC Code Tanner Graph",
        save_path=None
    )

    _draw_graph(render_info)


def render_surface(graph, x_errors, z_errors, data, node_to_index, distance):

    # First, obtain the positions of the qubits in a d x d grid
    qubits = [n for n, t in graph.nodes(data="node_type") if t == "qubit"]

    pos = {
        q: (i % distance, -(i // distance))
        for q in qubits
        for i in [int(q[1:])]
    }

    for check, t in graph.nodes(data="node_type"):
        if t not in {"x_check", "z_check"}:
            continue

        ns = list(graph[check])
        pos[check] = tuple(sum(pos[q][axis] for q in ns) / len(ns) for axis in (0, 1))

    node_groups, edge_groups = _get_render_groups(graph, x_errors, z_errors, data, node_to_index)

    render_info = RenderInfo(
        graph=graph,
        x_errors=x_errors,
        z_errors=z_errors,
        data=data,
        node_to_index=node_to_index,
        pos=pos,
        figsize=(10, 8),
        edge_groups=edge_groups,
        node_groups=node_groups,
        labels=None,
        legend=True,
        title="LDPC Code Tanner Graph",
        save_path=None
    )

    _draw_graph(render_info)