import torch
import networkx as nx
import matplotlib.pyplot as plt
import io
from dataclasses import dataclass
from PIL import Image


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
    return_fig: bool = False


def _get_subgraph_of_indices(code, indices):
    # Return the subgraph of the Tanner graph containing only the specified qubit indices and their neighboring checks.
    if indices is None:
        indices = torch.where(code.x_errors == 1)[0].tolist()

    nodes_to_include = set()
    for idx in indices:
        qubit_node = f"q{idx}"
        nodes_to_include.add(qubit_node)
        neighbors = code.graph.neighbors(qubit_node)
        nodes_to_include.update(neighbors)

    # Filter on only q and z check nodes to simplify visualization
    nodes_to_include = {n for n in nodes_to_include if code.graph.nodes[n]["node_type"] in ("qubit", "z_check")}

    return code.graph.subgraph(nodes_to_include)


def _is_node_type(graph, node, node_type):
    return graph.nodes[node]["node_type"] == node_type


def _get_render_groups(code):
    # Separate node lists
    qubits = [n for n in code.graph.nodes if _is_node_type(code.graph, n, "qubit")] #and x_errors[int(n[1:])] == 1]
    x_checks = [n for n in code.graph.nodes if _is_node_type(code.graph, n, "x_check")] #and data.x[node_to_index[n], 3] == 1]
    z_checks = [n for n in code.graph.nodes if _is_node_type(code.graph, n, "z_check")] #and data.x[node_to_index[n], 4] == 1]

    qubit_colors = [
        "purple" if code.x_errors[int(n[1:])] and code.z_errors[int(n[1:])]
        else "blue" if code.x_errors[int(n[1:])]
        else "orange" if code.z_errors[int(n[1:])]
        else "black"
        for n in qubits
    ]

    x_check_colors = ["red" if code.data.x[code.node_to_index[n], 3] == 1 else "lightcoral" for n in x_checks]
    z_check_colors = ["blue" if code.data.x[code.node_to_index[n], 4] == 1 else "lightblue" for n in z_checks]

    def _get_edge_error(q1, q2):
        qubit, check = (q1, q2) if _is_node_type(code.graph, q1, "qubit") else (q2, q1)
        # Sanity check to ensure homogeneity of the graph.
        assert _is_node_type(code.graph, qubit, "qubit") and not _is_node_type(code.graph, check, "qubit")

        qubit_idx = int(qubit[1:])
        return (
            1 if code.x_errors[qubit_idx] and _is_node_type(code.graph, check, "z_check")
            else 2 if code.z_errors[qubit_idx] and _is_node_type(code.graph, check, "x_check")
            else 0
        )

    edge_types = {0: [], 1: [], 2: []}
    for u, v in code.graph.edges:
        edge_types[_get_edge_error(u, v)].append((u, v))

    node_groups = [
        {"nodelist": qubits, "node_color": qubit_colors, "node_shape": "o", "node_size": 200, "label": "Data qubits"},
        {"nodelist": x_checks, "node_color": x_check_colors, "node_shape": "s", "node_size": 300, "label": "X checks"},
        {"nodelist": z_checks, "node_color": z_check_colors, "node_shape": "^", "node_size": 300, "label": "Z checks"}
    ]

    edge_groups = [
        {"edgelist": edge_types[0], "alpha": 0.3}, # No errors
        {"edgelist": edge_types[1], "edge_color": "blue", "width": 2.0}, # X error
        {"edgelist": edge_types[2], "edge_color": "red", "width": 2.0}, # Z error
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


    if render_info.return_fig:
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        plt.close(fig)
        return img

    plt.show()
    plt.close(fig)
    return None


def render_subgraph(code, indices=None, title=None, return_fig=False):

    subgraph = _get_subgraph_of_indices(code, indices)

    qubits = [n for n in subgraph.nodes if subgraph.nodes[n]["node_type"] == "qubit"]
    x_checks = [n for n in subgraph.nodes if subgraph.nodes[n]["node_type"] == "x_check" and code.data.x[code.node_to_index[n], 3] == 1]
    x_checks_syndrome = [n for n in subgraph.nodes if subgraph.nodes[n]["node_type"] == "x_check" and code.data.x[code.node_to_index[n], 3] == 0]
    z_checks = [n for n in subgraph.nodes if subgraph.nodes[n]["node_type"] == "z_check" and code.data.x[code.node_to_index[n], 4] == 1]
    z_checks_syndrome = [n for n in subgraph.nodes if subgraph.nodes[n]["node_type"] == "z_check" and code.data.x[code.node_to_index[n], 4] == 0]

    # Example key-value pair: "q17" -> "17"
    # qubit_labels = {n: str(int(n[1:])) for n in subgraph.nodes if subgraph.nodes[n]["node_type"] == "qubit"} if with_labels else None

    node_groups = [
        {"nodelist": qubits, "node_color": "orange", "node_shape": "o", "node_size": 200, "label": "Data qubits"},
        {"nodelist": z_checks, "node_color": "red", "node_shape": "s", "node_size": 300, "label": "Z checks"},
        {"nodelist": z_checks_syndrome, "node_color": "lightcoral", "node_shape": "s", "node_size": 300, "label": "Z checks (no syndrome)"},
        {"nodelist": x_checks, "node_color": "blue", "node_shape": "^", "node_size": 300, "label": "X checks"},
        {"nodelist": x_checks_syndrome, "node_color": "lightblue", "node_shape": "^", "node_size": 300, "label": "X checks (no syndrome)"}
    ]

    render_info = RenderInfo(
        graph=subgraph,
        x_errors=code.x_errors,
        z_errors=code.z_errors,
        data=code.data,
        node_to_index=code.node_to_index,
        pos=nx.spring_layout(subgraph, seed=42),
        figsize=(8, 6),
        edge_groups=None,
        node_groups=node_groups,
        labels=None,
        legend=False,
        title=title,
        save_path=None,
        return_fig=return_fig,
    )

    return _draw_graph(render_info)


def render_spring(code, title=None, return_fig=False):

    node_groups, edge_groups = _get_render_groups(code)

    render_info = RenderInfo(
        graph=code.graph,
        x_errors=code.x_errors,
        z_errors=code.z_errors,
        data=code.data,
        node_to_index=code.node_to_index,
        pos=nx.spring_layout(code.graph, seed=42),
        figsize=(10, 8),
        edge_groups=edge_groups,
        node_groups=node_groups,
        labels=None,
        legend=True,
        title=title,
        save_path=None,
        return_fig=return_fig,
    )

    return _draw_graph(render_info)


def render_surface(code, title=None, return_fig=False):

    # First, obtain the positions of the qubits in a d x d grid
    qubits = [n for n, t in code.graph.nodes(data="node_type") if t == "qubit"]

    pos = {
        q: (i % code.d, -(i // code.d))
        for q in qubits
        for i in [int(q[1:])]
    }

    for check, t in code.graph.nodes(data="node_type"):
        if t not in {"x_check", "z_check"}:
            continue

        ns = list(code.graph[check])
        pos[check] = tuple(sum(pos[q][axis] for q in ns) / len(ns) for axis in (0, 1))

    node_groups, edge_groups = _get_render_groups(code)

    render_info = RenderInfo(
        graph=code.graph,
        x_errors=code.x_errors,
        z_errors=code.z_errors,
        data=code.data,
        node_to_index=code.node_to_index,
        pos=pos,
        figsize=(10, 8),
        edge_groups=edge_groups,
        node_groups=node_groups,
        labels=None,
        legend=True,
        title=title,
        save_path=None,
        return_fig=return_fig,
    )

    return _draw_graph(render_info)