import gymnasium as gym
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch

class QLDPCCode(gym.Env):

    def __init__(self, l: int, m: int, **kwargs):

        super().__init__()
        self.device = kwargs.get("device", "cpu")

        self.l, self.m = l, m
        self.n_data, self.n_stabilizers = 2*l*m, 2*l*m

        self.error_rate = kwargs.get("error_rate", 0.05)

        self.H_x, self.H_z = self._init_parity_check_matrices(kwargs.get("params"))

        self.graph, self.data, self.node_to_index = self._init_graph()

        self.action_space = gym.spaces.Discrete(self.n_data)
        self.errors = np.zeros(self.n_data, dtype=np.int8)

        self.episode_steps = 0
        self.max_episode_length = kwargs.get("max_episode_length", 100)
        self.termination_threshold = kwargs.get("termination_threshold", 10)
        self.logical_operators = kwargs.get("logical_operators", [])

        self.previous_num_errors = 0
        self.previous_num_syndromes = 0


    @property
    def syndrome(self):
        # Get the current syndrome based on the errors and the parity check matrices.

        # NOTE: Currently only working with X flips (Z errors).
        return self.H_z @ self.errors % 2


    @property
    def observation(self):
        # The observation is in the form of the node features of the graph.
        # This includes the error rate for qubit nodes, and the syndrome for check nodes.
        # Note that the physical errors are not directly observable.

        return self.data

    def get_reward(self, num_syndromes, num_errors):
        # Reward the agent for surviving
        r = 1

        # Penalize remaining syndrome and physical errors
        r -= 0.25 * (num_syndromes - self.previous_num_syndromes)
        r -= 0.05 * (num_errors - self.previous_num_errors)

        # If syndrome cleared, give a large positive reward for successful decode,
        # but large negative reward if a logical operator is triggered.
        if num_syndromes == 0:
            r += 10.0 if num_errors == 0 else -10.0

        return float(r)


    @property
    def terminated(self):
        return sum(self.errors) > self.termination_threshold


    @property
    def truncated(self):
        return self.episode_steps >= self.max_episode_length


    def _init_parity_check_matrices(self, params):

        def __polynomial_to_matrix(terms, x, y):
            matrix = np.zeros_like(x @ y)

            for x_exp, y_exp in terms:
                term_matrix = np.linalg.matrix_power(x, x_exp) @ np.linalg.matrix_power(y, y_exp)
                matrix += term_matrix

            return matrix

        I_l = np.eye(self.l, dtype=np.int8)
        I_m = np.eye(self.m, dtype=np.int8)

        S_l = np.roll(I_l, 1, axis=1)
        S_m = np.roll(I_m, 1, axis=1)

        x = np.kron(S_l, I_m)
        y = np.kron(I_l, S_m)

        A = __polynomial_to_matrix(params["A"], x, y)
        B = __polynomial_to_matrix(params["B"], x, y)

        return np.hstack([A, B]), np.hstack([B.T, A.T])


    def _init_graph(self):

        ## Create a bipartite Tanner graph from the parity check matrices H_x and H_z in networkx.

        G = nx.Graph()

        n_x, n_qubits = self.H_x.shape
        n_z, _ = self.H_z.shape

        for q in range(n_qubits):
            G.add_node(f"q{q}", node_type="qubit", layer=1)

        for i in range(n_x):
            G.add_node(f"x{i}", node_type="x_check", layer=0)

        for i in range(n_z):
            G.add_node(f"z{i}", node_type="z_check", layer=2)

        for i in range(n_x):
            for j in range(n_qubits):
                if self.H_x[i, j] == 1:
                    G.add_edge(f"x{i}", f"q{j}")

        for i in range(n_z):
            for j in range(n_qubits):
                if self.H_z[i, j] == 1:
                    G.add_edge(f"z{i}", f"q{j}")


        ## Turn the graph into a PyG Data object.

        node_list = list(G.nodes)
        node_to_index = {n: i for i, n in enumerate(node_list)}

        # One-hot encode node types, plus additional feature specific to qubit type
        x = []
        for n in node_list:
            node_type = G.nodes[n]["node_type"]
            if node_type == "qubit":
                x.append([1, 0, 0, self.error_rate])
            elif node_type == "x_check":
                x.append([0, 1, 0, 0])  # Final feature encodes the measurement outcome, which is 0 for all nodes at initialization (no errors)
            elif node_type == "z_check":
                x.append([0, 0, 1, 0])  # Final feature encodes the measurement outcome, which is 0 for all nodes at initialization (no errors)
            else:
                raise ValueError("Unknown node type")
        x = torch.tensor(x, dtype=torch.float32, device=self.device)

        # Encode Edges
        edge_index = []
        for u, v in G.edges:
            edge_index.append([node_to_index[u], node_to_index[v]])
            edge_index.append([node_to_index[v], node_to_index[u]])  # Undirected graph, add both directions
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
        data = Data(x=x, edge_index=edge_index)


        return G, data, node_to_index


    def _update_graph(self):
        x = self.data.x.clone()

        for q in range(self.n_data):
            x[self.node_to_index[f"q{q}"], 3] = self.errors[q]

        for i in range(self.H_x.shape[0]):
            x[self.node_to_index[f"x{i}"], 3] = self.syndrome[i]

        for i in range(self.H_z.shape[0]):
            x[self.node_to_index[f"z{i}"], 3] = self.syndrome[i]

        self.data.x = x


    def _flip_randomly(self):
        # Randomly flip one or more bits according to the error rate.
        mask = np.random.rand(self.n_data) < self.error_rate
        self.errors[mask] ^= 1


    def reset(self, seed=None, options=None):
        self.errors = np.zeros(self.n_data, dtype=np.int8)
        self.episode_steps = 0
        self.previous_num_errors = 0
        self.previous_num_syndromes = 0

        self._flip_randomly()

        self._update_graph()

        return self.observation, {}


    def step(self, action):
        self.errors[action] ^= 1

        num_syndromes = self.syndrome.sum()
        num_errors = self.errors.sum()
        reward = self.get_reward(num_syndromes, num_errors)
        self.previous_num_errors = num_errors
        self.previous_num_syndromes = num_syndromes

        self._flip_randomly()

        self._update_graph()

        return self.observation, reward, self.terminated, self.truncated, {}


    def render(self, mode="human"):

        if mode == "edge_info":
            self._get_edge_information()
            return

        pos = nx.multipartite_layout(self.graph, subset_key="layer")

        # Separate node lists
        qubits = [n for n in self.graph.nodes if self.graph.nodes[n]["node_type"] == "qubit"]
        x_checks = [n for n in self.graph.nodes if self.graph.nodes[n]["node_type"] == "x_check"]
        z_checks = [n for n in self.graph.nodes if self.graph.nodes[n]["node_type"] == "z_check"]

        qubit_colors = ["orange" if self.errors[int(n[1:])] == 1 else "black" for n in qubits]
        x_check_colors = ["red" if self.data.x[self.node_to_index[n], 3] == 1 else "lightcoral" for n in x_checks]
        z_check_colors = ["blue" if self.data.x[self.node_to_index[n], 3] == 1 else "lightblue" for n in z_checks]

        # Color edges based on whether they are connected to an error qubit or not
        error_edges = []
        normal_edges = []

        for u, v in self.graph.edges:
            # Check if either endpoint is a qubit with error
            def is_error_qubit(node):
                return (
                        self.graph.nodes[node]["node_type"] == "qubit" and
                        self.errors[int(node[1:])] == 1
                )

            if is_error_qubit(u) or is_error_qubit(v):
                error_edges.append((u, v))
            else:
                normal_edges.append((u, v))


        plt.figure(figsize=(10, 8))

        # Draw edges
        # Draw normal edges
        nx.draw_networkx_edges(self.graph, pos,
                               edgelist=normal_edges,
                               alpha=0.3)

        # Draw error edges (highlighted)
        nx.draw_networkx_edges(self.graph, pos,
                               edgelist=error_edges,
                               edge_color="red",
                               width=2.0)

        # Draw nodes by type (different shapes & colors)
        nx.draw_networkx_nodes(self.graph, pos,
                               nodelist=qubits,
                               node_color=qubit_colors,
                               node_shape="o",
                               node_size=200,
                               label="Data qubits")

        nx.draw_networkx_nodes(self.graph, pos,
                               nodelist=x_checks,
                               node_color=x_check_colors,
                               node_shape="s",
                               node_size=300,
                               label="X checks")

        nx.draw_networkx_nodes(self.graph, pos,
                               nodelist=z_checks,
                               node_color=z_check_colors,
                               node_shape="^",
                               node_size=300,
                               label="Z checks")

        plt.legend(scatterpoints=1)
        plt.axis("off")
        plt.title("CSS Tanner Graph (Bicycle Code)")

        plt.show()

        # net = Network(notebook=True)
        # net.from_nx(G)
        # net.show("tanner_graph.html")


if __name__ == "__main__":
    code = QLDPCCode(l=3, m=3)

    print(code.H_x)
    print()
    print(code.H_z)