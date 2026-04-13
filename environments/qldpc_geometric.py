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

        H_x = np.hstack([A, B])
        H_z = np.hstack([B.T, A.T])

        return torch.tensor(H_x, dtype=torch.int32, device=self.device), torch.tensor(H_z, dtype=torch.int32, device=self.device)


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

        syndrome = self.syndrome
        # self.data.x[self.q_idx, 3] = self.errors
        self.data.x[self.x_idx, 3] = syndrome.float()
        # self.data.x[self.z_idx, 3] = syndrome


    def _flip_randomly(self):
        # Randomly flip one or more bits according to the error rate.
        mask = torch.rand(self.n_data, device=self.device) < self.error_rate
        self.errors[mask] = 1 - self.errors[mask]


    def _get_edge_information(self):
        x_check_idx, q_idx_x = torch.where(self.H_x == 1)
        z_check_idx, q_idx_z = torch.where(self.H_z == 1)

        # Print per qubit index which checks it is connected to
        for q in range(self.n_data):
            x_checks = x_check_idx[q_idx_x == q].cpu().numpy()
            z_checks = z_check_idx[q_idx_z == q].cpu().numpy()
            print(f"Qubit {q} is connected to X checks {x_checks} and Z checks {z_checks}")


    def reset(self, seed=None, options=None):
        self.errors.zero_()
        self.episode_steps = 0
        self.previous_num_errors = 0
        self.previous_num_syndromes = 0

        self._flip_randomly()

        self._update_graph()

        return self.observation, {}


    def step(self, action):
        raise NotImplementedError


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



class QLDPCTrainEnv(QLDPCCode):

    def __init__(self, l: int, m: int, **kwargs):
        super().__init__(l, m, **kwargs)


    # @property
    # def terminated(self):
    #     # Terminate the environment if there are no errors (successful decoding), or if any logical operator is triggered (logical failure).
    #
    #     if sum(self.errors) == 0:
    #         return True
    #
    #     # When any logical operator is triggered, the episode terminates with a negative reward.
    #     # Logical errors are detected by checking if all qubits in the given logical operator have errors.
    #     # logical_x_ops = self.logical_x & self.errors
    #     # # logical_z_ops = self.logical_z & self.errors
    #     #
    #     # for i in range(self.k):
    #     #     # if torch.all(logical_x_ops[i] == self.logical_x[i]) or torch.all(logical_z_ops[i] == self.logical_z[i]):
    #     #     if torch.all(logical_x_ops[i] == self.logical_x[i]):
    #     #         return True
    #
    #     return False


    def step(self, action):
        # Take a step without random flips afterwards.

        self.errors[action] = 1 - self.errors[action]

        num_syndromes = self.syndrome.sum()
        num_errors = self.errors.sum()
        reward = self.get_reward(num_syndromes, num_errors)
        self.previous_num_errors = num_errors
        self.previous_num_syndromes = num_syndromes

        self._update_graph()

        return self.observation, reward, self.terminated, self.truncated, {}


    def get_reward(self, num_syndromes, num_errors):
        # Reward the agent for surviving
        delta_syndromes = self.previous_num_syndromes - num_syndromes
        delta_errors = self.previous_num_errors - num_errors

        r = -0.1 # Base reward for each step survived

        r += 1.0 * delta_syndromes
        r += 0.3 * delta_errors

        if delta_errors > 0:
            r += 0.5 * delta_errors

        if self.terminated:
            r += 30 if sum(self.errors) == 0 else -30

        return float(r)



class QLDPCEvalEnv(QLDPCCode):

    def __init__(self, l: int, m: int, assert_env: bool = False, **kwargs):
        super().__init__(l, m, **kwargs)

        if assert_env:
            self._assert_environment()


    def _assert_environment(self):

        # Check that the parity check matrices have the correct dimensions
        assert self.H_x.shape == (self.n_stabilizers//2, self.n_data), f"H_x should have shape ({self.n_stabilizers//2}, {self.n_data}), got {self.H_x.shape}"
        assert self.H_z.shape == (self.n_stabilizers//2, self.n_data), f"H_z should have shape ({self.n_stabilizers//2}, {self.n_data}), got {self.H_z.shape}"

        # Sanity check dimensions
        assert self.logical_z.shape[0] == self.logical_x.shape[0] == self.k, f"Number of logical operators should match k ({self.k}), got {self.logical_z.shape[0]} and {self.logical_x.shape[0]}"
        assert self.logical_z.shape[1] == self.logical_x.shape[1] == self.n, f"Logical operators should have length n ({self.n}), got {self.logical_z.shape[1]} and {self.logical_x.shape[1]}"


        # Perform an action, then check that the syndrome updates correctly and that the reward is calculated as expected.
        # _, reward, _, _, _ = self.step(0)  # Flip the first qubit
        # expected_syndrome = self.H_z[:, 0] % 2
        # assert torch.all(self.syndrome == expected_syndrome), f"Syndrome did not update correctly after flipping the first qubit. Expected {expected_syndrome}, got {self.syndrome}"
        # assert torch.all(self.data.x[self.x_idx, 3] == self.syndrome.float()), f"Graph node features did not update correctly after flipping the first qubit. Expected {self.syndrome.float()}, got {self.data.x[self.x_idx, 3]}"
        #
        # self.step(0)


    @property
    def terminated(self):
        # Terminate the environment if any logical operator is triggered.

        # When any logical operator is triggered, the episode terminates with a negative reward.
        # Logical errors are detected by checking if all qubits in the given logical operator have errors.
        # logical_x_ops = self.logical_x & self.errors
        # # logical_z_ops = self.logical_z & self.errors
        #
        # for i in range(self.k):
        #     # if torch.all(logical_x_ops[i] == self.logical_x[i]) or torch.all(logical_z_ops[i] == self.logical_z[i]):
        #     if torch.all(logical_x_ops[i] == self.logical_x[i]):
        #         return True
        #
        # return False

        return self.syndrome.sum() == 0


    def step(self, action):
        self.errors[action] = 1 - self.errors[action]

        num_syndromes = self.syndrome.sum()
        num_errors = self.errors.sum()
        reward = self.get_reward(num_syndromes, num_errors)
        self.previous_num_errors = num_errors
        self.previous_num_syndromes = num_syndromes

        self._flip_randomly()

        self._update_graph()

        return self.observation, reward, self.terminated, self.truncated, {}


    def get_reward(self, num_syndromes, num_errors):
        # Reward for each step survived, ignoring the actual number of syndromes/errors.
        # This allows us to measure episode length until failure without biasing towards specific error/syndrome counts.

        return 1.0
