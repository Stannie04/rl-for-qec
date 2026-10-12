import gymnasium as gym
import networkx as nx
from torch_geometric.data import Data
import torch
from src.read_config import ConfigParser
from src.environment.parity_check_matrices import ParityCheckMatrices
from src.environment.render_methods import render_spring, render_subgraph, render_surface
from src.environment.logical_operator_arithmetic import get_logical_operators


class QECCode(gym.Env):
    def __init__(self, config: ConfigParser):
        super(QECCode, self).__init__()
        self.device = config.device
        self.code_type = config.code_type

        if config.code_type == "toric" or config.code_type == "ldpc":
            self.n, self.k, self.d = config.n, config.k, config.d
            self.l, self.m = config.l, config.m
            self.n_data, self.n_stabilizers = 2*self.l*self.m, self.l*self.m # NOTE: stabilizers are split evenly between X and Z, so total stabilizers is 2*l*m
            self.H = ParityCheckMatrices("ldpc", l=self.l, m=self.m, code_params=config.code_params, device=self.device)

        elif config.code_type == "surface":
            self.n, self.k, self.d = config.d**2, 1, config.d
            self.n_data, self.n_stabilizers = self.d**2, int((self.d**2-1)/2)
            self.H = ParityCheckMatrices("surface", d=self.d, device=self.device)

        elif config.code_type == "tile":
            self.n, self.k, self.d = config.n, config.k, config.d
            self.L = config.L # Size of the tile code in terms of number of tiles (L x L)
            self.H = ParityCheckMatrices("tile", L=self.L, code_params=config.code_params, device=self.device)
            self.n_data, self.n_stabilizers = self.H.x.shape[1], self.H.x.shape[0]

        self.no_op_index = self.n_data  # Action index for "no operation"
        self.graph, self.data, self.node_to_index = self._init_graph()

        self.feature_dim = self.data.x.shape[1]

        self.q_idx = torch.tensor([self.node_to_index[f"q{q}"] for q in range(self.n_data)], dtype=torch.long, device=self.device)
        self.x_idx = torch.tensor([self.node_to_index[f"x{i}"] for i in range(self.H.x.shape[0])], dtype=torch.long, device=self.device)
        self.z_idx = torch.tensor([self.node_to_index[f"z{i}"] for i in range(self.H.z.shape[0])], dtype=torch.long, device=self.device)

        self.x_errors = torch.zeros(self.n_data, dtype=torch.long, device=self.device)
        self.z_errors = torch.zeros(self.n_data, dtype=torch.long, device=self.device)
        # X and Z checks may differ in number, so size each syndrome by its own matrix:
        # x_syndrome = H_x @ z_errors (X checks), z_syndrome = H_z @ x_errors (Z checks).
        self.x_syndrome = torch.zeros(self.H.x.shape[0], dtype=torch.long, device=self.device)
        self.z_syndrome = torch.zeros(self.H.z.shape[0], dtype=torch.long, device=self.device)
        self.num_x_errors, self.num_z_errors = 0, 0

        self.logical_x, self.logical_x_T, self.logical_z, self.logical_z_T = get_logical_operators(self.H.x, self.H.z, self.n_data, self.k, self.device)
        self.qubit_to_x, self.qubit_to_z = self._get_connected_checks()


    def get_logical_state(self):
        logical_x_state = (self.x_errors.float().unsqueeze(0) @ self.logical_z_T) % 2
        logical_z_state = (self.z_errors.float().unsqueeze(0) @ self.logical_x_T) % 2

        return logical_x_state, logical_z_state


    def has_logical_error(self) -> torch.Tensor:

        syndrome_x = (self.H.z.float() @ self.x_errors.float()) % 2
        syndrome_z = (self.H.x.float() @ self.z_errors.float()) % 2
        logical_x_state, logical_z_state = self.get_logical_state()
        x_is_logical = (
            not bool(torch.any(syndrome_x != 0).item())
            and bool(torch.any(logical_x_state != 0).item())
        )
        z_is_logical = (
            not bool(torch.any(syndrome_z != 0).item())
            and bool(torch.any(logical_z_state != 0).item())
        )
        return torch.tensor(x_is_logical or z_is_logical, dtype=torch.bool, device=self.device)


    def is_error_free(self) -> bool:
        return self.num_x_errors == 0 and self.num_z_errors == 0


    def reset_syndrome(self) -> None:

        self.x_syndrome = ((self.H.x.float() @ self.z_errors.float()) % 2).long()
        self.z_syndrome = ((self.H.z.float() @ self.x_errors.float()) % 2).long()


    def update_graph(self, llr) -> None:
        # Graph node features are structured as follows:
        # [is_qubit, is_x_check, is_z_check, x_syndrome, z_syndrome]
        self.data.x[self.q_idx, 5] = llr

        self.data.x[self.x_idx, 3] = self.x_syndrome.float()
        self.data.x[self.z_idx, 4] = self.z_syndrome.float()


    def flip(self, qubit_index, error_type=1) -> None:
        # error_type = 1: X error
        # error_type = 2: Z error
        # error_type = 3: Y error
        if error_type != 2:
            self.num_x_errors += 1 - 2 * self.x_errors[qubit_index]
            self.x_errors[qubit_index] ^= 1
            self.z_syndrome[self.qubit_to_z[qubit_index]] ^= 1

        if error_type != 1:
            self.num_z_errors += 1 - 2 * self.z_errors[qubit_index]
            self.z_errors[qubit_index] ^= 1
            self.x_syndrome[self.qubit_to_x[qubit_index]] ^= 1


    def flip_randomly(self, error_rate) -> None:
        action_mask = torch.rand(self.n_data, device=self.device) < error_rate
        actions = torch.where(action_mask)[0]

        for action in actions.tolist():
            self.flip(action)


    def flip_set_number_of_qubits(self, num_flips) -> None:
        indices = torch.randperm(self.n_data, device=self.device)[:num_flips]
        for index in indices.tolist():
            self.flip(index)


    def set_error_pattern(self, error_pattern_x, error_pattern_z) -> None:
        if len(error_pattern_x) != self.n_data or len(error_pattern_z) != self.n_data:
            raise ValueError(f"Error patterns must have length {self.n_data}, got {len(error_pattern_x)} and {len(error_pattern_z)}")

        self.x_errors = torch.tensor(error_pattern_x, dtype=torch.long, device=self.device)
        self.z_errors = torch.tensor(error_pattern_z, dtype=torch.long, device=self.device)
        self.num_x_errors = error_pattern_x.sum()
        self.num_z_errors = error_pattern_z.sum()
        self.reset_syndrome()


    def clear_errors(self) -> None:
        self.x_errors.zero_()
        self.z_errors.zero_()
        self.x_syndrome.zero_()
        self.z_syndrome.zero_()
        self.num_x_errors, self.num_z_errors = 0, 0


    def get_number_of_overlapping_stabilizers(self, indices=None):

        if indices is None:
            indices = torch.where(self.x_errors == 1)[0].tolist()

        x_overlap = self.H.z[:, indices].sum(axis=1)

        # Count the number of times 2 occurs in the array
        num_x_overlaps_one = (x_overlap == 1).sum().item()
        num_x_overlaps_two = (x_overlap == 2).sum().item()
        return (num_x_overlaps_one, num_x_overlaps_two)



    def render(self, mode="normal"):
        if mode=="subgraph":
            render_subgraph(self.graph, self.data, self.node_to_index, x_errors=self.x_errors, z_errors=self.z_errors)
        else:
            match self.code_type:
                case "toric" | "ldpc"| "tile": render_spring(self.graph, self.x_errors, self.z_errors, self.data, self.node_to_index)
                case "surface": render_surface(self.graph, self.x_errors, self.z_errors, self.data, self.node_to_index, self.d)
                case _: raise ValueError(f"Unknown code type: {self.code_type}")
    #
    # Private helper functions for initializing the code structure, calculating logical operators, and rendering the graph.
    #

    def _get_connected_checks(self):
        qubit_to_x = {}
        qubit_to_z = {}

        for q in range(self.n_data):
            qubit_to_x[q] = torch.where(self.H.x[:, q] == 1)[0]
            qubit_to_z[q] = torch.where(self.H.z[:, q] == 1)[0]

        return qubit_to_x, qubit_to_z


    def _init_graph(self):

        ## Create a bipartite Tanner graph from the parity check matrices H_x and H_z in networkx.

        G = nx.Graph()

        n_x, n_qubits = self.H.x.shape
        n_z, _ = self.H.z.shape

        for q in range(n_qubits):
            G.add_node(f"q{q}", node_type="qubit", layer=1)

        for i in range(n_x):
            G.add_node(f"x{i}", node_type="x_check", layer=0)

        for i in range(n_z):
            G.add_node(f"z{i}", node_type="z_check", layer=2)

        for i in range(n_x):
            for j in range(n_qubits):
                if self.H.x[i, j] == 1:
                    G.add_edge(f"x{i}", f"q{j}")

        for i in range(n_z):
            for j in range(n_qubits):
                if self.H.z[i, j] == 1:
                    G.add_edge(f"z{i}", f"q{j}")


        ## Turn the graph into a PyG Data object.

        node_list = list(G.nodes)
        node_to_index = {n: i for i, n in enumerate(node_list)}
        # One-hot encode node types, plus additional feature specific to qubit type.
        # Node features are structured as follows:
        # [is_qubit, is_x_check, is_z_check, x_syndrome, z_syndrome, LLR]
        x = []
        for n in node_list:
            node_type = G.nodes[n]["node_type"]
            if node_type == "qubit":
                x.append([1, 0, 0, 0, 0, 0])
            elif node_type == "x_check":
                x.append([0, 1, 0, 0, 0, 0])
            elif node_type == "z_check":
                x.append([0, 0, 1, 0, 0, 0])
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
