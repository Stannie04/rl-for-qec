import gymnasium as gym
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data, HeteroData
import torch
import galois
from src.read_config import ConfigParser


class QLDPCCode(gym.Env):
    def __init__(self, config: ConfigParser, validate=False):
        super(QLDPCCode, self).__init__()
        self.device = config.device

        self.n, self.k, self.d = config.n, config.k, config.d
        self.l, self.m = config.l, config.m
        self.n_data, self.n_stabilizers = 2*self.l*self.m, 2*self.l*self.m
        self.no_op_index = self.n_data  # Action index for "no operation"

        self.H_x, self.H_x_T, self.H_z, self.H_z_T = self._init_parity_check_matrices(config.code_params)
        self.graph, self.data, self.node_to_index = self._init_graph()

        self.feature_dim = self.data.x.shape[1]

        self.q_idx = torch.tensor([self.node_to_index[f"q{q}"] for q in range(self.n_data)], dtype=torch.long, device=self.device)
        self.x_idx = torch.tensor([self.node_to_index[f"x{i}"] for i in range(self.H_x.shape[0])], dtype=torch.long, device=self.device)
        self.z_idx = torch.tensor([self.node_to_index[f"z{i}"] for i in range(self.H_z.shape[0])], dtype=torch.long, device=self.device)

        self.x_errors = torch.zeros(self.n_data, dtype=torch.float32, device=self.device)
        self.z_errors = torch.zeros(self.n_data, dtype=torch.float32, device=self.device)

        self.logical_x, self.logical_z = self._get_logical_operators()

        if validate:
            self._assert_valid_code()


    def has_logical_error(self) -> bool:
        syndrome_x = (self.x_errors.unsqueeze(0) @ self.logical_z.T) % 2
        syndrome_z = (self.z_errors.unsqueeze(0) @ self.logical_x.T) % 2

        return (syndrome_x.any() or syndrome_z.any()).item()
        # return self.x_errors.sum().item() > 6 or self.z_errors.sum().item() > 6


    def is_error_free(self) -> bool:
        return (self.x_errors.sum() == 0 and self.z_errors.sum() == 0).item()


    def get_syndrome(self):
        """
        Returns:
            x_syndrome: Tensor of shape (n_x,) where n_x is the number of X stabilizers. Each entry is 0 or 1 indicating whether that stabilizer is violated by Z errors.
            z_syndrome: Tensor of shape (n_z,) where n_z is the number of Z stabilizers. Each entry is 0 or 1 indicating whether that stabilizer is violated by X errors.
        """

        x_syndrome = (self.H_x @ self.z_errors) % 2
        z_syndrome = (self.H_z @ self.x_errors) % 2

        return x_syndrome, z_syndrome


    def get_action_mask(self):

        _, z_syndrome = self.get_syndrome()

        active_checks = torch.where(z_syndrome > 0)[0]

        mask = torch.zeros(self.n_data + 1, device=self.device)

        for check in active_checks:
            connected = torch.where(self.H_z[check] > 0)[0]
            mask[connected] = 1

        #
        # allow no-op always
        #
        mask[self.no_op_index] = 1

        return mask


    def update_graph(self):
        # Graph node features are structured as follows:
        # [is_qubit, is_x_check, is_z_check, x_syndrome, z_syndrome]

        x_syndrome, z_syndrome = self.get_syndrome()

        self.data.x[self.x_idx, 3] = x_syndrome.float()
        self.data.x[self.z_idx, 4] = z_syndrome.float()


    def flip(self, qubit_index, error_type=1):
        # error_type = 1: X error
        # error_type = 2: Z error
        # error_type = 3: Y error
        if qubit_index == self.no_op_index:
            return

        if error_type != 2:
            self.x_errors[qubit_index] = 1 - self.x_errors[qubit_index]

        if error_type != 1:
            self.z_errors[qubit_index] = 1 - self.z_errors[qubit_index]


    def flip_randomly(self, error_rate):
        action_mask = torch.rand(self.n_data, device=self.device) < error_rate
        actions = torch.where(action_mask)[0]

        for action in actions:
            self.flip(action)


    def flip_set_number_of_qubits(self, num_flips):
        self.flip(torch.randperm(self.n_data, device=self.device)[:num_flips])


    #
    # Private helper functions for initializing the code structure, calculating logical operators, and rendering the graph.
    #

    def _get_logical_operators(self):
        # The logical operators can be derived from parity check matrices as
        # a basis for the kernel of H_x and H_z.

        GF2 = galois.GF(2)

        def quotient_basis(null_space, row_space):
            """
            Return a basis for null_space modulo row_space.
            Keeps nullspace vectors that are independent from the stabilizers.
            """
            GF2 = galois.GF(2)
            null_space = GF2(np.atleast_2d(np.array(null_space, dtype=int)))
            row_space = GF2(np.atleast_2d(np.array(row_space, dtype=int)))

            if row_space.size == 0:
                return null_space

            current = row_space.copy()
            current_rank = current.row_space().shape[0] if current.ndim == 2 else 0
            logicals = []

            for v in null_space:
                candidate = np.vstack([current, v])
                new_rank = GF2(candidate).row_space().shape[0]
                if new_rank > current_rank:
                    logicals.append(v)
                    current = candidate
                    current_rank = new_rank

            if len(logicals) == 0:
                return GF2.Zeros((0, null_space.shape[1]))

            return GF2(np.vstack(logicals))

        H_x_gf2 = GF2(self.H_x.cpu().numpy().astype(np.int8))
        H_z_gf2 = GF2(self.H_z.cpu().numpy().astype(np.int8))

        # logical Z = ker(H_x) / row(H_z)
        # logical X = ker(H_z) / row(H_x)
        logical_z = torch.tensor(quotient_basis(H_x_gf2.null_space(), H_z_gf2.row_space()), dtype=torch.float32, device=self.device)
        logical_x = torch.tensor(quotient_basis(H_z_gf2.null_space(), H_x_gf2.row_space()), dtype=torch.float32, device=self.device)

        return logical_x, logical_z


    def _init_parity_check_matrices(self, params):

        def __polynomial_to_matrix(terms, x, y, z):
            matrix = np.zeros_like(x @ y @ z, dtype=np.int8)
            for x_exp, y_exp, z_exp in terms:
                term_matrix = np.linalg.matrix_power(x, x_exp) @ np.linalg.matrix_power(y, y_exp) @ np.linalg.matrix_power(z, z_exp)
                matrix = (matrix + term_matrix) % 2

            return matrix

        I_l = np.eye(self.l, dtype=np.int8)
        I_m = np.eye(self.m, dtype=np.int8)

        S_l = np.roll(I_l, 1, axis=1)
        S_m = np.roll(I_m, 1, axis=1)

        x = np.kron(S_l, I_m)
        y = np.kron(I_l, S_m)
        z = np.kron(S_l, S_m)

        A = __polynomial_to_matrix(params["A"], x, y, z)
        B = __polynomial_to_matrix(params["B"], x, y, z)

        H_x = np.hstack([A, B])
        H_z = np.hstack([B.T, A.T])

        H_x = torch.tensor(H_x, dtype=torch.float32, device=self.device)
        H_z = torch.tensor(H_z, dtype=torch.float32, device=self.device)
        H_x_T = H_x.t().to_sparse()
        H_z_T = H_z.t().to_sparse()

        return H_x, H_x_T, H_z, H_z_T


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
                # x.append([1, 0, 0, 0, self.error_rate])
                x.append([1, 0, 0, 0, 0])
            elif node_type == "x_check":
                # x.append([0, 1, 0, 0, 0])  # Final feature encodes the measurement outcome, which is 0 for all nodes at initialization (no errors)
                x.append([0, 1, 0, 0, 0])
            elif node_type == "z_check":
                # x.append([0, 0, 1, 0, 0])  # Final feature encodes the measurement outcome, which is 0 for all nodes at initialization (no errors)
                x.append([0, 0, 1, 0, 0])
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


    def _get_edge_information(self):
        x_check_idx, q_idx_x = torch.where(self.H_x == 1)
        z_check_idx, q_idx_z = torch.where(self.H_z == 1)

        # Print per qubit index which checks it is connected to
        for q in range(self.n_data):
            x_checks = x_check_idx[q_idx_x == q].cpu().numpy()
            z_checks = z_check_idx[q_idx_z == q].cpu().numpy()
            print(f"Qubit {q} is connected to X checks {x_checks} and Z checks {z_checks}")


    @staticmethod
    def _gf2_rank(mat: torch.Tensor) -> int:
        """Rank over GF(2)."""
        arr = mat.detach().to("cpu").numpy().astype(np.int8, copy=False)
        if arr.size == 0:
            return 0
        GF2 = galois.GF(2)
        return GF2(arr).row_space().shape[0]


    def _assert_valid_code(self):
        # Basic presence checks
        for name in ("H_x", "H_z", "logical_x", "logical_z", "k"):
            if not hasattr(self, name):
                raise ValueError(f"Missing required attribute: {name}")

        # Shape checks
        if self.H_x.ndim != 2 or self.H_z.ndim != 2:
            raise ValueError("H_x and H_z must both be 2D tensors")

        if self.logical_x.ndim != 2 or self.logical_z.ndim != 2:
            raise ValueError("logical_x and logical_z must both be 2D tensors")

        if self.H_x.shape[1] != self.H_z.shape[1]:
            raise ValueError(
                f"H_x and H_z must have the same number of columns, got "
                f"{self.H_x.shape[1]} and {self.H_z.shape[1]}"
            )

        n = self.H_x.shape[1]

        if self.logical_x.shape[1] != n or self.logical_z.shape[1] != n:
            raise ValueError(
                f"logical_x/logical_z must each have {n} columns, got "
                f"{self.logical_x.shape[1]} and {self.logical_z.shape[1]}"
            )

        # Binary checks
        for name, mat in (("H_x", self.H_x), ("H_z", self.H_z),
                          ("logical_x", self.logical_x), ("logical_z", self.logical_z)):
            if not torch.all((mat == 0) | (mat == 1)):
                raise ValueError(f"{name} must be binary (contain only 0/1 entries)")

        # Commutation: H_x H_z^T = 0 mod 2
        commutation = (self.H_x @ self.H_z.T) % 2
        if torch.any(commutation != 0):
            raise ValueError("Invalid code: H_x and H_z do not commute over GF(2)")

        # Expected number of logical qubits for a CSS code:
        # k = n - rank(H_x) - rank(H_z)
        rank_hx = self._gf2_rank(self.H_x)
        rank_hz = self._gf2_rank(self.H_z)
        expected_k = n - rank_hx - rank_hz

        if expected_k < 0:
            raise ValueError(
                f"Invalid code: computed negative logical qubit count k={expected_k}"
            )

        if self.k != expected_k:
            raise ValueError(
                f"Invalid code: expected k={expected_k} from ranks, but config k={self.k}"
            )

        if self.logical_x.shape[0] != self.k:
            raise ValueError(
                f"Number of logical X operators does not match k: "
                f"{self.logical_x.shape[0]} vs {self.k}"
            )

        if self.logical_z.shape[0] != self.k:
            raise ValueError(
                f"Number of logical Z operators does not match k: "
                f"{self.logical_z.shape[0]} vs {self.k}"
            )

        # Logical X must commute with Z stabilizers; logical Z must commute with X stabilizers
        x_vs_zstab = (self.logical_x @ self.H_z.T) % 2
        z_vs_xstab = (self.logical_z @ self.H_x.T) % 2

        if torch.any(x_vs_zstab != 0):
            raise ValueError("logical_x does not commute with all Z stabilizers")

        if torch.any(z_vs_xstab != 0):
            raise ValueError("logical_z does not commute with all X stabilizers")

        # Logical operators must be independent modulo stabilizers
        rank_hx_with_logicals = self._gf2_rank(torch.cat([self.H_x, self.logical_x], dim=0))
        rank_hz_with_logicals = self._gf2_rank(torch.cat([self.H_z, self.logical_z], dim=0))

        if rank_hx_with_logicals != rank_hx + self.k:
            raise ValueError(
                "logical_x vectors are not linearly independent modulo row(H_x)"
            )

        if rank_hz_with_logicals != rank_hz + self.k:
            raise ValueError(
                "logical_z vectors are not linearly independent modulo row(H_z)"
            )

        # Pairing between chosen logical bases should be nondegenerate
        # (not necessarily identity, but full rank over GF(2)).
        pairing = (self.logical_x @ self.logical_z.T) % 2
        if self._gf2_rank(pairing) != self.k:
            raise ValueError(
                "logical_x and logical_z do not form a valid dual pairing"
            )

        print("Code validation passed: all checks successful.")
        return True

    def render(self, mode="human"):

        if mode == "edge_info":
            self._get_edge_information()
            return

        pos = nx.multipartite_layout(self.graph, subset_key="layer")

        # Separate node lists
        qubits = [n for n in self.graph.nodes if self.graph.nodes[n]["node_type"] == "qubit"]
        x_checks = [n for n in self.graph.nodes if self.graph.nodes[n]["node_type"] == "x_check"]
        z_checks = [n for n in self.graph.nodes if self.graph.nodes[n]["node_type"] == "z_check"]

        qubit_colors = ["orange" if self.x_errors[int(n[1:])] == 1 else "black" for n in qubits]
        x_check_colors = ["red" if self.data.x[self.node_to_index[n], 0] == 1 else "lightcoral" for n in
                          x_checks]
        z_check_colors = ["blue" if self.data.x[self.node_to_index[n], 0] == 1 else "lightblue" for n in
                          z_checks]

        # Color edges based on whether they are connected to an error qubit or not
        error_edges = []
        normal_edges = []

        for u, v in self.graph.edges:
            # Check if either endpoint is a qubit with error
            def is_error_qubit(node):
                return (
                        self.graph.nodes[node]["node_type"] == "qubit" and
                        self.x_errors[int(node[1:])] == 1
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