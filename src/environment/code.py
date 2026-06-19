import gymnasium as gym
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data, HeteroData
import torch
import galois
from src.read_config import ConfigParser
from PIL import Image
import io


class QLDPCCode(gym.Env):
    def __init__(self, config: ConfigParser, validate=False):
        super(QLDPCCode, self).__init__()
        self.device = config.device

        self.n, self.k, self.d = config.n, config.k, config.d
        self.l, self.m = config.l, config.m
        self.n_data, self.n_stabilizers = 2*self.l*self.m, self.l*self.m # NOTE: stabilizers are split evenly between X and Z, so total stabilizers is 2*l*m
        self.no_op_index = self.n_data  # Action index for "no operation"

        self.H_x, self.H_x_T, self.H_z, self.H_z_T = self._init_parity_check_matrices(config.code_params)
        self.graph, self.data, self.node_to_index = self._init_graph()

        self.feature_dim = self.data.x.shape[1]

        self.q_idx = torch.tensor([self.node_to_index[f"q{q}"] for q in range(self.n_data)], dtype=torch.long, device=self.device)
        self.x_idx = torch.tensor([self.node_to_index[f"x{i}"] for i in range(self.H_x.shape[0])], dtype=torch.long, device=self.device)
        self.z_idx = torch.tensor([self.node_to_index[f"z{i}"] for i in range(self.H_z.shape[0])], dtype=torch.long, device=self.device)

        self.x_errors = torch.zeros(self.n_data, dtype=torch.long, device=self.device)
        self.z_errors = torch.zeros(self.n_data, dtype=torch.long, device=self.device)
        self.x_syndrome = torch.zeros(self.n_stabilizers, dtype=torch.long, device=self.device)
        self.z_syndrome = torch.zeros(self.n_stabilizers, dtype=torch.long, device=self.device)
        self.num_x_errors, self.num_z_errors = 0, 0

        self.logical_x, self.logical_x_T, self.logical_z, self.logical_z_T = self._get_logical_operators()
        self.qubit_to_x, self.qubit_to_z = self._get_connected_checks()

        if validate:
            self._assert_valid_code()


    def get_logical_state(self):
        logical_x_state = (self.x_errors.float().unsqueeze(0) @ self.logical_z_T) % 2
        logical_z_state = (self.z_errors.float().unsqueeze(0) @ self.logical_x_T) % 2

        return logical_x_state, logical_z_state


    def has_logical_error(self) -> torch.Tensor:
        syndrome_x = (self.x_errors.float().unsqueeze(0) @ self.logical_z_T) % 2
        syndrome_z = (self.z_errors.float().unsqueeze(0) @ self.logical_x_T) % 2

        return syndrome_x.any() or syndrome_z.any()


    def is_error_free(self) -> bool:
        return self.num_x_errors == 0 and self.num_z_errors == 0


    def reset_syndrome(self) -> None:

        self.x_syndrome = ((self.H_x.float() @ self.z_errors.float()) % 2).long()
        self.z_syndrome = ((self.H_z.float() @ self.x_errors.float()) % 2).long()


    def update_graph(self, error_rate) -> None:
        # Graph node features are structured as follows:
        # [is_qubit, is_x_check, is_z_check, x_syndrome, z_syndrome]
        er = torch.tensor(error_rate, dtype=torch.float32, device=self.data.x.device)
        llr = torch.log1p(-er + 1e-10) - torch.log(er + 1e-10)
        self.data.x[self.q_idx, 5] = llr

        self.data.x[self.x_idx, 3] = self.x_syndrome.float()
        self.data.x[self.z_idx, 4] = self.z_syndrome.float()

    def flip(self, qubit_index, error_type=1) -> None:
        # error_type = 1: X error
        # error_type = 2: Z error
        # error_type = 3: Y error
        if qubit_index == self.no_op_index:
            return

        if error_type != 2:
            self.num_x_errors += 1 - 2 * self.x_errors[qubit_index]
            self.x_errors[qubit_index] ^= 1
            self.z_syndrome ^= self.H_z[:, qubit_index].flatten()

        if error_type != 1:
            self.num_z_errors += 1 - 2 * self.z_errors[qubit_index]
            self.z_errors[qubit_index] ^= 1
            self.x_syndrome ^= self.H_x[:, qubit_index].flatten()


    def flip_randomly(self, error_rate) -> None:
        action_mask = torch.rand(self.n_data, device=self.device) < error_rate
        actions = torch.where(action_mask)[0]

        for action in actions.tolist():
            self.flip(action)


    def flip_set_number_of_qubits(self, num_flips) -> None:
        self.flip(torch.randperm(self.n_data, device=self.device)[:num_flips])


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

    #
    # Private helper functions for initializing the code structure, calculating logical operators, and rendering the graph.
    #

    def _get_connected_checks(self):
        qubit_to_x = {}
        qubit_to_z = {}

        for q in range(self.n_data):
            x_checks = torch.where(self.H_x[:, q] == 1)[0].cpu().numpy()
            z_checks = torch.where(self.H_z[:, q] == 1)[0].cpu().numpy()
            qubit_to_x[q] = self.x_idx[x_checks]
            qubit_to_z[q] = self.z_idx[z_checks]

        return qubit_to_x, qubit_to_z


    def _get_logical_operators(self):
        # The logical operators can be derived from parity check matrices as
        # a basis for the kernel of H_x and H_z.

        GF2 = galois.GF(2)

        def quotient_basis(null_space, row_space):
            """
            Return a basis for null_space modulo row_space.
            Keeps nullspace vectors that are independent of the stabilizers.
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
        logical_z = torch.tensor(quotient_basis(H_x_gf2.null_space(), H_z_gf2.row_space()), dtype=torch.long, device=self.device)
        logical_x = torch.tensor(quotient_basis(H_z_gf2.null_space(), H_x_gf2.row_space()), dtype=torch.long, device=self.device)

        return logical_x, logical_x.T.float(), logical_z, logical_z.T.float()


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

        H_x = torch.tensor(H_x, dtype=torch.long, device=self.device)
        H_z = torch.tensor(H_z, dtype=torch.long, device=self.device)
        H_x_T = H_x.t().contiguous()
        H_z_T = H_z.t().contiguous()

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
        qubits = [n for n in self.graph.nodes if self.graph.nodes[n]["node_type"] == "qubit" and self.x_errors[int(n[1:])] == 1]
        x_checks = [n for n in self.graph.nodes if self.graph.nodes[n]["node_type"] == "x_check" and self.data.x[self.node_to_index[n], 3] == 1]
        z_checks = [n for n in self.graph.nodes if self.graph.nodes[n]["node_type"] == "z_check" and self.data.x[self.node_to_index[n], 4] == 1]
        qubit_colors = ["orange" if self.x_errors[int(n[1:])] == 1 else "black" for n in qubits]
        x_check_colors = ["red" if self.data.x[self.node_to_index[n], 3] == 1 else "lightcoral" for n in
                          x_checks]
        z_check_colors = ["blue" if self.data.x[self.node_to_index[n], 4] == 1 else "lightblue" for n in
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


    def number_of_overlapping_stabilizers(self, indices):
        x_overlap = self.H_z[:, indices].sum(axis=1)

        # Count the number of times 2 occurs in the array
        num_x_overlaps_one = (x_overlap == 1).sum().item()
        num_x_overlaps_two = (x_overlap == 2).sum().item()
        return (num_x_overlaps_one, num_x_overlaps_two)


    def get_subgraph_of_indices(self, indices):
        # Return the subgraph of the Tanner graph containing only the specified qubit indices and their neighboring checks.
        nodes_to_include = set()
        for idx in indices:
            qubit_node = f"q{idx}"
            nodes_to_include.add(qubit_node)
            neighbors = self.graph.neighbors(qubit_node)
            nodes_to_include.update(neighbors)

        # Filter on only q and z check nodes to simplify visualization
        nodes_to_include = {n for n in nodes_to_include if self.graph.nodes[n]["node_type"] in ("qubit", "z_check")}

        return self.graph.subgraph(nodes_to_include)


    def render_subgraph(self, indices=None, overlap=None, mistakes=None, total=None):

        if indices is not None:
            subgraph = self.get_subgraph_of_indices(indices)
        else:
            error_indices = torch.where(self.x_errors == 1)[0].tolist()
            subgraph = self.get_subgraph_of_indices(error_indices)

        pos = nx.spring_layout(subgraph, seed=42)  # Use a fixed seed for consistent layouts across runs

        fig = plt.figure(figsize=(8, 6))
        # nx.draw(subgraph, pos, with_labels=True, node_color="lightblue", edge_color="gray")
        nx.draw_networkx_nodes(subgraph, pos,
                               nodelist=[n for n in subgraph.nodes if subgraph.nodes[n]["node_type"] == "qubit"],
                               node_color="orange",
                               node_shape="o",
                               node_size=200,
                               label="Data qubits")

        nx.draw_networkx_nodes(subgraph, pos,
                               nodelist=[n for n in subgraph.nodes if subgraph.nodes[n]["node_type"] == "z_check" and self.data.x[self.node_to_index[n], 4] == 1],
                               node_color="red",
                               node_shape="s",
                               node_size=300,
                               label="X checks")

        nx.draw_networkx_nodes(subgraph, pos,
                               nodelist=[n for n in subgraph.nodes if subgraph.nodes[n]["node_type"] == "z_check" and self.data.x[self.node_to_index[n], 4] == 0],
                               node_color="lightcoral",
                               node_shape="s",
                               node_size=300,
                               label="X checks (no syndrome)")

        nx.draw_networkx_edges(subgraph, pos, edge_color="gray")

        if overlap is not None and mistakes is not None and total is not None:
            plt.title(f"Pattern {overlap} (Mistake frequency: {mistakes} / {total}, {100 * mistakes / total:.2f}%)")

        plt.axis("off")

        plt.show()

        # buf = io.BytesIO()
        # plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        # plt.close(fig)
        # buf.seek(0)
        #
        # img = Image.open(buf).convert("RGB")
        # return img

