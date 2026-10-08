import gymnasium as gym
import numpy as np
import networkx as nx
from torch_geometric.data import Data
import torch
import galois
from src.read_config import ConfigParser
from src.environment.tile_code import tile_code_from_params
from src.environment.render_methods import render_ldpc, render_subgraph, render_surface


class QECCode(gym.Env):
    def __init__(self, config: ConfigParser, validate=False):
        super(QECCode, self).__init__()
        self.device = config.device
        self.code_type = config.code_type

        if config.code_type == "toric" or config.code_type == "ldpc":
            self.n, self.k, self.d = config.n, config.k, config.d
            self.l, self.m = config.l, config.m
            self.n_data, self.n_stabilizers = 2*self.l*self.m, self.l*self.m # NOTE: stabilizers are split evenly between X and Z, so total stabilizers is 2*l*m
            self.H_x, self.H_x_T, self.H_z, self.H_z_T = self._init_parity_check_matrices_ldpc(config.code_params)
        elif config.code_type == "surface":
            self.n, self.k, self.d = config.d**2, 1, config.d
            self.n_data, self.n_stabilizers = self.d**2, int((self.d**2-1)/2)
            self.H_x, self.H_x_T, self.H_z, self.H_z_T = self._init_parity_check_matrices_surface()
        elif config.code_type == "tile":
            self.n, self.k, self.d = config.n, config.k, config.d
            self.L = config.L
            self.H_x, self.H_x_T, self.H_z, self.H_z_T = self._init_parity_check_matrices_tile(config.code_params)
            self.n_data, self.n_stabilizers = self.H_x.shape[1], self.H_x.shape[0]

        self.rank_H_x, self.rank_H_z = self._gf2_rank(self.H_x), self._gf2_rank(self.H_z)
        self.no_op_index = self.n_data  # Action index for "no operation"
        self.graph, self.data, self.node_to_index = self._init_graph()

        self.feature_dim = self.data.x.shape[1]

        self.q_idx = torch.tensor([self.node_to_index[f"q{q}"] for q in range(self.n_data)], dtype=torch.long, device=self.device)
        self.x_idx = torch.tensor([self.node_to_index[f"x{i}"] for i in range(self.H_x.shape[0])], dtype=torch.long, device=self.device)
        self.z_idx = torch.tensor([self.node_to_index[f"z{i}"] for i in range(self.H_z.shape[0])], dtype=torch.long, device=self.device)

        self.x_errors = torch.zeros(self.n_data, dtype=torch.long, device=self.device)
        self.z_errors = torch.zeros(self.n_data, dtype=torch.long, device=self.device)
        # X and Z checks may differ in number, so size each syndrome by its own matrix:
        # x_syndrome = H_x @ z_errors (X checks), z_syndrome = H_z @ x_errors (Z checks).
        self.x_syndrome = torch.zeros(self.H_x.shape[0], dtype=torch.long, device=self.device)
        self.z_syndrome = torch.zeros(self.H_z.shape[0], dtype=torch.long, device=self.device)
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
        x_logical_dim = self._logical_support_dimension(
            self.x_errors != 0,
            commuting_checks=self.H_z, stabilizers=self.H_x, stabilizer_rank=self.rank_H_x)

        z_logical_dim = self._logical_support_dimension(
            self.z_errors != 0,
            commuting_checks=self.H_x, stabilizers=self.H_z, stabilizer_rank=self.rank_H_z,)

        return torch.tensor(
            (x_logical_dim > 0) or (z_logical_dim > 0),
            dtype=torch.bool,
            device=self.device,
        )


    def is_error_free(self) -> bool:
        return self.num_x_errors == 0 and self.num_z_errors == 0


    def reset_syndrome(self) -> None:

        self.x_syndrome = ((self.H_x.float() @ self.z_errors.float()) % 2).long()
        self.z_syndrome = ((self.H_z.float() @ self.x_errors.float()) % 2).long()


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

    def render(self, mode="normal"):
        if mode=="subgraph":
            render_subgraph(self.graph, self.data, self.node_to_index, x_errors=self.x_errors, z_errors=self.z_errors)
        else:
            match self.code_type:
                case "toric" | "ldpc": render_ldpc(self.graph, self.x_errors, self.z_errors, self.data, self.node_to_index)
                case "surface": render_surface(self.graph, self.x_errors, self.z_errors, self.data, self.node_to_index, self.d)
                case _: raise ValueError(f"Unknown code type: {self.code_type}")
    #
    # Private helper functions for initializing the code structure, calculating logical operators, and rendering the graph.
    #

    def _logical_support_dimension(self,error_support: torch.Tensor,commuting_checks: torch.Tensor,stabilizers: torch.Tensor,stabilizer_rank: int) -> int:

        # Indices where an error is present / absent.
        support_idx = torch.where(error_support)[0]
        outside_idx = torch.where(~error_support)[0]

        support_size = int(support_idx.numel())

        if support_size == 0:
            return 0

        commuting_rank = self._gf2_rank(commuting_checks[:, support_idx])

        normalizer_support_dim = support_size - commuting_rank
        stabilizer_outside_rank = self._gf2_rank(stabilizers[:, outside_idx])

        stabilizer_support_dim = stabilizer_rank - stabilizer_outside_rank
        logical_support_dim = normalizer_support_dim - stabilizer_support_dim

        return max(0, logical_support_dim)


    def _get_connected_checks(self):
        qubit_to_x = {}
        qubit_to_z = {}

        for q in range(self.n_data):
            qubit_to_x[q] = torch.where(self.H_x[:, q] == 1)[0]
            qubit_to_z[q] = torch.where(self.H_z[:, q] == 1)[0]

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


    def _init_parity_check_matrices_ldpc(self, params):

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


    def _init_parity_check_matrices_tile(self, params):
        H_x, H_z = tile_code_from_params(self.L, params)

        H_x = torch.tensor(H_x, dtype=torch.long, device=self.device)
        H_z = torch.tensor(H_z, dtype=torch.long, device=self.device)
        H_x_T = H_x.t().contiguous()
        H_z_T = H_z.t().contiguous()

        return H_x, H_x_T, H_z, H_z_T


    def _init_parity_check_matrices_surface(self):
        # Sanity check.
        assert self.d >= 3 and self.d % 2 == 1

        d = self.d
        n = d * d

        H_x = []
        H_z = []

        # Bulk plaquettes: checkerboard X/Z pattern.
        for r in range(d - 1):
            for c in range(d - 1):
                q = [
                    r * d + c,
                    r * d + c + 1,
                    (r + 1) * d + c,
                    (r + 1) * d + c + 1,
                ]

                row = np.zeros(n, dtype=np.uint8)
                row[q] = 1

                if (r + c) % 2 == 0:
                    H_x.append(row)
                else:
                    H_z.append(row)

        # X boundary checks
        for r in range(d - 1):
            if r % 2 == 1:
                row = np.zeros(n, dtype=np.uint8)
                row[r * d] = 1
                row[(r + 1) * d] = 1
                H_x.append(row)

            if r % 2 == 0:
                row = np.zeros(n, dtype=np.uint8)
                row[r * d + (d - 1)] = 1
                row[(r + 1) * d + (d - 1)] = 1
                H_x.append(row)

        # Z boundary checks
        for c in range(d - 1):
            if c % 2 == 0:
                row = np.zeros(n, dtype=np.uint8)
                row[c] = 1
                row[c + 1] = 1
                H_z.append(row)

            if c % 2 == 1:
                row = np.zeros(n, dtype=np.uint8)
                row[(d - 1) * d + c] = 1
                row[(d - 1) * d + c + 1] = 1
                H_z.append(row)

        H_x = torch.tensor(np.asarray(H_x), dtype=torch.long,device=self.device)
        H_z = torch.tensor(np.asarray(H_z), dtype=torch.long,device=self.device)
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


    def number_of_overlapping_stabilizers(self, indices=None):

        if indices is None:
            indices = torch.where(self.x_errors == 1)[0].tolist()

        x_overlap = self.H_z[:, indices].sum(axis=1)

        # Count the number of times 2 occurs in the array
        num_x_overlaps_one = (x_overlap == 1).sum().item()
        num_x_overlaps_two = (x_overlap == 2).sum().item()
        return (num_x_overlaps_one, num_x_overlaps_two)

