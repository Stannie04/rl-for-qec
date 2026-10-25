import torch
from src.environment.logical_operator_arithmetic import gf2_rank


class ParityCheckMatrices:
    def __init__(self, code_type, device="cpu", **kwargs):

        self.device = device

        match code_type:
            case "ldpc": self.x, self.z = self._init_parity_check_matrices_ldpc(**kwargs)
            case "tile": self.x, self.z = self._init_parity_check_matrices_tile(**kwargs)
            case "surface": self.x, self.z = self._init_parity_check_matrices_surface(**kwargs)
            case _: raise ValueError(f"Unknown code type: {code_type}")

        self.x_float, self.z_float = self.x.float(), self.z.float()
        self.x_T, self.z_T = self.x.t().contiguous(), self.z.t().contiguous()
        self.x_T_float, self.z_T_float = self.x_T.float(), self.z_T.float()
        self.x_rank, self.z_rank = gf2_rank(self.x), gf2_rank(self.z)


    def _init_parity_check_matrices_ldpc(self, l, m, code_params):

        def __polynomial_to_matrix(terms, x, y, z):
            matrix = np.zeros_like(x @ y @ z, dtype=np.int8)
            for x_exp, y_exp, z_exp in terms:
                term_matrix = np.linalg.matrix_power(x, x_exp) @ np.linalg.matrix_power(y, y_exp) @ np.linalg.matrix_power(z, z_exp)
                matrix = (matrix + term_matrix) % 2

            return matrix

        I_l = np.eye(l, dtype=np.int8)
        I_m = np.eye(m, dtype=np.int8)

        S_l = np.roll(I_l, 1, axis=1)
        S_m = np.roll(I_m, 1, axis=1)

        x = np.kron(S_l, I_m)
        y = np.kron(I_l, S_m)
        z = np.kron(S_l, S_m)

        A = __polynomial_to_matrix(code_params["A"], x, y, z)
        B = __polynomial_to_matrix(code_params["B"], x, y, z)

        H_x = np.hstack([A, B])
        H_z = np.hstack([B.T, A.T])

        return torch.tensor(H_x, dtype=torch.long, device=self.device),  torch.tensor(H_z, dtype=torch.long, device=self.device)


    def _init_parity_check_matrices_tile(self, L, M, code_params):
        H_x, H_z = build_tile_code(L, M, code_params["x_tile"], code_params["z_tile"])
        return torch.tensor(H_x, dtype=torch.long, device=self.device), torch.tensor(H_z, dtype=torch.long, device=self.device)


    def _init_parity_check_matrices_surface(self, d):
        assert d >= 3
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

            if (r + d - 1) % 2 == 0:
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

            if (d - 1 + c) % 2 == 1:
                row = np.zeros(n, dtype=np.uint8)
                row[(d - 1) * d + c] = 1
                row[(d - 1) * d + c + 1] = 1
                H_z.append(row)

        return torch.tensor(np.asarray(H_x), dtype=torch.long,device=self.device), torch.tensor(np.asarray(H_z), dtype=torch.long,device=self.device)



"""Planar tile codes: CSS codes on an L x L grid with two qubits per site.

Each site (i, j) of a planar L x L grid carries two qubits, one in block "L"
(t = 0) and one in block "R" (t = 1), so the code has n = 2 * L * L qubits with
qubit index q = t * L * L + i * L + j.  A tile is a set of B x B box offsets
(dx, dy), one set per block; placing a tile at grid position (u, v) acts on the
site (u + dx, v + dy) of the corresponding block, dropping offsets that fall
outside the grid.  There is one X tile and one Z tile, giving the X and Z parity
checks of a CSS code.  The tiles for the [[288, 8, 12]] code realize the
polynomials f = x + x^2 + y^2 and g = 1 + x^2 y + x^2 y^2 (and their duals).

References:
    - Steffan et al., "Tile codes: high-efficiency quantum codes on a lattice
      with boundary", arXiv:2504.09171.
    - Liang & Chen, arXiv:2504.08887.
"""

import numpy as np

BLOCKS = ("L", "R")  # block index t = 0 -> "L", t = 1 -> "R"


def _tile_row(tile, u, v, L, M):
    cols = []
    for t, block in enumerate(BLOCKS):
        for dx, dy in tile[block]:
            i, j = u + dx, v + dy
            if 0 <= i < L and 0 <= j < M:
                cols.append(t * L * M + i * M + j)
    return cols

def _in_bulk(u, v, L, M, B):
    return 0 <= u <= L - B and 0 <= v <= M - B

def _used_x(u, v, L, M, B):
    if _in_bulk(u, v, L, M, B):
        return True
    return 0 <= u <= L - B and (-(B - 1) <= v <= -1 or M - B + 1 <= v <= M - 1)

def _used_z(u, v, L, M, B):
    if _in_bulk(u, v, L, M, B):
        return True
    return 0 <= v <= M - B and (-(B - 1) <= u <= -1 or L - B + 1 <= u <= L - 1)


def _dense(rows, n):
    H = np.zeros((len(rows), n), dtype=np.uint8)
    for r, cols in enumerate(rows):
        H[r, cols] = 1
    return H


def build_tile_code(L, M, x_tile, z_tile, box=None):
    """Build the CSS parity check matrices (H_x, H_z) of a planar tile code.

    Returns two uint8 numpy arrays with n = 2 * L * L columns.  `box` is the tile
    box size B; it defaults to 1 + the largest offset over both tiles.
    """
    if box is None:
        box = 1 + max(max(dx, dy) for tile in (x_tile, z_tile) for block in BLOCKS for dx, dy in tile[block])
    B = box
    x_rows, z_rows = [], []
    n = 2 * L * M
    for u in range(-(B - 1), L):
        for v in range(-(B - 1), M):
            if _used_x(u, v, L, M, B):
                cols = _tile_row(x_tile, u, v, L, M)
                if cols:
                    x_rows.append(cols)
            if _used_z(u, v, L, M, B):
                cols = _tile_row(z_tile, u, v, L, M)
                if cols:
                    z_rows.append(cols)

    return _dense(x_rows, n), _dense(z_rows, n)