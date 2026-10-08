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


def _tile_row(tile, u, v, L):
    """Qubit indices of the tile placed at (u, v); out-of-grid offsets dropped."""
    cols = []
    for t, block in enumerate(BLOCKS):
        for dx, dy in tile[block]:
            i, j = u + dx, v + dy
            if 0 <= i < L and 0 <= j < L:
                cols.append(t * L * L + i * L + j)
    return cols


def _in_bulk(u, v, L, B):
    return 0 <= u <= L - B and 0 <= v <= L - B


def _used_x(u, v, L, B):
    """X checks: bulk positions plus X boundary positions (u in bulk, bad v)."""
    if _in_bulk(u, v, L, B):
        return True
    return 0 <= u <= L - B and (-(B - 1) <= v <= -1 or L - B + 1 <= v <= L - 1)


def _used_z(u, v, L, B):
    """Z checks: bulk positions plus Z boundary positions (v in bulk, bad u)."""
    if _in_bulk(u, v, L, B):
        return True
    return 0 <= v <= L - B and (-(B - 1) <= u <= -1 or L - B + 1 <= u <= L - 1)


def _dense(rows, n):
    H = np.zeros((len(rows), n), dtype=np.uint8)
    for r, cols in enumerate(rows):
        H[r, cols] = 1
    return H


def build_tile_code(L, x_tile, z_tile, box=None):
    """Build the CSS parity check matrices (H_x, H_z) of a planar tile code.

    Returns two uint8 numpy arrays with n = 2 * L * L columns.  `box` is the tile
    box size B; it defaults to 1 + the largest offset over both tiles.
    """
    if box is None:
        box = 1 + max(max(dx, dy) for tile in (x_tile, z_tile) for block in BLOCKS for dx, dy in tile[block])
    B = box
    n = 2 * L * L

    x_rows, z_rows = [], []
    for u in range(-(B - 1), L):
        for v in range(-(B - 1), L):
            if _used_x(u, v, L, B):
                cols = _tile_row(x_tile, u, v, L)
                if cols:
                    x_rows.append(cols)
            if _used_z(u, v, L, B):
                cols = _tile_row(z_tile, u, v, L)
                if cols:
                    z_rows.append(cols)

    return _dense(x_rows, n), _dense(z_rows, n)


def tile_code_from_params(L, code_params):
    """Build a tile code from a config dict with keys "x_tile" and "z_tile"."""
    return build_tile_code(L, code_params["x_tile"], code_params["z_tile"])
