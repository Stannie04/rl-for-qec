import importlib.util
from pathlib import Path

import numpy as np
import pytest

# Load the module straight from its file: the `src.environment` package __init__ pulls in
# torch/galois-backed code that this numpy-only builder does not need.
_spec = importlib.util.spec_from_file_location("tile_code", Path(__file__).with_name("tile_code.py"))
tile_code = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tile_code)
build_tile_code = tile_code.build_tile_code
tile_code_from_params = tile_code.tile_code_from_params

ORACLE = Path(__file__).with_name("test_data") / "tile_288_8_12_oracle.npz"

L, B = 12, 3
X_TILE = {"L": [[1, 0], [2, 0], [0, 2]], "R": [[2, 1], [2, 2], [0, 0]]}  # f = x + x^2 + y^2
Z_TILE = {"L": [[2, 2], [0, 1], [0, 0]], "R": [[1, 2], [0, 2], [2, 0]]}  # g, f duals


def gf2_rank(H):
    """Rank over GF(2) by Gaussian elimination on a copy of the uint8 matrix."""
    A = (H.copy() % 2).astype(np.uint8)
    rows, cols = A.shape
    rank = 0
    for c in range(cols):
        pivot = None
        for r in range(rank, rows):
            if A[r, c]:
                pivot = r
                break
        if pivot is None:
            continue
        A[[rank, pivot]] = A[[pivot, rank]]
        for r in range(rows):
            if r != rank and A[r, c]:
                A[r] ^= A[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def row_set(H):
    return {tuple(int(x) for x in row) for row in H}


def test_matches_oracle_as_row_sets():
    H_x, H_z = build_tile_code(L, X_TILE, Z_TILE)
    oracle = np.load(ORACLE)
    assert H_x.dtype == np.uint8 and H_z.dtype == np.uint8
    assert row_set(H_x) == row_set(oracle["H_x"])
    assert row_set(H_z) == row_set(oracle["H_z"])


def test_shapes_ranks_commutation_and_k():
    H_x, H_z = build_tile_code(L, X_TILE, Z_TILE)
    assert H_x.shape == (140, 288)
    assert H_z.shape == (140, 288)
    assert gf2_rank(H_x) == 140
    assert gf2_rank(H_z) == 140
    assert int((H_x @ H_z.T % 2).sum()) == 0
    assert 288 - gf2_rank(H_x) - gf2_rank(H_z) == 8


def test_hand_computed_rows_at_origin():
    H_x, H_z = build_tile_code(L, X_TILE, Z_TILE)
    # Row order follows the (u, v) sweep: (0, 0) is X row 2 and Z row 20.
    assert set(np.flatnonzero(H_x[2]).tolist()) == {12, 24, 2, 169, 170, 144}
    assert set(np.flatnonzero(H_z[20]).tolist()) == {26, 1, 0, 158, 146, 168}


def test_bulk_and_boundary_weights_and_qubit_coverage():
    H_x, H_z = build_tile_code(L, X_TILE, Z_TILE)
    w_x, w_z = H_x.sum(1), H_z.sum(1)

    # X rows: 14 per u (v = -2, -1 boundary, then 10 bulk v = 0..9, then 2 boundary).
    x_bulk = {r for r in range(H_x.shape[0]) if 2 <= r % 14 <= 11}
    # Z rows: 10 per u; bulk iff u in 0..9, i.e. the 3rd..12th of the 14 u-blocks.
    z_bulk = {r for r in range(H_z.shape[0]) if 2 <= r // 10 <= 11}
    assert len(x_bulk) == 100 and len(z_bulk) == 100

    assert all(w_x[r] == 6 for r in x_bulk)
    assert all(w_z[r] == 6 for r in z_bulk)
    assert all(w_x[r] in {2, 3, 4} for r in range(H_x.shape[0]) if r not in x_bulk)
    assert all(w_z[r] in {2, 3, 4} for r in range(H_z.shape[0]) if r not in z_bulk)

    assert bool((H_x.sum(0) > 0).all())
    assert bool((H_z.sum(0) > 0).all())


def test_generality_l6():
    H_x, H_z = build_tile_code(6, X_TILE, Z_TILE)
    assert H_x.shape[1] == H_z.shape[1] == 2 * 36
    assert int((H_x @ H_z.T % 2).sum()) == 0


def test_from_params_matches_build():
    H_x, H_z = tile_code_from_params(L, {"x_tile": X_TILE, "z_tile": Z_TILE})
    H_x_ref, H_z_ref = build_tile_code(L, X_TILE, Z_TILE)
    assert np.array_equal(H_x, H_x_ref) and np.array_equal(H_z, H_z_ref)


def test_default_box_equals_explicit_box():
    assert np.array_equal(
        build_tile_code(L, X_TILE, Z_TILE)[0], build_tile_code(L, X_TILE, Z_TILE, box=B)[0]
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
