"""RL environment integration test for the [[288, 8, 12]] tile code.

This exercises the `code_type: tile` branch of QLDPCCode, so it needs the full
environment stack (galois/torch/torch_geometric).  It is skipped where galois is
not installed; run it in the `rl-for-qec` conda environment.
"""

import types
from pathlib import Path

import pytest

pytest.importorskip("galois")

import torch  # noqa: E402
import yaml  # noqa: E402

from src.environment.code import QLDPCCode  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "code_config.yml"
CODE_NAME = "288_8_12_tile"
N_X, N_Z, N_DATA = 140, 140, 288


def tile_config():
    """Minimal stand-in for ConfigParser holding the YAML fields of the tile code."""
    with open(CONFIG_PATH) as f:
        fields = yaml.safe_load(f)[CODE_NAME]
    return types.SimpleNamespace(device=torch.device("cpu"), **fields)


@pytest.fixture(scope="module")
def code():
    """One shared build: constructing + validating the code is the expensive part."""
    return QLDPCCode(tile_config(), validate=True)


def test_dimensions_and_syndrome_buffer_sizes(code):
    assert (code.n, code.k, code.d) == (288, 8, 12)
    assert code.L == 12
    assert code.n_data == N_DATA
    assert code.n_stabilizers == N_X

    assert code.H_x.shape == (N_X, N_DATA)
    assert code.H_z.shape == (N_Z, N_DATA)
    assert code.H_x_T.shape == (N_DATA, N_X)
    assert code.H_z_T.shape == (N_DATA, N_Z)

    # X and Z syndromes are sized by their own check matrices.
    assert code.x_syndrome.shape == (N_X,)
    assert code.z_syndrome.shape == (N_Z,)
    assert code.x_syndrome.numel() == code.H_x.shape[0]
    assert code.z_syndrome.numel() == code.H_z.shape[0]


def test_logical_operators_have_k_rows(code):
    assert code.logical_x.shape == (8, N_DATA)
    assert code.logical_z.shape == (8, N_DATA)


@pytest.mark.parametrize("qubit", [0, 137, 287])
def test_single_qubit_flip_matches_check_matrix_column(code, qubit):
    # X error on a single qubit excites exactly the Z checks touching it.
    code.clear_errors()
    code.flip(qubit, error_type=1)
    expected_z = code.H_z[:, qubit]
    assert torch.equal(code.z_syndrome, expected_z)
    assert int(code.z_syndrome.sum()) == int(expected_z.sum()) > 0
    assert int(code.x_syndrome.sum()) == 0

    # Z error on a single qubit excites exactly the X checks touching it.
    code.clear_errors()
    code.flip(qubit, error_type=2)
    assert torch.equal(code.x_syndrome, code.H_x[:, qubit])
    assert int(code.x_syndrome.sum()) > 0
    assert int(code.z_syndrome.sum()) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
