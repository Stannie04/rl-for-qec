"""RL environment integration test for the [[288, 8, 12]] tile code.

This exercises the `code_type: tile` branch of QECCode, so it needs the full
environment stack (galois/torch/torch_geometric).  It is skipped where galois is
not installed; run it in the `rl-for-qec` conda environment.
"""

import types
from pathlib import Path

import pytest

pytest.importorskip("galois")

import torch  # noqa: E402
import yaml  # noqa: E402

from src.environment.code import QECCode  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "code_config.yml"
CODE_NAME = "288_8_12_tile"
N_X, N_Z, N_DATA = 140, 140, 288


def tile_config():
    """Minimal stand-in for ConfigParser holding the YAML fields of the tile code."""
    with open(CONFIG_PATH) as f:
        fields = yaml.safe_load(f)[CODE_NAME]

    fields["noise_model"] = "depolarizing"
    return types.SimpleNamespace(device=torch.device("cpu"), **fields)


@pytest.fixture(scope="module")
def code():
    """One shared build: constructing + validating the code is the expensive part."""
    return QECCode(tile_config())


def test_dimensions_and_syndrome_buffer_sizes(code):
    assert (code.n, code.k, code.d) == (288, 8, 12)
    assert code.L == 12
    assert code.n_data == N_DATA
    assert code.n_stabilizers == N_X

    assert code.H.x.shape == (N_X, N_DATA)
    assert code.H.z.shape == (N_Z, N_DATA)
    assert code.H.x_T.shape == (N_DATA, N_X)
    assert code.H.z_T.shape == (N_DATA, N_Z)

    # X and Z syndromes are sized by their own check matrices.
    assert code.x_syndrome.shape == (N_X,)
    assert code.z_syndrome.shape == (N_Z,)
    assert code.x_syndrome.numel() == code.H.x.shape[0]
    assert code.z_syndrome.numel() == code.H.z.shape[0]


def test_logical_operators_have_k_rows(code):
    assert code.logical.x.shape == (8, N_DATA)
    assert code.logical.z.shape == (8, N_DATA)


@pytest.mark.parametrize("qubit", [0, 137, 287])
def test_single_qubit_flip_matches_check_matrix_column(code, qubit):
    # X error on a single qubit excites exactly the Z checks touching it.
    code.clear_errors()
    code.flip(qubit)
    expected_z = code.H.z[:, qubit]
    assert torch.equal(code.z_syndrome, expected_z)
    assert int(code.z_syndrome.sum()) == int(expected_z.sum()) > 0
    assert int(code.x_syndrome.sum()) == 0

    # Z error on a single qubit excites exactly the X checks touching it.
    code.clear_errors()
    code.flip(qubit + code.n_data)  # Z error is offset by n_data
    assert torch.equal(code.x_syndrome, code.H.x[:, qubit])
    assert int(code.x_syndrome.sum()) > 0
    assert int(code.z_syndrome.sum()) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
