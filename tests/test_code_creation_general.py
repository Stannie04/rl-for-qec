"""Tests for QECCode construction and CSS-code validity.

The constructor tests replace ParityCheckMatrices and get_logical_operators with
small deterministic examples. This isolates QECCode's initialization logic
(graph construction, tensor shapes, indices, syndromes, and derived attributes)
from the algorithms tested in the parity-matrix and logical-operator test files.

The CSS-code validator below intentionally lives in tests, not in QECCode.
"""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.environment.code as code_module  # noqa: E402
from src.environment.code import QECCode  # noqa: E402


def _gf2_rank(matrix: torch.Tensor) -> int:
    """Compute matrix rank over GF(2), independently of production helpers."""
    a = matrix.detach().to(device="cpu", dtype=torch.int64).clone() % 2
    if a.ndim != 2:
        raise AssertionError("GF(2) rank requires a 2D matrix")

    n_rows, n_cols = a.shape
    pivot_row = 0
    for col in range(n_cols):
        if pivot_row >= n_rows:
            break
        candidates = torch.nonzero(a[pivot_row:, col], as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        pivot = pivot_row + int(candidates[0])
        if pivot != pivot_row:
            temp = a[pivot_row].clone()
            a[pivot_row] = a[pivot]
            a[pivot] = temp

        for row in range(n_rows):
            if row != pivot_row and int(a[row, col]) == 1:
                a[row] = torch.bitwise_xor(a[row], a[pivot_row])
        pivot_row += 1

    return pivot_row


def assert_valid_code(code) -> bool:
    """Test-side validator for the CSS-code invariants expected of QECCode."""
    for name in ("n", "k", "d", "n_data", "H", "logical"):
        assert hasattr(code, name), f"Missing required attribute: {name}"

    hx, hz = code.H.x, code.H.z
    lx, lz = code.logical.x, code.logical.z

    assert hx.ndim == hz.ndim == 2, "H_x and H_z must both be 2D tensors"
    assert lx.ndim == lz.ndim == 2, "logical.x and logical.z must both be 2D tensors"
    assert hx.shape[1] == hz.shape[1], (
        f"H_x and H_z must have the same number of columns, got "
        f"{hx.shape[1]} and {hz.shape[1]}"
    )
    n_data = hx.shape[1]
    assert code.n_data == n_data, (
        f"n_data must match parity-check width: {code.n_data} vs {n_data}"
    )
    assert lx.shape[1] == n_data and lz.shape[1] == n_data, (
        f"logical.x/logical.z must each have {n_data} columns, got "
        f"{lx.shape[1]} and {lz.shape[1]}"
    )

    for name, matrix in (("H_x", hx), ("H_z", hz), ("logical.x", lx), ("logical.z", lz)):
        assert torch.all((matrix == 0) | (matrix == 1)).item(), (
            f"{name} must be binary (contain only 0/1 entries)"
        )

    hx_i = hx.to(torch.int64)
    hz_i = hz.to(torch.int64)
    lx_i = lx.to(torch.int64)
    lz_i = lz.to(torch.int64)

    commutation = (hx_i @ hz_i.T) % 2
    assert not torch.any(commutation != 0).item(), (
        "Invalid code: H_x and H_z do not commute over GF(2)"
    )

    rank_hx = _gf2_rank(hx)
    rank_hz = _gf2_rank(hz)
    expected_k = n_data - rank_hx - rank_hz
    assert expected_k >= 0, f"Invalid code: computed negative logical qubit count k={expected_k}"
    assert code.k == expected_k, (
        f"Invalid code: expected k={expected_k} from ranks, but config k={code.k}"
    )
    assert lx.shape[0] == code.k, (
        f"Number of logical X operators does not match k: {lx.shape[0]} vs {code.k}"
    )
    assert lz.shape[0] == code.k, (
        f"Number of logical Z operators does not match k: {lz.shape[0]} vs {code.k}"
    )

    x_vs_zstab = (lx_i @ hz_i.T) % 2
    assert not torch.any(x_vs_zstab != 0).item(), (
        "logical.x does not commute with all Z stabilizers"
    )
    z_vs_xstab = (lz_i @ hx_i.T) % 2
    assert not torch.any(z_vs_xstab != 0).item(), (
        "logical.z does not commute with all X stabilizers"
    )

    rank_hx_with_logicals = _gf2_rank(torch.cat((hx, lx), dim=0))
    rank_hz_with_logicals = _gf2_rank(torch.cat((hz, lz), dim=0))
    assert rank_hx_with_logicals == rank_hx + code.k, (
        "logical.x vectors are not linearly independent modulo row(H_x)"
    )
    assert rank_hz_with_logicals == rank_hz + code.k, (
        "logical.z vectors are not linearly independent modulo row(H_z)"
    )

    pairing = (lx_i @ lz_i.T) % 2
    assert _gf2_rank(pairing) == code.k, (
        "logical.x and logical.z do not form a valid dual pairing"
    )

    # QECCode caches the transposes for logical-state calculations.
    if hasattr(code, "logical.x_T"):
        assert torch.equal(code.logical.x_T, code.logical.x.T), "logical.x_T is not logical.x.T"
    if hasattr(code, "logical.z_T"):
        assert torch.equal(code.logical.z_T, code.logical.z.T), "logical.z_T is not logical.z.T"

    return True


class _FakeParityCheckMatrices:
    """Small, known CSS matrices used to isolate QECCode initialization."""

    def __init__(self, _code_type, **kwargs):
        device = kwargs.get("device", "cpu")

        if "d" in kwargs:
            # A simple [[9, 1]] CSS example for the surface-code constructor path.
            n_data = int(kwargs["d"]) ** 2
            if n_data != 9:
                raise ValueError("The test double supports surface distance d=3 only")
            hx = torch.zeros((4, n_data), dtype=torch.float32, device=device)
            for i in range(4):
                hx[i, 2 * i : 2 * i + 2] = 1
            hz = hx.clone()
        elif "L" in kwargs:
            # Tile constructor path: keep the fixture small but with several logicals.
            n_data = 2 * int(kwargs["L"]) ** 2
            hx = torch.zeros((1, n_data), dtype=torch.float32, device=device)
            hx[0, :2] = 1
            hz = hx.clone()
        elif "l" in kwargs and "m" in kwargs:
            n_data = 2 * int(kwargs["l"]) * int(kwargs["m"])
            hx = torch.zeros((1, n_data), dtype=torch.float32, device=device)
            hx[0, :2] = 1
            hz = hx.clone()
        else:
            raise AssertionError(f"Unexpected ParityCheckMatrices arguments: {kwargs}")

        self.x = hx
        self.z = hz
        self.x_rank = _gf2_rank(hx)
        self.z_rank = _gf2_rank(hz)


class _FakeLogicalOperators:
    """Fake logical operators for isolated QECCode initialization tests."""

    def __init__(self, H_x, H_z, n_data, k, device):
        self.x, self.x_T, self.z, self.z_T, = _fake_get_logical_operators(H_x, H_z, n_data, k, device)
        self.x_T_float = self.x_T.float()
        self.z_T_float = self.z_T.float()


def _fake_get_logical_operators(hx, hz, n_data, k, device):
    """Return logical bases dual under the binary dot product for test matrices."""
    lx = torch.zeros((k, n_data), dtype=torch.float32, device=device)
    lz = torch.zeros((k, n_data), dtype=torch.float32, device=device)

    if n_data == 9 and hx.shape[0] == 4 and hz.shape[0] == 4:
        # The ninth qubit is unconstrained by the four pair checks.
        lx[0, 8] = 1
        lz[0, 8] = 1
    else:
        # For H_x = H_z = [1100...], columns 2..n-1 form a logical basis.
        if k != n_data - 2:
            raise AssertionError(f"Unexpected fixture dimensions n={n_data}, k={k}")
        indices = torch.arange(k, device=device)
        lx[indices, indices + 2] = 1
        lz[indices, indices + 2] = 1

    return lx, lx.T.contiguous(), lz, lz.T.contiguous()


@pytest.fixture
def use_deterministic_code_dependencies(monkeypatch):
    """Patch matrix/logical generation so tests focus on QECCode's constructor."""
    monkeypatch.setattr(code_module, "ParityCheckMatrices", _FakeParityCheckMatrices)
    monkeypatch.setattr(code_module, "LogicalOperators", _FakeLogicalOperators)


def _config(code_type: str):
    """Create a minimal config with the attributes QECCode reads for each type."""
    if code_type in {"ldpc", "toric"}:
        n_data = 4  # 2*l*m with l=2, m=1
        return SimpleNamespace(
            code_type=code_type,
            device="cpu",
            n=n_data,
            k=n_data - 2,
            d=2,
            l=2,
            m=1,
            code_params={},
        )
    if code_type == "surface":
        return SimpleNamespace(code_type="surface", device="cpu", d=3, code_params={})
    if code_type == "tile":
        n_data = 8  # test-double convention: 2*L*L with L=2
        return SimpleNamespace(
            code_type="tile",
            device="cpu",
            n=n_data,
            k=n_data - 2,
            d=3,
            L=2,
            code_params={},
        )
    raise ValueError(f"Unknown test code type: {code_type}")


@pytest.mark.parametrize("code_type", ["ldpc", "toric", "surface", "tile"])
def test_constructor_initializes_supported_code_types(code_type, use_deterministic_code_dependencies):
    config = _config(code_type)
    code = QECCode(config)

    assert code.code_type == code_type
    assert code.device == "cpu"
    assert code.n_data == code.H.x.shape[1] == code.H.z.shape[1]
    assert code.no_op_index == code.n_data
    assert code.feature_dim == 6

    # All graph nodes and edge indices should agree with the matrices.
    n_x_checks = code.H.x.shape[0]
    n_z_checks = code.H.z.shape[0]
    expected_nodes = code.n_data + n_x_checks + n_z_checks
    assert code.graph.number_of_nodes() == expected_nodes
    assert code.data.x.shape == (expected_nodes, 6)
    assert code.data.edge_index.ndim == 2 and code.data.edge_index.shape[0] == 2
    expected_undirected_edges = int(code.H.x.sum().item() + code.H.z.sum().item())
    assert code.data.edge_index.shape[1] == 2 * expected_undirected_edges

    assert code.q_idx.shape == (code.n_data,)
    assert code.x_idx.shape == (n_x_checks,)
    assert code.z_idx.shape == (n_z_checks,)
    assert code.x_errors.shape == (code.n_data,)
    assert code.z_errors.shape == (code.n_data,)
    assert torch.count_nonzero(code.x_errors).item() == 0
    assert torch.count_nonzero(code.z_errors).item() == 0
    assert code.x_syndrome.shape == (n_x_checks,)
    assert code.z_syndrome.shape == (n_z_checks,)
    assert torch.count_nonzero(code.x_syndrome).item() == 0
    assert torch.count_nonzero(code.z_syndrome).item() == 0
    assert code.num_x_errors == 0 and code.num_z_errors == 0

    assert code.logical.x.shape == (code.k, code.n_data)
    assert code.logical.z.shape == (code.k, code.n_data)
    assert code.logical.x_T.shape == (code.n_data, code.k)
    assert code.logical.z_T.shape == (code.n_data, code.k)
    assert torch.equal(code.logical.x_T, code.logical.x.T)
    assert torch.equal(code.logical.z_T, code.logical.z.T)

    # Every data qubit has a connectivity entry, even if it touches no checks.
    assert set(code.qubit_to_x) == set(range(code.n_data))
    assert set(code.qubit_to_z) == set(range(code.n_data))
    for q in range(code.n_data):
        assert torch.equal(code.qubit_to_x[q], torch.where(code.H.x[:, q] == 1)[0])
        assert torch.equal(code.qubit_to_z[q], torch.where(code.H.z[:, q] == 1)[0])

    assert assert_valid_code(code)


def test_graph_contains_correct_node_types_and_initial_features(use_deterministic_code_dependencies):
    code = QECCode(_config("ldpc"))

    for q in range(code.n_data):
        node = f"q{q}"
        assert node in code.graph
        assert code.graph.nodes[node]["node_type"] == "qubit"
        assert code.data.x[code.node_to_index[node]].tolist() == [1, 0, 0, 0, 0, 0]

    for i in range(code.H.x.shape[0]):
        node = f"x{i}"
        assert node in code.graph
        assert code.graph.nodes[node]["node_type"] == "x_check"
        assert code.data.x[code.node_to_index[node]].tolist() == [0, 1, 0, 0, 0, 0]

    for i in range(code.H.z.shape[0]):
        node = f"z{i}"
        assert node in code.graph
        assert code.graph.nodes[node]["node_type"] == "z_check"
        assert code.data.x[code.node_to_index[node]].tolist() == [0, 0, 1, 0, 0, 0]


def test_test_side_validator_accepts_valid_css_code(use_deterministic_code_dependencies):
    code = QECCode(_config("ldpc"))
    assert assert_valid_code(code) is True


def test_test_side_validator_rejects_non_binary_parity_matrix(use_deterministic_code_dependencies):
    code = QECCode(_config("ldpc"))
    code.H.x[0, 0] = 2
    with pytest.raises(AssertionError, match="H.x must be binary"):
        assert_valid_code(code)


def test_test_side_validator_rejects_non_commuting_stabilizers(use_deterministic_code_dependencies):
    code = QECCode(_config("ldpc"))
    code.H.z[0].zero_()
    code.H.z[0, 0] = 1  # H_x dot H_z is now 1 over GF(2).
    with pytest.raises(AssertionError, match="do not commute over GF\\(2\\)"):
        assert_valid_code(code)


def test_test_side_validator_rejects_wrong_logical_qubit_count(use_deterministic_code_dependencies):
    code = QECCode(_config("ldpc"))
    code.k += 1
    with pytest.raises(AssertionError, match="config k="):
        assert_valid_code(code)


def test_test_side_validator_rejects_wrong_logical_operator_width(use_deterministic_code_dependencies):
    code = QECCode(_config("ldpc"))
    code.logical.x = torch.zeros((code.k, code.n_data + 1), dtype=torch.float32)
    with pytest.raises(AssertionError, match="logical.x/logical.z must each"):
        assert_valid_code(code)


def test_test_side_validator_rejects_logical_x_that_does_not_commute(use_deterministic_code_dependencies):
    code = QECCode(_config("ldpc"))
    code.logical.x[0, 0] = 1  # Gives odd overlap with the Z-check [1, 1, 0, 0].
    with pytest.raises(AssertionError, match="logical.x does not commute"):
        assert_valid_code(code)


def test_test_side_validator_rejects_logical_z_that_does_not_commute(use_deterministic_code_dependencies):
    code = QECCode(_config("ldpc"))
    code.logical.z[0, 0] = 1  # Gives odd overlap with the X-check [1, 1, 0, 0].
    with pytest.raises(AssertionError, match="logical.z does not commute"):
        assert_valid_code(code)


def test_test_side_validator_rejects_logical_in_stabilizer_row_space(use_deterministic_code_dependencies):
    code = QECCode(_config("ldpc"))
    code.logical.x[0] = code.H.x[0]
    with pytest.raises(AssertionError, match="not linearly independent modulo row\\(H.x\\)"):
        assert_valid_code(code)


def test_test_side_validator_rejects_parity_matrices_with_different_widths(use_deterministic_code_dependencies):
    code = QECCode(_config("ldpc"))
    code.H.z = torch.zeros((code.H.z.shape[0], code.n_data - 1), dtype=torch.float32)
    with pytest.raises(AssertionError, match="same number of columns"):
        assert_valid_code(code)
