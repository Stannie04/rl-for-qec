"""Regression tests for syndrome/coset-based logical-error classification.

Run from the repository root on the ``coset-catalogue`` branch, in the
project environment (which includes galois and torch-geometric)::

    python -m pytest tests/test_logical_error_classification.py -v

The small [[18, 4, 4]] tests enumerate every syndrome-free X and Z residual.
The gross-code integration tests use results/cosets/144_12_12_ldpc/enum_catalogue.csv
as ground truth and exercise every representative under every torus translation.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Importing src.environment.code requires the QEC environment dependencies.
pytest.importorskip("galois")
pytest.importorskip("gymnasium")
pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("networkx")
pytest.importorskip("matplotlib")
pytest.importorskip("PIL")
pytest.importorskip("yaml")

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment.code import QECCode  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "code_config.yml"
CATALOGUE_PATH = ROOT / "results" / "cosets" / "144_12_12_ldpc" / "enum_catalogue.csv"
CODE_NAME_SMALL = "18_4_4_ldpc"
CODE_NAME_GROSS = "144_12_12_ldpc"
GROSS_N = 144
GROSS_L, GROSS_M = 6, 12
GROSS_BLOCK = GROSS_L * GROSS_M


def _build_code(code_name: str) -> QECCode:
    if not CONFIG_PATH.exists():
        pytest.skip(f"Missing code configuration: {CONFIG_PATH}")
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        all_configs = yaml.safe_load(stream)
    if code_name not in all_configs:
        pytest.skip(f"{code_name} is not defined in {CONFIG_PATH}")
    config_fields = all_configs[code_name]
    config_fields["noise_model"] = "depolarizing"
    config = SimpleNamespace(device=torch.device("cpu"), **config_fields)
    return QECCode(config)


@pytest.fixture(scope="module")
def small_code() -> QECCode:
    return _build_code(CODE_NAME_SMALL)


@pytest.fixture(scope="module")
def gross_code() -> QECCode:
    if not CATALOGUE_PATH.exists():
        pytest.skip(f"Gross-code coset catalogue is not present: {CATALOGUE_PATH}")
    return _build_code(CODE_NAME_GROSS)


@pytest.fixture(scope="module")
def coset_tools():
    """Load catalogue helpers by file path, avoiding train_utils package imports."""
    module_path = ROOT / "src" / "train_utils" / "coset_catalogue.py"
    if not module_path.exists():
        pytest.skip(f"Coset-catalogue tools are not present: {module_path}")
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    spec = importlib.util.spec_from_file_location("coset_catalogue_test_module", module_path)
    if spec is None or spec.loader is None:
        pytest.skip(f"Cannot load coset tools from {module_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"Coset-catalogue dependencies are unavailable: {exc}")
    return module


@pytest.fixture(scope="module")
def gross_setup(coset_tools):
    return coset_tools.code_setup(CODE_NAME_GROSS)


@pytest.fixture(scope="module")
def catalogue_rows():
    if not CATALOGUE_PATH.exists():
        pytest.skip(f"Gross-code coset catalogue is not present: {CATALOGUE_PATH}")
    with CATALOGUE_PATH.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _set_residual(
    code: QECCode,
    *,
    x: np.ndarray | None = None,
    z: np.ndarray | None = None,
) -> None:
    """Set residuals without relying on cached syndrome buffers.

    has_logical_error should classify from the current error vectors and their
    actual syndromes, not depend on whether an external caller refreshed caches.
    """
    n = code.n_data
    if x is None:
        code.x_errors.zero_()
    else:
        code.x_errors.copy_(torch.as_tensor(x, dtype=torch.long, device=code.device))
        code.x_syndrome = (code.H.z @ code.x_errors) % 2

    if z is None:
        code.z_errors.zero_()
    else:
        code.z_errors.copy_(torch.as_tensor(z, dtype=torch.long, device=code.device))
        code.z_syndrome = (code.H.x @ code.z_errors) % 2

    assert code.x_errors.numel() == n and code.z_errors.numel() == n


def _env_flags_logical_error(code: QECCode) -> bool:
    result = code.has_logical_error()
    return bool(result.item() if isinstance(result, torch.Tensor) else result)


def _to_numpy(matrix: torch.Tensor) -> np.ndarray:
    return matrix.detach().cpu().numpy().astype(np.uint8) % 2


def _gf2_rank(matrix: np.ndarray) -> int:
    """Small independent GF(2) rank helper for enumerating a basis."""
    a = (np.asarray(matrix).copy() % 2).astype(np.uint8)
    if a.ndim != 2 or a.size == 0:
        return 0
    rows, cols = a.shape
    rank = 0
    for col in range(cols):
        pivots = np.flatnonzero(a[rank:, col])
        if pivots.size == 0:
            continue
        pivot = rank + int(pivots[0])
        a[[rank, pivot]] = a[[pivot, rank]]
        for row in range(rows):
            if row != rank and a[row, col]:
                a[row] ^= a[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def _independent_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = (np.asarray(matrix).copy() % 2).astype(np.uint8)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D matrix")
    chosen: list[np.ndarray] = []
    rank = 0
    for row in matrix:
        candidate = np.vstack([*chosen, row]) if chosen else row[None, :]
        candidate_rank = _gf2_rank(candidate)
        if candidate_rank > rank:
            chosen.append(row.copy())
            rank = candidate_rank
    if not chosen:
        return np.zeros((0, matrix.shape[1]), dtype=np.uint8)
    return np.asarray(chosen, dtype=np.uint8)


def _span_vector(basis: np.ndarray, mask: int) -> np.ndarray:
    vector = np.zeros(basis.shape[1], dtype=np.uint8)
    bit = 0
    while mask:
        if mask & 1:
            vector ^= basis[bit]
        bit += 1
        mask >>= 1
    return vector


def _x_to_z_dual_symmetry(error_x: np.ndarray) -> np.ndarray:
    """Map an X-logical representative to a Z-logical one for the gross code.

    For Hx=[A B], Hz=[B^T A^T], spatial inversion J obeys A J = J A^T
    and B J = J B^T. Thus (u, v) in ker(Hz) maps to (J v, J u) in ker(Hx).
    """
    error_x = (np.asarray(error_x).copy() % 2).astype(np.uint8)
    if error_x.shape != (GROSS_N,):
        raise ValueError(f"Expected an error of length {GROSS_N}")
    out = np.zeros_like(error_x)
    for a in range(GROSS_L):
        for b in range(GROSS_M):
            q = a * GROSS_M + b
            reflected_q = ((-a) % GROSS_L) * GROSS_M + ((-b) % GROSS_M)
            # Swap the L/R blocks and invert both torus coordinates.
            out[q] = error_x[GROSS_BLOCK + reflected_q]
            out[GROSS_BLOCK + q] = error_x[reflected_q]
    return out


def _random_heavy_stabilizer_sums(
    check_matrix: np.ndarray,
    *,
    seed: int,
    count: int = 12,
    minimum_weight: int = 12,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    rows, n = check_matrix.shape
    out: list[np.ndarray] = []
    for _ in range(5000):
        if len(out) >= count:
            break
        coefficients = rng.integers(0, 2, size=rows, dtype=np.uint8)
        if not coefficients.any():
            continue
        candidate = (
            coefficients.astype(np.int16) @ check_matrix.astype(np.int16) % 2
        ).astype(np.uint8)
        if int(candidate.sum()) >= minimum_weight:
            out.append(candidate)
    assert len(out) == count, "Could not construct enough heavy stabilizer sums"
    return out


def _assert_x_residual_matches_coset(code, setup, error: np.ndarray) -> None:
    error = np.asarray(error, dtype=np.uint8) % 2
    assert not (code.H.z.cpu().numpy().astype(np.int64) @ error.astype(np.int64) % 2).any()
    assert setup.coset(error) != 0
    _set_residual(code, x=error)
    assert _env_flags_logical_error(code), f"X logical residual missed; weight={int(error.sum())}"


def _assert_z_residual_matches_coset(code, setup, error: np.ndarray) -> None:
    error = np.asarray(error, dtype=np.uint8) % 2
    assert not (code.H.x.cpu().numpy().astype(np.int64) @ error.astype(np.int64) % 2).any()
    assert np.any((setup.Lx_dual.astype(np.int64) @ error.astype(np.int64)) % 2)
    _set_residual(code, z=error)
    assert _env_flags_logical_error(code), f"Z logical residual missed; weight={int(error.sum())}"


def test_small_code_exhaustively_classifies_every_syndrome_free_residual(small_code):
    """Enumerate the full X and Z kernels of the [[18, 4, 4]] code."""
    hx, hz = _to_numpy(small_code.H.x), _to_numpy(small_code.H.z)
    lx, lz = _to_numpy(small_code.logical.x), _to_numpy(small_code.logical.z)

    cases = (
        ("X", hx, hz, lx, lz),
        ("Z", hz, hx, lz, lx),
    )
    for pauli, stabilizers, commuting_checks, logical_basis, dual_logical_basis in cases:
        # row(stabilizers) plus logical_basis spans the full kernel of the
        # opposite check matrix. Enumerating this basis tests all zero-syndrome
        # residuals, including every stabilizer and every logical coset.
        basis = _independent_rows(np.vstack([stabilizers, logical_basis]))
        expected_dimension = small_code.n_data - _gf2_rank(commuting_checks)
        assert basis.shape[0] == expected_dimension
        assert basis.shape[0] <= 12, "Small-code exhaustive test unexpectedly grew"

        for mask in range(1 << basis.shape[0]):
            error = _span_vector(basis, mask)
            assert not (commuting_checks.astype(np.int64) @ error.astype(np.int64) % 2).any()
            expected = bool((dual_logical_basis.astype(np.int64) @ error.astype(np.int64) % 2).any())
            if pauli == "X":
                _set_residual(small_code, x=error)
            else:
                _set_residual(small_code, z=error)
            actual = _env_flags_logical_error(small_code)
            assert actual == expected, (
                f"{pauli} residual mask={mask}: expected coset verdict {expected}, "
                f"got {actual}, weight={int(error.sum())}"
            )


def test_nonzero_syndrome_is_detected_not_counted_as_logical_failure(small_code):
    """A nonzero syndrome is detectable even if logical overlaps are nonzero."""
    hx, hz = _to_numpy(small_code.H.x), _to_numpy(small_code.H.z)
    lx, lz = _to_numpy(small_code.logical.x), _to_numpy(small_code.logical.z)
    n = small_code.n_data
    rng = np.random.default_rng(20261008)

    for pauli, checks, logical_dual in (("X", hz, lz), ("Z", hx, lx)):
        found = None
        for _ in range(1000):
            error = rng.integers(0, 2, size=n, dtype=np.uint8)
            detected = bool((checks.astype(np.int64) @ error.astype(np.int64) % 2).any())
            has_logical_overlap = bool((logical_dual.astype(np.int64) @ error.astype(np.int64) % 2).any())
            if detected and has_logical_overlap:
                found = error
                break
        assert found is not None, "Expected a detectable residual with nonzero logical overlap"
        if pauli == "X":
            _set_residual(small_code, x=found)
        else:
            _set_residual(small_code, z=found)
        assert not _env_flags_logical_error(small_code)


@pytest.mark.slow
@pytest.mark.integration
def test_gross_catalogue_representatives_and_all_translates_are_logical(gross_code, gross_setup, catalogue_rows):
    """Check all 95 catalogue representatives and their 72 torus translates."""
    assert gross_code.n_data == GROSS_N
    assert gross_setup.n == GROSS_N and gross_setup.k == 12
    assert len(catalogue_rows) == 95
    weight_counts = {weight: sum(int(row["weight"]) == weight for row in catalogue_rows)
                     for weight in (12, 14, 16)}
    assert weight_counts == {12: 9, 14: 26, 16: 60}
    assert sum(int(row["n_cosets"]) for row in catalogue_rows) == (1 << 12) - 1

    all_x_cosets: set[int] = set()
    for row in catalogue_rows:
        support = [int(q) for q in row["support"].split()]
        rep_x = np.zeros(GROSS_N, dtype=np.uint8)
        rep_x[support] = 1
        assert int(rep_x.sum()) == int(row["weight"])
        rep_z = _x_to_z_dual_symmetry(rep_x)

        translated_x_cosets: set[int] = set()
        for permutation in gross_setup.perms:
            shifted_x = gross_setup.shifted(rep_x, permutation)
            shifted_z = gross_setup.shifted(rep_z, permutation)

            _assert_x_residual_matches_coset(gross_code, gross_setup, shifted_x)
            _assert_z_residual_matches_coset(gross_code, gross_setup, shifted_z)
            translated_x_cosets.add(gross_setup.coset(shifted_x))
            all_x_cosets.add(gross_setup.coset(shifted_x))

        assert len(translated_x_cosets) == int(row["n_cosets"]), (
            f"Catalogue orbit {row['orbit']} should contain {row['n_cosets']} "
            f"distinct cosets, got {len(translated_x_cosets)}"
        )

    assert all_x_cosets == set(range(1, 1 << gross_setup.k)), (
        "Translated catalogue representatives should cover all 4095 nontrivial X cosets"
    )


@pytest.mark.slow
@pytest.mark.integration
def test_gross_stabilizer_rows_and_heavy_row_sums_are_not_logical_failures(gross_code, gross_setup):
    hx, hz = gross_code.H.x.cpu().numpy(), gross_code.H.z.cpu().numpy()

    # Every individual stabilizer generator must be harmless.
    for row_index, row in enumerate(hx):
        assert gross_setup.coset(row.astype(np.uint8)) == 0
        _set_residual(gross_code, x=row)
        assert not _env_flags_logical_error(gross_code), f"X stabilizer row {row_index} flagged"

    for row_index, row in enumerate(hz):
        assert not (gross_setup.Lx_dual.astype(np.int64) @ row.astype(np.int64) % 2).any()
        _set_residual(gross_code, z=row)
        assert not _env_flags_logical_error(gross_code), f"Z stabilizer row {row_index} flagged"

    # Random sums provide heavy stabilizers that are not just individual rows.
    heavy_x = _random_heavy_stabilizer_sums(hx, seed=1241)
    heavy_z = _random_heavy_stabilizer_sums(hz, seed=1242)
    assert min(int(e.sum()) for e in heavy_x + heavy_z) >= 12

    for error in heavy_x:
        assert gross_setup.coset(error) == 0
        _set_residual(gross_code, x=error)
        assert not _env_flags_logical_error(gross_code), f"Heavy X stabilizer (w={int(error.sum())}) flagged"

    for error in heavy_z:
        assert not (gross_setup.Lx_dual.astype(np.int64) @ error.astype(np.int64) % 2).any()
        _set_residual(gross_code, z=error)
        assert not _env_flags_logical_error(gross_code), f"Heavy Z stabilizer (w={int(error.sum())}) flagged"


@pytest.mark.slow
@pytest.mark.integration
def test_gross_representative_plus_stabilizer_stays_in_nontrivial_coset(gross_code, gross_setup, catalogue_rows):
    """Adding a same-type stabilizer cannot turn a logical into a harmless residual."""
    hx, hz = gross_code.H.x[0].cpu().numpy(), gross_code.H.z[0].cpu().numpy()
    for row in catalogue_rows:
        support = [int(q) for q in row["support"].split()]
        rep_x = np.zeros(GROSS_N, dtype=np.uint8)
        rep_x[support] = 1
        rep_z = _x_to_z_dual_symmetry(rep_x)

        modified_x = rep_x ^ hx.astype(np.uint8)
        modified_z = rep_z ^ hz.astype(np.uint8)
        assert gross_setup.coset(modified_x) == gross_setup.coset(rep_x) != 0
        assert np.any((gross_setup.Lx_dual.astype(np.int64) @ modified_z.astype(np.int64)) % 2)
        _assert_x_residual_matches_coset(gross_code, gross_setup, modified_x)
        _assert_z_residual_matches_coset(gross_code, gross_setup, modified_z)


@pytest.mark.slow
@pytest.mark.integration
def test_gross_random_syndrome_free_residuals_match_coset_oracle(gross_code, gross_setup):
    """Random kernel vectors agree with the direct Lz e / Lx e coset oracle."""
    hx, hz = gross_code.H.x.cpu().numpy(), gross_code.H.z.cpu().numpy()
    # The catalogue's logical bases are independent of QECCode's selected basis.
    x_generators = np.vstack([hx, gross_setup.Lx_dual]).astype(np.uint8)
    z_generators = np.vstack([hz, gross_setup.Lz]).astype(np.uint8)
    rng = np.random.default_rng(14412012)

    for _ in range(128):
        coeff = rng.integers(0, 2, size=x_generators.shape[0], dtype=np.uint8)
        error_x = (coeff.astype(np.int16) @ x_generators.astype(np.int16) % 2).astype(np.uint8)
        assert not (hz.astype(np.int64) @ error_x.astype(np.int64) % 2).any()
        expected_x = gross_setup.coset(error_x) != 0
        _set_residual(gross_code, x=error_x)
        assert _env_flags_logical_error(gross_code) == expected_x

        coeff = rng.integers(0, 2, size=z_generators.shape[0], dtype=np.uint8)
        error_z = (coeff.astype(np.int16) @ z_generators.astype(np.int16) % 2).astype(np.uint8)
        assert not (hx.astype(np.int64) @ error_z.astype(np.int64) % 2).any()
        expected_z = bool((gross_setup.Lx_dual.astype(np.int64) @ error_z.astype(np.int64) % 2).any())
        _set_residual(gross_code, z=error_z)
        assert _env_flags_logical_error(gross_code) == expected_z


@pytest.mark.integration
def test_disjoint_same_coset_logicals_sum_to_harmless_stabilizer(gross_code, gross_setup):
    """Regression for the weight-24 false positive from support containment."""
    a_support = [3, 4, 15, 25, 73, 77, 85, 87, 88, 110, 111, 122]
    b_support = [21, 33, 35, 44, 92, 93, 94, 96, 104, 107, 129, 142]
    a = np.zeros(GROSS_N, dtype=np.uint8)
    b = np.zeros(GROSS_N, dtype=np.uint8)
    a[a_support] = 1
    b[b_support] = 1
    assert not np.any(a & b), "The counterexample logicals are meant to be disjoint"
    assert int(a.sum()) == int(b.sum()) == 12

    # The provided representatives lie in the same nontrivial coset; their XOR
    # therefore has the zero logical syndrome and is a stabilizer (weight 24).
    assert gross_setup.coset(a) == gross_setup.coset(b) == 1
    residual = a ^ b
    assert int(residual.sum()) == 24
    assert not (gross_code.H.z.cpu().numpy().astype(np.int64) @ residual.astype(np.int64) % 2).any()
    assert gross_setup.coset(residual) == 0

    _set_residual(gross_code, x=residual)
    logical_x_state, _ = gross_code.get_logical_state()
    assert not bool(torch.any(logical_x_state != 0).item())
    assert not _env_flags_logical_error(gross_code), (
        "A stabilizer made from two disjoint, same-coset logicals must not count as a failure"
    )
