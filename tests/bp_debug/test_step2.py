"""Contract tests for ``scripts/bp_debug/step2_weight1.py`` (card BD3, issue #13).

Run from the repository root::

    python -m pytest tests/bp_debug/test_step2.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("galois")  # src.environment.code imports galois
pytest.importorskip("ldpc")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bp_debug import common, step2_weight1  # noqa: E402

SURFACE_CODES = ("d3_surface", "d5_surface")
_ENVS = {}


def env_for(code_name):
    """Build the QECEnv once per code (the Tanner graph construction is not free)."""
    if code_name not in _ENVS:
        _ENVS[code_name] = common.make_env(common.load_config(code_name))
    return _ENVS[code_name]


def test_sample_qubits():
    assert np.array_equal(step2_weight1.sample_qubits(9, quick=False), np.arange(9))
    # A code no larger than the quick budget is still enumerated in full.
    assert np.array_equal(step2_weight1.sample_qubits(step2_weight1.QUICK_QUBITS, quick=True),
                          np.arange(step2_weight1.QUICK_QUBITS))

    for n in (9, 25, 288):
        sampled = step2_weight1.sample_qubits(n, quick=True)
        assert len(sampled) == step2_weight1.QUICK_QUBITS
        assert len(set(sampled.tolist())) == len(sampled)
        assert sampled[0] == 0 and sampled[-1] == n - 1  # both ends of the code are covered


@pytest.mark.parametrize("code_name", SURFACE_CODES)
def test_check_degrees_and_boundary_match_matrices(code_name):
    code = env_for(code_name).code
    H_x, H_z = common._as_bits(code.H.x), common._as_bits(code.H.z)

    n_x, n_z = step2_weight1.check_degrees(code)
    assert np.array_equal(n_x, H_x.sum(axis=0))
    assert np.array_equal(n_z, H_z.sum(axis=0))

    boundary = step2_weight1.boundary_qubits(code)
    assert boundary.dtype == bool
    assert np.array_equal(boundary, (n_x == 1) | (n_z == 1))
    # Surface code layout: bulk qubits touch two stabilizers of each type, boundary qubits one.
    assert set(np.unique(n_x)) == {1, 2} and set(np.unique(n_z)) == {1, 2}
    assert boundary.any() and (~boundary).any()
    assert not boundary[(n_x == 2) & (n_z == 2)].any()


def test_quick_run_writes_contract_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(step2_weight1, "RESULTS_DIR", tmp_path)
    results = step2_weight1.main(["--quick", "--codes", "d5_surface", "--seed", "0"])

    json_path = tmp_path / f"{step2_weight1.MODULE}.json"
    md_path = tmp_path / f"{step2_weight1.MODULE}.md"
    assert json_path.exists() and md_path.exists()
    assert json.loads(json_path.read_text()) == results
    assert results["meta"]["module"] == step2_weight1.MODULE
    assert results["meta"]["quick"] and results["meta"]["seed"] == 0

    data = results["codes"]["d5_surface"]
    qubits = data["sampled_qubits"]
    assert data["n"] == 25 and len(qubits) == step2_weight1.QUICK_QUBITS
    assert set(data["priors"]) == {"repo", "0.001", "0.01", "0.1"}
    assert data["priors"]["repo"] == pytest.approx(2 * (0.001 * 144 / data["n"]) / 3)
    assert data["priors"]["0.001"] == 0.001 and data["priors"]["0.1"] == 0.1

    boundary = set(data["boundary_qubits"])
    for pauli in common.PAULIS:
        assert set(data["paulis"][pauli]) == set(data["priors"])
        for row in data["paulis"][pauli].values():
            assert row["n"] == len(qubits)
            assert 0 <= row["failures"] <= row["n"]
            assert row["fraction"] == pytest.approx(row["failures"] / row["n"])
            assert len(row["failing_qubits"]) <= step2_weight1.MAX_LISTED
            assert row["failing_qubits"] == sorted(row["failing_qubits"])
            assert set(row["failing_qubits"]) <= set(qubits)
            assert set(row["boundary_failing_qubits"]) <= set(row["failing_qubits"]) & boundary
            assert row["num_boundary_failing"] == len(row["boundary_failing_qubits"])

    md = md_path.read_text()
    assert "Caption" in md and "## d5_surface" in md and "## Observation" in md
    assert 0 < len(results["observations"]) <= 5


def test_full_run_matches_raw_decode_and_repo_agent(tmp_path, monkeypatch):
    """The d3 surface (n=9) full run: contract rows must match the raw decode and the repo BP path."""
    from src.train_utils.inference import classical_inference, get_agent_and_inference

    monkeypatch.setattr(step2_weight1, "RESULTS_DIR", tmp_path)
    results = step2_weight1.main(["--codes", "d3_surface"])
    data = results["codes"]["d3_surface"]
    assert data["sampled_qubits"] == list(range(9))  # no --quick: every qubit is tested

    config = common.load_config("d3_surface")
    env = common.make_env(config)
    p_prior = 2 * env.curriculum_error_rate / 3
    assert data["priors"]["repo"] == pytest.approx(p_prior)

    for pauli in common.PAULIS:
        shots = common.single_qubit_shots(config, pauli)
        failures = common.decode_bp(env.code, shots, p_prior=p_prior, max_iter=step2_weight1.MAX_ITER)["failure"]
        row = data["paulis"][pauli]["repo"]
        assert row["failing_qubits"] == np.flatnonzero(failures).tolist()
        assert row["failures"] == int(failures.sum())

        # The same shots through the repo's own BPAgent/classical_inference pipeline.
        repo_env = common.make_env(config, shots)
        agent, inference = get_agent_and_inference(config, repo_env, "bp")
        repo_failures = np.array([not bool(inference(agent, repo_env)) for _ in repo_env.shots])
        assert np.array_equal(failures, repo_failures)
