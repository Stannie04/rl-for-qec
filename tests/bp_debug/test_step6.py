"""Contract tests for ``scripts/bp_debug/step6_reference.py`` (card BD7, issue #13).

Run from the repository root::

    python -m pytest tests/bp_debug/test_step6.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("galois")  # src.environment.code imports galois
pytest.importorskip("ldpc")
pytest.importorskip("pymatching")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bp_debug import step6_reference as step6  # noqa: E402
from scripts.bp_debug.common import (  # noqa: E402
    decode_bp,
    load_config,
    make_env,
    random_shots,
    single_qubit_shots,
    uniform_shots,
)

CODE = "d3_surface"  # the smallest surface code: the MWPM reference is exact there
_ENVS = {}


def env_for(code_name=CODE):
    """Build the QECEnv once per code (the Tanner graph construction is not free)."""
    if code_name not in _ENVS:
        _ENVS[code_name] = make_env(load_config(code_name))
    return _ENVS[code_name]


def test_mwpm_reference_decodes_every_single_qubit_shot():
    config = load_config(CODE)
    code = env_for().code

    for pauli in ("X", "Z"):
        shots = single_qubit_shots(config, pauli)
        assert not step6.mwpm_failure(code, shots).any()

    zeros = np.zeros((3, 2, config.n), dtype=np.int8)
    assert not step6.mwpm_failure(code, zeros).any()

    # Weight-1 uniform shots are one X and one Z flip, each single-qubit correctable.
    assert not step6.mwpm_failure(code, uniform_shots(config, 50, max_weight=1, seed=0)).any()


def test_mwpm_reference_beats_repo_bp_on_single_qubit_x_shots():
    """PROBE: plain BP (the repo path, see test_common) fails on 4 of the 9 single-qubit
    X shots of the d3 surface code, while the MWPM reference decodes all of them."""
    config = load_config(CODE)
    env = env_for()
    shots = single_qubit_shots(config, "X")

    bp = decode_bp(env.code, shots, p_prior=2 * env.curriculum_error_rate / 3, max_iter=step6.BP_MAX_ITER)
    assert bp["failure"].any()
    assert not step6.mwpm_failure(env.code, shots).any()


def test_repo_mwpm_path_matches_the_pymatching_route():
    config = load_config(CODE)
    code = env_for().code
    shots = uniform_shots(config, 20, max_weight=2, seed=0)

    repo_failure, independent_failure, examples = step6.repo_mwpm_path(config, shots)
    assert repo_failure.dtype == bool and independent_failure.dtype == bool
    assert repo_failure.shape == independent_failure.shape == (len(shots),)
    assert np.array_equal(repo_failure, step6.mwpm_failure(code, shots))
    assert examples == []
    assert np.array_equal(repo_failure, independent_failure)


def test_repo_verdict_agrees_with_the_independent_logical_check():
    config = load_config(CODE)
    shots = random_shots(config, 60, 0.005, seed=1)

    repo_failure, independent_failure, examples = step6.repo_mwpm_path(config, shots)
    assert examples == []
    assert np.array_equal(repo_failure, independent_failure)


def test_evaluate_dataset_reports_contract_metrics():
    config = load_config(CODE)
    env = env_for()
    shots = random_shots(config, 30, 0.003, seed=0)

    result = step6.evaluate_dataset(config, env.code, shots, 2 * env.curriculum_error_rate / 3)
    assert result["n"] == 30
    for key in ("bp_failures", "bp_failure_fraction", "bp_non_converged", "mwpm_failures",
                "mwpm_failure_fraction", "repo_mwpm_failures", "repo_mwpm_failure_fraction",
                "repo_vs_independent_disagreements", "repo_vs_independent_disagreement_examples"):
        assert key in result
    assert result["repo_vs_independent_disagreements"] == 0
    assert result["mwpm_failures"] <= result["bp_failures"]


def test_quick_main_writes_json_and_markdown(monkeypatch):
    monkeypatch.chdir(ROOT)
    payload = step6.main(["--quick", "--codes", CODE, "--n-per-weight", "10", "--seed", "0"])

    json_path = ROOT / "results/bp_debug/step6_reference.json"
    md_path = ROOT / "results/bp_debug/step6_reference.md"
    assert json_path.exists() and md_path.exists()

    loaded = json.loads(json_path.read_text())
    assert loaded["codes"] == [CODE] and loaded["quick"] is True
    assert loaded["shots_per_dataset"] == 10
    assert set(loaded["iid"]) == {CODE} and set(loaded["uniform"][CODE]) == {"1", "2", "3"}
    assert loaded["iid"][CODE]["0.001"]["n"] == 10
    assert payload["observations"]

    markdown = md_path.read_text()
    assert "## Observation" in markdown and "Seed 0" in markdown
