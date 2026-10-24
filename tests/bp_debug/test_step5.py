"""Contract tests for ``scripts/bp_debug/step5_convergence_audit.py`` (card BD6, issue #13).

Run from the repository root::

    python -m pytest tests/bp_debug/test_step5.py -p no:cacheprovider
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

from scripts.bp_debug import step5_convergence_audit as step5  # noqa: E402
from scripts.bp_debug.common import (  # noqa: E402
    decode_bp,
    load_config,
    make_env,
    single_qubit_shots,
    syndromes,
    uniform_shots,
)


def test_classify_partitions_the_failing_shots():
    """Precedence a > b > c > d, disjoint classes, passing shots in no class."""
    failing = np.array([1, 1, 1, 1, 0, 1], dtype=bool)
    converged = np.array([0, 1, 1, 1, 1, 0], dtype=bool)
    syndrome_free = np.array([0, 1, 1, 0, 1, 0], dtype=bool)
    logical_failure = np.array([0, 1, 0, 0, 0, 0], dtype=bool)
    repo_failure = np.array([1, 0, 1, 1, 0, 1], dtype=bool)

    masks = step5.classify(failing, converged, syndrome_free, logical_failure, repo_failure)
    assert list(masks) == list(step5.CLASSES)
    assert masks["a_not_converged"].tolist() == [True, False, False, False, False, True]
    assert masks["b_converged_logical"].tolist() == [False, True, False, False, False, False]
    assert masks["c_evaluation_bug"].tolist() == [False, False, True, False, False, False]
    assert masks["d_other"].tolist() == [False, False, False, True, False, False]

    # Precedence: a shot that is not converged stays in (a) even when it also looks logical.
    masks = step5.classify(
        np.ones(1, dtype=bool), np.zeros(1, dtype=bool), np.ones(1, dtype=bool),
        np.ones(1, dtype=bool), np.ones(1, dtype=bool),
    )
    assert masks["a_not_converged"][0] and not masks["b_converged_logical"][0]


def test_repo_failure_matches_the_repo_path_on_d5_surface():
    """The residual-state evaluation must agree with the repo `classical_inference` path.

    That equivalence is what lets the audit classify every shot while only re-running the repo
    path on the failing subset.
    """
    config = load_config("d5_surface", "bp")
    env = make_env(config)
    shots = np.concatenate(
        [uniform_shots(config, 10, max_weight=1, seed=0), single_qubit_shots(config, "X")]
    )
    decoded = decode_bp(
        env.code, shots, p_prior=2 * env.curriculum_error_rate / 3, max_iter=step5.MAX_ITER
    )

    cheap = step5.repo_failure(env.code, shots, decoded["x_pred"], decoded["z_pred"])
    repo = step5.repo_path_failure(config, shots)
    assert cheap.dtype == bool and repo.dtype == bool
    assert cheap.any(), "no failing shot in the fixture, the comparison would be vacuous"
    assert np.array_equal(cheap, repo)


def test_effective_convergence_accepts_the_flag_or_a_free_syndrome():
    """A plane counts as converged when the `ldpc` flag says so OR its residual syndrome is free.

    A plane whose hard decision already satisfies its syndrome has converged even if `ldpc` reports
    otherwise, so the `n_not_converged_*_but_*_syndrome_free` counters must not be able to push a
    shot into class (a).
    """
    converged_x = np.array([0, 1, 0, 0], dtype=bool)
    converged_z = np.array([1, 0, 0, 0], dtype=bool)
    x_free = np.array([1, 0, 0, 0], dtype=bool)
    z_free = np.array([0, 1, 1, 0], dtype=bool)

    effective = step5.effective_convergence(converged_x, converged_z, x_free, z_free)
    assert effective.tolist() == [True, True, False, False]

    # The mismatch case: flag False in a plane whose syndrome is free is still a converged shot.
    masks = step5.classify(
        np.ones(1, dtype=bool), effective[:1], np.ones(1, dtype=bool), np.zeros(1, dtype=bool),
        np.ones(1, dtype=bool),
    )
    assert not masks["a_not_converged"][0]


def test_converged_plane_has_a_syndrome_free_residual():
    """`converge` True in a plane means that plane's residual syndrome is zero.

    This is the per-plane pairing `audit_code` relies on: `converged_x`/`x_pred` belong to the
    Z-check syndrome (`syndromes`' *second* output), `converged_z`/`z_pred` to the X-check one
    (`syndromes`' first output).  Crossed names would report phantom flag/syndrome mismatches.
    """
    config = load_config("d5_surface", "bp")
    env = make_env(config)
    shots = uniform_shots(config, 5, max_weight=1, seed=0)
    decoded = decode_bp(
        env.code, shots, p_prior=2 * env.curriculum_error_rate / 3, max_iter=step5.MAX_ITER
    )

    residual = np.stack(
        [shots[:, 0] ^ decoded["x_pred"], shots[:, 1] ^ decoded["z_pred"]], axis=1
    ).astype(np.uint8)
    x_check, z_check = syndromes(env.code, residual)
    x_free, z_free = ~z_check.any(axis=1), ~x_check.any(axis=1)

    assert decoded["converged_x"].any() and decoded["converged_z"].any(), "fixture is vacuous"
    assert np.array_equal(decoded["converged_x"], x_free)
    assert np.array_equal(decoded["converged_z"], z_free)


def test_quick_run_writes_json_and_markdown(tmp_path):
    results = step5.main(["--quick", "--codes", "d5_surface", "--seed", "0", "--out-dir", str(tmp_path)])

    assert results["module"] == "step5_convergence_audit"
    assert results["n_shots"] == step5.QUICK_SHOTS
    record = results["codes"]["d5_surface"]

    counts = {name: entry["count"] for name, entry in record["classes"].items()}
    assert list(counts) == list(step5.CLASSES)
    assert sum(counts.values()) == record["n_failing"]
    assert record["n_repo_path_disagreements"] == 0  # the repo path was re-run on the failing subset
    # These counters cross-check `ldpc`'s `converge` flag against each plane's own residual
    # syndrome (0 for a consistent flag).  A non-zero value is a finding to report, not a bug, so
    # the test only requires the field to exist as a non-negative int.
    for field in ("n_not_converged_x_but_x_syndrome_free", "n_not_converged_z_but_z_syndrome_free"):
        value = record["convergence"][field]
        assert isinstance(value, int) and not isinstance(value, bool) and value >= 0

    for entry in record["classes"].values():
        assert len(entry["examples"]) == min(entry["count"], step5.EXAMPLES)
        assert all(0 <= index < record["n_shots"] for index in entry["examples"])

    json_path = tmp_path / "step5_convergence_audit.json"
    md_path = tmp_path / "step5_convergence_audit.md"
    assert json.loads(json_path.read_text())["codes"]["d5_surface"] == record

    markdown = md_path.read_text()
    assert "## Observation" in markdown and markdown.endswith("\n")
    assert len(markdown.split("## Observation")[1].strip().splitlines()) <= 5
