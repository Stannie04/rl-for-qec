"""Contract tests for ``scripts/bp_debug/step3_prior_sweep.py`` (card BD4, issue #13).

Run from the repository root::

    python -m pytest tests/bp_debug/test_step3.py -p no:cacheprovider
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

from scripts.bp_debug import step3_prior_sweep as step3  # noqa: E402
from scripts.bp_debug.common import decode_bp, load_config, make_env, random_shots, uniform_shots  # noqa: E402

CODE = "d5_surface"


@pytest.fixture(scope="module")
def quick_payload(tmp_path_factory):
    """One `--quick` run of `main`, written into a temporary results directory."""
    out = tmp_path_factory.mktemp("bp_debug")
    original = step3.RESULTS_DIR
    step3.RESULTS_DIR = out
    try:
        payload = step3.main(["--quick", "--codes", CODE, "--seed", "0"])
    finally:
        step3.RESULTS_DIR = original
    return payload, out


def test_prior_mapping_matches_contract_labels():
    priors = step3.priors_for(0.003, 0.002)
    assert set(priors) == set(step3.PRIOR_LABELS)
    assert priors["matched_2p_over_3"] == pytest.approx(0.002)
    assert priors["2p"] == pytest.approx(0.006)
    assert priors["p"] == pytest.approx(0.003)
    assert priors["repo"] == pytest.approx(0.002)
    assert (priors["0.01"], priors["0.05"], priors["0.1"]) == (0.01, 0.05, 0.1)

    # Uniform weight 1 = one X flip among 25 qubits -> matched prior 1/25, iid rate 3/(2*25).
    assert step3.nominal_p_for_weight(1, 25) == pytest.approx(0.06)
    assert step3.priors_for(step3.nominal_p_for_weight(1, 25), 0.0)["matched_2p_over_3"] == pytest.approx(0.04)


def test_parse_args_defaults_and_quick_flags():
    args = step3.parse_args([])
    assert args.codes == list(step3.DEFAULT_CODES)
    assert args.n_per_weight == step3.DEFAULT_N_PER_WEIGHT
    assert args.seed == 0 and not args.quick

    args = step3.parse_args(["--quick", "--codes", CODE, "--seed", "7"])
    assert args.quick and args.seed == 7 and args.codes == [CODE]

def test_quick_run_writes_json_and_markdown(quick_payload):
    payload, out = quick_payload

    assert payload["module"] == "step3_prior_sweep"
    assert payload["quick"] is True
    assert payload["n_per_weight"] == step3.QUICK_N_PER_WEIGHT
    assert payload["n_per_p"] == step3.QUICK_N_PER_P
    assert payload["prior_labels"] == list(step3.PRIOR_LABELS)
    assert payload["part_a_weights"] == list(step3.PART_A_WEIGHTS)
    assert payload["part_b_p_values"] == list(step3.PART_B_P_VALUES)

    json_path = out / "step3_prior_sweep.json"
    md_path = out / "step3_prior_sweep.md"
    assert json.loads(json_path.read_text()) == payload

    entry = payload["codes"][CODE]
    config = load_config(CODE)
    assert entry["n"] == config.n
    assert entry["repo_prior"] == pytest.approx(2 * config.curriculum_start_error_rate_144 * 144 / config.n / 3)

    for weight in payload["part_a_weights"]:
        block = entry["part_a"][str(weight)]
        assert block["n_shots"] == payload["n_per_weight"]
        assert block["p_nominal"] == pytest.approx(step3.nominal_p_for_weight(weight, config.n))
        expected = step3.priors_for(block["p_nominal"], entry["repo_prior"])
        for label in step3.PRIOR_LABELS:
            row = block["priors"][label]
            assert row["prior"] == pytest.approx(expected[label])
            assert row["n"] == block["n_shots"]
            assert 0 <= row["failures"] <= row["n"]
            assert row["failure_fraction"] == pytest.approx(row["failures"] / row["n"])

    for p in payload["part_b_p_values"]:
        block = entry["part_b"][str(p)]
        assert block["p"] == p and block["n_shots"] == payload["n_per_p"]
        expected = step3.priors_for(p, entry["repo_prior"])
        for label in step3.PRIOR_LABELS:
            row = block["priors"][label]
            assert row["prior"] == pytest.approx(expected[label])
            assert row["failure_fraction"] == pytest.approx(row["failures"] / row["n"])

    markdown = md_path.read_text()
    assert "uniform" in markdown and "iid depolarizing" in markdown
    assert f"sample sizes {payload['n_per_weight']} shots per (code, weight)" in markdown
    assert "## Observation" in markdown
    bullets = [line for line in markdown.splitlines() if line.startswith("- ")]
    assert 0 < len(bullets) <= 5


def test_quick_cells_reproduce_from_the_documented_seed_scheme(quick_payload):
    """The reported failure counts must come from `decode_bp` on the shots the caption describes."""
    payload, _ = quick_payload
    entry = payload["codes"][CODE]
    config = load_config(CODE)
    code = make_env(config).code
    base_seed = payload["seed"] + 100 * list(payload["codes"]).index(CODE)

    # Part A, weight 2, matched prior.
    uniform = uniform_shots(config, payload["n_per_weight"], max_weight=max(payload["part_a_weights"]), seed=base_seed)
    block = uniform[payload["n_per_weight"]: 2 * payload["n_per_weight"]]
    row = entry["part_a"]["2"]["priors"]["matched_2p_over_3"]
    failures = decode_bp(code, block, p_prior=row["prior"], max_iter=payload["max_iter"])["failure"]
    assert int(failures.sum()) == row["failures"]

    # Part B, last p, fixed prior 0.05.
    p_index = len(payload["part_b_p_values"]) - 1
    p = payload["part_b_p_values"][p_index]
    shots = random_shots(config, payload["n_per_p"], p, seed=base_seed + p_index + 1)
    row = entry["part_b"][str(p)]["priors"]["0.05"]
    failures = decode_bp(code, shots, p_prior=row["prior"], max_iter=payload["max_iter"])["failure"]
    assert int(failures.sum()) == row["failures"]


def test_quick_failure_fractions_are_probabilities(quick_payload):
    payload, _ = quick_payload
    fractions = [
        row["failure_fraction"]
        for entry in payload["codes"].values()
        for part in ("part_a", "part_b")
        for block in entry[part].values()
        for row in block["priors"].values()
    ]
    assert fractions and all(0.0 <= value <= 1.0 for value in fractions)
    assert all(np.isfinite(value) for value in fractions)
