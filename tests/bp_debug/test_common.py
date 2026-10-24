"""Contract tests for ``scripts/bp_debug/common.py`` (card BD1, issue #13).

Run from the repository root::

    python -m pytest tests/bp_debug/test_common.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("galois")  # src.environment.code imports galois
pytest.importorskip("ldpc")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bp_debug.common import (  # noqa: E402
    decode_bp,
    independent_logical_failure,
    load_config,
    make_env,
    random_shots,
    single_qubit_shots,
    syndromes,
    uniform_shots,
    weights,
)

SURFACE_CODES = ("d3_surface", "d5_surface")
_ENVS = {}


def env_for(code_name):
    """Build the QECEnv once per code (the Tanner graph construction is not free)."""
    if code_name not in _ENVS:
        _ENVS[code_name] = make_env(load_config(code_name))
    return _ENVS[code_name]


def test_shapes_and_dtypes():
    config = load_config("d3_surface")
    n = config.n

    uniform = uniform_shots(config, 4, max_weight=3, seed=0)
    assert uniform.shape == (12, 2, n)
    assert uniform.dtype == np.int8
    assert np.array_equal(uniform, uniform_shots(config, 4, max_weight=3, seed=0))

    random = random_shots(config, 7, 0.01, seed=0)
    assert random.shape == (7, 2, n)
    assert random.dtype == np.int8
    assert np.array_equal(random, random_shots(config, 7, 0.01, seed=0))

    for pauli in ("X", "Y", "Z"):
        shots = single_qubit_shots(config, pauli)
        assert shots.shape == (n, 2, n)
        assert shots.dtype == np.int8
        assert np.array_equal(weights(shots), np.ones(n))

    x_shots = single_qubit_shots(config, "X")
    assert np.array_equal(x_shots[:, 0], np.eye(n, dtype=np.int8))
    assert not x_shots[:, 1].any()

    y_shots = single_qubit_shots(config, "Y")
    assert np.array_equal(y_shots[:, 0], y_shots[:, 1])

    with pytest.raises(ValueError):
        single_qubit_shots(config, "Q")


def test_weights_counts_touched_qubits():
    config = load_config("d3_surface")

    # Nominal weight 1 = one X position AND one Z position, which may coincide.
    unit = uniform_shots(config, 5, max_weight=1, seed=0)
    assert unit.shape[0] == 5
    touched = weights(unit)
    assert ((touched >= 1) & (touched <= 2)).all()

    zeros = np.zeros((2, 2, config.n), dtype=np.int8)
    assert np.array_equal(weights(zeros), np.zeros(2))


@pytest.mark.parametrize("code_name", SURFACE_CODES)
def test_syndromes_match_env_code(code_name):
    config = load_config(code_name)
    code = env_for(code_name).code
    shots = uniform_shots(config, 12, max_weight=5, seed=3)

    x_syndrome, z_syndrome = syndromes(code, shots)
    assert x_syndrome.dtype == np.uint8 and z_syndrome.dtype == np.uint8
    assert x_syndrome.shape == (len(shots), code.H.x.shape[0])
    assert z_syndrome.shape == (len(shots), code.H.z.shape[0])

    for i, (x_errors, z_errors) in enumerate(shots):
        code.set_error_pattern(x_errors, z_errors)
        assert np.array_equal(x_syndrome[i], code.x_syndrome.cpu().numpy())
        assert np.array_equal(z_syndrome[i], code.z_syndrome.cpu().numpy())


@pytest.mark.parametrize("code_name", SURFACE_CODES)
def test_independent_logical_failure_matches_code_oracle(code_name):
    config = load_config(code_name)
    code = env_for(code_name).code
    shots = uniform_shots(config, 40, max_weight=5, seed=1)  # 200 low-weight shots
    assert len(shots) == 200

    expected = np.zeros(len(shots), dtype=bool)
    for i, (x_errors, z_errors) in enumerate(shots):
        code.set_error_pattern(x_errors, z_errors)
        expected[i] = (not bool(code.is_syndrome_free())) or code.has_logical_error()

    no_prediction = independent_logical_failure(code, shots, np.zeros_like(shots[:, 0]), np.zeros_like(shots[:, 1]))
    assert no_prediction.dtype == bool
    assert np.array_equal(no_prediction, expected)

    # A perfect prediction leaves a zero residual, which never fails.
    assert not independent_logical_failure(code, shots, shots[:, 0], shots[:, 1]).any()


def test_zero_error_shots_never_fail():
    config = load_config("d3_surface")
    code = env_for("d3_surface").code
    shots = np.zeros((4, 2, config.n), dtype=np.int8)
    zeros = np.zeros((4, config.n), dtype=np.int8)

    code.set_error_pattern(shots[0, 0], shots[0, 1])
    assert bool(code.is_syndrome_free()) and not code.has_logical_error()
    assert not independent_logical_failure(code, shots, zeros, zeros).any()


def test_decode_bp_returns_contract_dict():
    config = load_config("d3_surface")
    code = env_for("d3_surface").code
    shots = uniform_shots(config, 3, max_weight=2, seed=0)

    result = decode_bp(code, shots, p_prior=0.01)
    assert set(result) == {"x_pred", "z_pred", "converged_x", "converged_z", "failure"}
    for key in ("x_pred", "z_pred"):
        assert result[key].shape == (6, config.n)
        assert result[key].dtype == np.uint8
        assert set(np.unique(result[key])) <= {0, 1}
    for key in ("converged_x", "converged_z", "failure"):
        assert result[key].shape == (6,)
        assert result[key].dtype == bool

    with pytest.raises(ValueError):
        decode_bp(code, shots, p_prior=0.01, decoder="nope")


def test_decode_bp_supports_osd_and_lsd():
    config = load_config("d3_surface")
    code = env_for("d3_surface").code
    shots = uniform_shots(config, 1, max_weight=1, seed=0)

    for decoder in ("bp_osd", "bp_lsd"):
        result = decode_bp(code, shots, p_prior=0.01, decoder=decoder)
        assert result["x_pred"].shape == (1, config.n)
        assert bool(result["converged_x"][0]) and bool(result["converged_z"][0])


def test_single_qubit_x_shots_decoded_by_bp_on_d3_surface():
    """PROBE: plain BP does not decode *every* single-qubit X shot on the d3 surface.

    The contract expects all of them to succeed; with the repo's prior BP fails on qubits
    2, 3, 5 and 6 (it never converges there) - exactly the failure the investigation is
    about.  The test therefore pins `decode_bp` against the repo's own `BPAgent` path,
    and checks that OSD does decode all nine shots.
    """
    from src.train_utils.inference import classical_inference, get_agent_and_inference

    config = load_config("d3_surface")
    shots = single_qubit_shots(config, "X")
    env = make_env(config, shots)  # fresh env: classical_inference consumes env.shots in order

    agent, inference = get_agent_and_inference(config, env, "bp")
    repo_success = np.array([bool(inference(agent, env)) for _ in shots])

    p_prior = 2 * env.curriculum_error_rate / 3  # the repo's `bp` prior for depolarizing noise
    result = decode_bp(env.code, shots, p_prior=p_prior)
    assert np.array_equal(result["failure"], ~repo_success)

    assert not decode_bp(env.code, shots, p_prior=p_prior, decoder="bp_osd")["failure"].any()
