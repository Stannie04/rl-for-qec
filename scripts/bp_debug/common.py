"""Shared helpers for the BP debug investigation (issue #13).

The binding contract is ``scripts/bp_debug/INTERFACES.md``; every function below
implements one of its entries.  Run scripts from the repository root as
``python -m scripts.bp_debug.<module>``.
"""

import numpy as np
import torch

from src.environment import QECEnv
from src.read_config import ConfigParser

PAULIS = ("X", "Y", "Z")
DECODERS = ("bp", "bp_osd", "bp_lsd")

# PROBE: the contract assumes `bp` is a key of model_config.yml, but that file only has
# `sac`, `moe` and `encoder`.  The repo builds every config from a model-config key
# (`main.py` defaults to `sac`) and passes the agent name separately to
# `get_agent_and_inference`; the classic decoders need no model hyperparameters, so they
# map onto the default `sac` entry.  `config.agent_name` is left as the key actually
# used, so the config can still be rebuilt the way `evaluation.py` does.
MODEL_CONFIG_KEYS = {"bp": "sac", "bp_osd": "sac", "bp_lsd": "sac", "mwpm": "sac"}

# Logical bases are cached per parity check matrix pair: computing them is a GF(2)
# row reduction over the dense H, and `independent_logical_failure` runs per shot.
_LOGICAL_BASES: dict = {}


def load_config(code_name, agent_name="bp", noise_model="depolarizing"):
    """`ConfigParser` for `code_name` with the repo's `bp` agent settings."""
    config = ConfigParser("configs", MODEL_CONFIG_KEYS.get(agent_name, agent_name), code_name, None)
    config.update({"noise_model": noise_model})
    return config


def make_env(config, shots=None):
    """`QECEnv` for `config`; with `shots` they are served in order by `env.reset`."""
    return QECEnv(config, shots=shots)


def _seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def uniform_shots(config, n_per_weight, max_weight=5, seed=0):
    """`create_dataset_from_uniform_shots`, seeded.

    Weight w means w X errors AND w Z errors at independent random positions, so up
    to 2w qubits are touched per shot.
    """
    from src.train_utils.datasets import create_dataset_from_uniform_shots  # pulls in the agent stack

    _seed(seed)
    return create_dataset_from_uniform_shots(config, num_samples_per_error=n_per_weight, max_error=max_weight)


def random_shots(config, n, p, seed=0):
    """`create_dataset_from_random_shots` (iid depolarizing at physical rate p), seeded."""
    from src.train_utils.datasets import create_dataset_from_random_shots  # pulls in the agent stack

    _seed(seed)
    return create_dataset_from_random_shots(config, num_samples=n, error_rate=p)


def single_qubit_shots(config, pauli):
    """One shot per qubit: a single `pauli` in {"X", "Y", "Z"} on that qubit. Shape (n, 2, n)."""
    if pauli not in PAULIS:
        raise ValueError(f"pauli must be one of {PAULIS}, got {pauli!r}")

    shots = np.zeros((config.n, 2, config.n), dtype=np.int8)
    qubits = np.arange(config.n)
    if pauli in ("X", "Y"):
        shots[qubits, 0, qubits] = 1
    if pauli in ("Z", "Y"):
        shots[qubits, 1, qubits] = 1
    return shots


def _as_bits(array):
    """0/1 uint8 copy of an error pattern or parity check matrix (numpy, torch or list)."""
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(array).astype(np.uint8) % 2)


def _rref(matrix):
    """Reduced row echelon form over GF(2); returns (non-zero rows, pivot columns)."""
    rows = _as_bits(matrix).copy()
    pivots = []
    for column in range(rows.shape[1]):
        row = len(pivots)
        if row == rows.shape[0]:
            break
        candidates = np.flatnonzero(rows[row:, column])
        if candidates.size == 0:
            continue
        pivot = row + candidates[0]
        rows[[row, pivot]] = rows[[pivot, row]]
        others = np.flatnonzero(rows[:, column])
        rows[others[others != row]] ^= rows[row]
        pivots.append(column)
    return rows[: len(pivots)], pivots


def _rank(matrix):
    matrix = np.asarray(matrix, dtype=np.uint8)
    return 0 if matrix.size == 0 else _rref(matrix)[0].shape[0]


def _null_space(matrix):
    """Basis (one per row) of {v : matrix @ v == 0} over GF(2)."""
    matrix = _as_bits(matrix)
    reduced, pivots = _rref(matrix)
    free = [column for column in range(matrix.shape[1]) if column not in pivots]
    basis = np.zeros((len(free), matrix.shape[1]), dtype=np.uint8)
    for i, column in enumerate(free):
        basis[i, column] = 1
        for row, pivot in enumerate(pivots):
            basis[i, pivot] = reduced[row, column]
    return basis


def _quotient_basis(null_basis, row_space):
    """Rows of `null_basis` that extend `row_space`: a basis of null_basis/row_space."""
    width = null_basis.shape[1]
    current = _as_bits(row_space).reshape(-1, width)
    rank = _rank(current)
    logicals = []
    for vector in null_basis:
        extended = np.vstack([current, vector])
        if _rank(extended) > rank:
            current, rank = extended, rank + 1
            logicals.append(vector)
    return np.array(logicals, dtype=np.uint8).reshape(-1, width)


def _logical_bases(code):
    """(logical X, logical Z) bases of ker(H_z)/rowspace(H_x) and ker(H_x)/rowspace(H_z)."""
    H_x, H_z = _as_bits(code.H.x), _as_bits(code.H.z)
    key = (H_x.shape, H_z.shape, H_x.tobytes(), H_z.tobytes())
    if key not in _LOGICAL_BASES:
        _LOGICAL_BASES[key] = (
            _quotient_basis(_null_space(H_z), H_x),
            _quotient_basis(_null_space(H_x), H_z),
        )
    return _LOGICAL_BASES[key]


def syndromes(code, shots):
    """(x_syndrome (N, m_x), z_syndrome (N, m_z)) uint8, from `code.H`, not `env.step`."""
    H_x, H_z = _as_bits(code.H.x).astype(np.int64), _as_bits(code.H.z).astype(np.int64)
    shots = _as_bits(shots)
    x_syndrome = (shots[:, 1] @ H_x.T) % 2  # X checks read out the Z errors
    z_syndrome = (shots[:, 0] @ H_z.T) % 2  # Z checks read out the X errors
    return x_syndrome.astype(np.uint8), z_syndrome.astype(np.uint8)


def independent_logical_failure(code, shots, x_pred, z_pred):
    """Bool (N,): the residual has a non-zero syndrome, or anticommutes with a logical."""
    shots = _as_bits(shots)
    residual = np.stack([shots[:, 0] ^ _as_bits(x_pred), shots[:, 1] ^ _as_bits(z_pred)], axis=1)
    x_syndrome, z_syndrome = syndromes(code, residual)
    logical_x, logical_z = _logical_bases(code)
    logical_x_failure = (residual[:, 0].astype(np.int64) @ logical_z.astype(np.int64).T) % 2
    logical_z_failure = (residual[:, 1].astype(np.int64) @ logical_x.astype(np.int64).T) % 2
    return np.asarray(
        x_syndrome.any(axis=1)
        | z_syndrome.any(axis=1)
        | logical_x_failure.any(axis=1)
        | logical_z_failure.any(axis=1),
        dtype=bool,
    )


def _make_decoder(kind, H, p_prior, max_iter, bp_method, ms_scaling_factor, schedule):
    import ldpc

    decoder_cls = {"bp": ldpc.BpDecoder, "bp_osd": ldpc.BpOsdDecoder, "bp_lsd": ldpc.BpLsdDecoder}[kind]
    return decoder_cls(
        H,
        error_rate=p_prior,
        max_iter=max_iter,
        bp_method=bp_method,
        ms_scaling_factor=ms_scaling_factor,
        schedule=schedule,
    )


def _converged(decoder, kind):
    """`decoder.converge`; OSD/LSD always fall back to post-processing, which counts as done."""
    return True if kind != "bp" else bool(decoder.converge)


def decode_bp(code, shots, *, p_prior, max_iter=30, bp_method="product_sum",
              ms_scaling_factor=1.0, schedule="parallel", decoder="bp"):
    """Decode both error planes independently with `ldpc` BP(/OSD/LSD).

    `p_prior` is the per-qubit error rate handed to `ldpc` unchanged; the caller owns
    the 2p/3 convention.  Returns x_pred, z_pred (N, n) uint8, converged_x,
    converged_z (N,) bool and failure (N,) bool.
    """
    if decoder not in DECODERS:
        raise ValueError(f"decoder must be one of {DECODERS}, got {decoder!r}")

    H_x, H_z = _as_bits(code.H.x), _as_bits(code.H.z)
    x_syndrome, z_syndrome = syndromes(code, shots)
    num_shots, n = x_syndrome.shape[0], H_x.shape[1]

    args = (p_prior, max_iter, bp_method, ms_scaling_factor, schedule)
    x_decoder = _make_decoder(decoder, H_z, *args)
    z_decoder = _make_decoder(decoder, H_x, *args)

    x_pred = np.zeros((num_shots, n), dtype=np.uint8)
    z_pred = np.zeros((num_shots, n), dtype=np.uint8)
    converged_x = np.zeros(num_shots, dtype=bool)
    converged_z = np.zeros(num_shots, dtype=bool)
    for i in range(num_shots):
        x_pred[i] = x_decoder.decode(z_syndrome[i])
        z_pred[i] = z_decoder.decode(x_syndrome[i])
        converged_x[i] = _converged(x_decoder, decoder)
        converged_z[i] = _converged(z_decoder, decoder)

    return {
        "x_pred": x_pred,
        "z_pred": z_pred,
        "converged_x": converged_x,
        "converged_z": converged_z,
        "failure": independent_logical_failure(code, shots, x_pred, z_pred),
    }


def weights(shots):
    """Qubits touched per shot (X|Z support); the nominal weight label is not recoverable."""
    return _as_bits(shots).any(axis=1).sum(axis=1)
