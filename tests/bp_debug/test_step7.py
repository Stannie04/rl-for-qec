"""Contract tests for ``scripts/bp_debug/step7_stratified.py`` (card BD8, issue #13).

Run from the repository root::

    python -m pytest tests/bp_debug/test_step7.py
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

from scripts.bp_debug.step7_stratified import (  # noqa: E402
    BIN_ORDER,
    DEFAULT_CODES,
    DEFAULT_N_PER_P,
    FINE_BINS,
    FINE_P,
    OVERFLOW_BIN,
    P_VALUES,
    QUICK_N_PER_P,
    bin_labels,
    flip_counts,
    main,
    parse_args,
    p_key,
    reconstructed_ler,
    resolve_n_per_p,
    summarise_bins,
)

ALL_P_KEYS = ["0.001", "0.002", "0.003", "0.004", "0.005"]


def test_bin_order_and_p_keys():
    assert BIN_ORDER[-1] == OVERFLOW_BIN
    assert len(BIN_ORDER) == 16  # every (n_x, n_z) with n_x + n_z <= 4, plus the "5+" bin
    assert set(BIN_ORDER) == {f"({a},{b})" for a in range(5) for b in range(5 - a)} | {OVERFLOW_BIN}
    assert [p_key(p) for p in P_VALUES] == ALL_P_KEYS
    assert p_key(FINE_P) == "0.001"
    assert set(FINE_BINS) <= set(BIN_ORDER)


def test_default_and_quick_sizes():
    assert resolve_n_per_p(parse_args([])) == DEFAULT_N_PER_P
    assert resolve_n_per_p(parse_args(["--quick"])) == QUICK_N_PER_P
    assert resolve_n_per_p(parse_args(["--quick", "--n-per-weight", "7"])) == 7
    assert list(parse_args([]).codes) == list(DEFAULT_CODES)
    assert parse_args(["--codes", "d5_surface"]).codes == ["d5_surface"]


def test_flip_counts_and_bin_labels():
    n = 6
    shots = np.zeros((6, 2, n), dtype=np.int8)
    shots[1, 0, 0] = 1  # X
    shots[2, 1, 1] = 1  # Z
    shots[3, 0, 2] = shots[3, 1, 2] = 1  # Y: one flip in both planes
    shots[4, 0, 0:3] = 1
    shots[4, 1, 3:6] = 1  # (3,3) -> overflow
    shots[5, 0, 0:4] = 1  # (4,0)

    n_x, n_z = flip_counts(shots)
    assert list(n_x) == [0, 1, 0, 1, 3, 4]
    assert list(n_z) == [0, 0, 1, 1, 3, 0]

    labels = bin_labels(n_x, n_z)
    assert list(labels) == ["(0,0)", "(1,0)", "(0,1)", "(1,1)", OVERFLOW_BIN, "(4,0)"]
    assert set(labels) <= set(BIN_ORDER)


def test_summarise_bins_partitions_all_shots():
    labels = np.array(["(0,0)", "(1,0)", "(1,0)", "(1,0)", OVERFLOW_BIN, "(2,1)"])
    failure = np.array([False, True, False, True, True, False])

    bins = summarise_bins(labels, failure)
    assert list(bins) == list(BIN_ORDER)
    assert sum(bins[label]["shots"] for label in BIN_ORDER) == len(labels)
    assert sum(bins[label]["failures"] for label in BIN_ORDER) == int(failure.sum())
    assert sum(bins[label]["share"] for label in BIN_ORDER) == pytest.approx(1.0)

    assert bins["(1,0)"]["shots"] == 3
    assert bins["(1,0)"]["share"] == pytest.approx(0.5)
    assert bins["(1,0)"]["failures"] == 2
    assert bins["(1,0)"]["failure_fraction"] == pytest.approx(2 / 3)
    assert bins["(0,0)"]["failure_fraction"] == 0.0
    assert bins["(3,0)"]["failure_fraction"] is None  # empty bin
    assert bins["(3,0)"]["shots"] == 0

    assert reconstructed_ler(bins) == pytest.approx(float(failure.mean()))


def test_main_quick_run_writes_json_and_markdown(tmp_path):
    result = main([
        "--quick",
        "--codes", "d3_surface",
        "--n-per-weight", "30",
        "--seed", "0",
        "--out-dir", str(tmp_path),
    ])

    assert result["module"] == "step7_stratified"
    assert result["n_per_p"] == 30
    assert result["quick"] is True
    assert list(result["codes"]) == ["d3_surface"]

    json_path = tmp_path / "step7_stratified.json"
    md_path = tmp_path / "step7_stratified.md"
    assert json_path.exists() and md_path.exists()

    payload = json.loads(json_path.read_text())
    assert payload == result  # no numpy scalars leaked into the payload

    code = payload["codes"]["d3_surface"]
    assert code["n"] == 9
    assert code["p_prior"] == pytest.approx(2 * code["curriculum_error_rate"] / 3)

    for p in ALL_P_KEYS:
        stats = code["per_p"][p]
        assert stats["n_shots"] == 30
        assert set(stats["bins"]) == set(BIN_ORDER)
        assert sum(b["shots"] for b in stats["bins"].values()) == 30
        assert sum(b["failures"] for b in stats["bins"].values()) == stats["failures"]
        assert stats["reconstructed_ler"] == pytest.approx(stats["direct_ler"], abs=1e-12)

    markdown = md_path.read_text()
    assert "## Observation" in markdown
    assert "## Failure fraction by `(n_x, n_z)` at p=0.001" in markdown
    assert "reconstructed LER" in markdown
