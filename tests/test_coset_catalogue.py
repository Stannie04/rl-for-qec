"""Oracle tests for src/train_utils/coset_catalogue.py.

The module is loaded by file path: ``import src.train_utils.coset_catalogue``
pulls in the package ``__init__`` (torch/galois), which these tests do not need.
Plotting (``plot_lowest``) is deliberately not tested.
"""

import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "train_utils" / "coset_catalogue.py"
CODE_NAME = "18_4_4_ldpc"


@pytest.fixture(scope="module")
def cc():
    spec = importlib.util.spec_from_file_location("coset_catalogue_under_test", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(open(ROOT / "configs" / "code_config.yml"))[CODE_NAME]


@pytest.fixture(scope="module")
def code(cc, cfg):
    """(l, m, H_x, H_z) of 18_4_4_ldpc."""
    return cc.build_code(cfg)


@pytest.fixture(scope="module")
def catalogue(cc):
    """(l, m, k, done) for 18_4_4_ldpc at max_weight=4."""
    return cc.build_catalogue(CODE_NAME, max_weight=4)


def _rank(cc, M):
    return cc.gf2_rref(M)[0].shape[0]


def test_build_code_shapes_and_commute(cc, cfg):
    l, m, H_x, H_z = cc.build_code(cfg)
    assert (l, m) == (3, 3)
    assert H_x.shape == (9, 18) and H_z.shape == (9, 18)
    assert not ((H_x.astype(int) @ H_z.T.astype(int)) % 2).any()


def test_gf2_inv(cc):
    rng = np.random.default_rng(0)
    k = 5
    # Product of random elementary ops -> random invertible GF(2) matrix.
    M = np.eye(k, dtype=np.uint8)
    for _ in range(20):
        i, j = rng.integers(0, k, 2)
        if i == j:
            continue
        if rng.integers(0, 2):                  # swap rows (RHS is a copy)
            M[[i, j]] = M[[j, i]]
        else:                                   # row i ^= row j
            M[i] ^= M[j]
    assert _rank(cc, M) == k                    # invertible by construction
    assert np.array_equal((M.astype(int) @ cc.gf2_inv(M).astype(int)) % 2,
                          np.eye(k, dtype=int))


def test_translation_perms_structure(cc):
    perms = cc.translation_perms(3, 3)
    assert len(perms) == 9
    for p in perms:
        assert sorted(p) == list(range(18))           # a permutation of range(18)
        assert sorted(p[:9]) == list(range(9))        # L block -> L block
        assert sorted(p[9:] - 9) == list(range(9))    # R block -> R block


def test_translation_perms_preserve_H_z(cc, code):
    l, m, H_x, H_z = code
    for p in cc.translation_perms(l, m):
        # Column permutation H_z[:, p] leaves the row space of H_z invariant.
        assert _rank(cc, np.vstack([H_z, H_z[:, p]])) == _rank(cc, H_z)
        for e in H_x:                           # rows of H_x are in null(H_z)
            if not e.any():
                continue
            assert not (H_z.astype(int) @ e % 2).any()
            ep = np.zeros_like(e)
            ep[p] = e                           # module convention: q -> p[q]
            assert not (H_z.astype(int) @ ep % 2).any()


def test_catalogue_max_weight_4(cc, code, catalogue):
    l, m, k, done = catalogue
    assert (l, m, k) == (3, 3, 4)
    assert len(done) == 2 ** k - 1 == 15
    assert sorted(done) == list(range(1, 2 ** k))

    H_z = code[3]
    for c, (rep, w, orbit, status) in done.items():
        assert w == 4, c
        assert status == "optimal", c
        assert rep is not None and rep.sum() == 4
        assert not (H_z.astype(int) @ rep.astype(int) % 2).any()

    orbits = Counter(o for _, _, o, _ in done.values())
    assert len(orbits) == 5                     # 5 translation orbits
    assert set(orbits.values()) == {3}          # each of size 3


def test_catalogue_max_weight_3(cc):
    l, m, k, done = cc.build_catalogue(CODE_NAME, max_weight=3)
    assert (l, m, k) == (3, 3, 4)
    assert len(done) == 15
    for c, (rep, w, orbit, status) in done.items():
        assert rep is None, c
        assert w == -1, c                       # -1: proven above the cap
        assert status == "above", c


def test_compact_shift_keeps_weight_and_blocks(cc, catalogue):
    l, m, k, done = catalogue
    assert [w for _, w, _, _ in done.values()] == [4] * 15
    for rep, _, _, _ in done.values():
        out = cc.compact_shift(rep, l, m)
        assert out.sum() == rep.sum()
        assert out[:l * m].sum() == rep[:l * m].sum()
        assert out[l * m:].sum() == rep[l * m:].sum()


# -------------------------------------------------------------- code setup
def _signature(done):
    """done with the rep arrays made comparable (tuple instead of ndarray)."""
    return {c: (None if rep is None else tuple(rep.tolist()), w, o, st)
            for c, (rep, w, o, st) in done.items()}


def test_code_setup_and_partition_orbits(cc, code, catalogue):
    l, m, H_x, H_z = code
    setup = cc.code_setup(CODE_NAME)
    assert (setup.l, setup.m, setup.k, setup.n) == (3, 3, 4, 18)
    assert np.array_equal(setup.H_x, H_x) and np.array_equal(setup.H_z, H_z)
    assert len(setup.perms) == 9 and list(setup.weights) == [1, 2, 4, 8]

    orbit_of, orbits = cc.partition_orbits(setup)
    assert [c for c, _ in orbits] == [1, 2, 3, 4, 5]        # smallest coset of each orbit
    assert orbit_of.shape == (16,) and orbit_of[0] == -1    # trivial coset: no orbit
    assert (orbit_of[1:] >= 0).all()                        # covers every coset 1..15
    assert [int(np.count_nonzero(orbit_of == i)) for i in range(5)] == [3] * 5
    for i, (c, s) in enumerate(orbits):
        assert np.array_equal(s, [(c >> j) & 1 for j in range(setup.k)])
        assert int(np.nonzero(orbit_of == i)[0][0]) == c    # numbered by smallest id
        e0 = (s @ setup.Lx_dual % 2).astype(np.uint8)
        assert setup.coset(e0) == c                         # coset/shifted are wired up
        assert all(orbit_of[setup.coset(setup.shifted(e0, p))] == i for p in setup.perms)

    # build_catalogue numbers its orbits exactly like this (same ids, same order).
    assert {c: o for c, (_, _, o, _) in catalogue[3].items()} == dict(enumerate(orbit_of[1:], 1))


def test_checkpoint_round_trip(cc, tmp_path, monkeypatch):
    path = tmp_path / "c.jsonl"
    first = cc.build_catalogue(CODE_NAME, 4, 60, workers=1, checkpoint=path)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [rec["orbit"] for rec in lines] == [0, 1, 2, 3, 4]   # one line per orbit
    assert {rec["status"] for rec in lines} == {"optimal"}
    for rec in lines:
        assert tuple(rec["rep"]) == tuple(sorted(rec["rep"]))   # rep = support indices
        assert rec["weight"] == len(rec["rep"])

    monkeypatch.setattr(cc, "min_weight_rep",           # no MILP on the second run
                        lambda *a, **kw: pytest.fail("orbit not restored from checkpoint"))
    second = cc.build_catalogue(CODE_NAME, 4, 60, workers=1, checkpoint=path)

    assert first[:3] == second[:3]
    assert _signature(first[3]) == _signature(second[3])
    assert len(path.read_text().splitlines()) == 5       # restored lines are not re-appended


def test_checkpoint_round_trip_above_cap(cc, tmp_path, monkeypatch):
    """'above' records (no representative) resume without a MILP either."""
    path = tmp_path / "above.jsonl"
    first = cc.build_catalogue(CODE_NAME, 3, 60, workers=1, checkpoint=path)
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    assert {rec["status"] for rec in recs} == {"above"}
    assert all(rec["rep"] is None and rec["weight"] is None for rec in recs)

    monkeypatch.setattr(cc, "min_weight_rep",
                        lambda *a, **kw: pytest.fail("orbit not restored from checkpoint"))
    second = cc.build_catalogue(CODE_NAME, 3, 60, workers=1, checkpoint=path)
    assert _signature(first[3]) == _signature(second[3])
