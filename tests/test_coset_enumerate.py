"""Oracle tests for src/train_utils/coset_enumerate.py.

The module is imported as ``coset_enumerate`` after adding ``src/train_utils`` to
``sys.path``: ``import src.train_utils.coset_enumerate`` would execute the package
``__init__`` (torch/galois), and the module itself imports its sibling
``coset_catalogue`` the same way.  Every output goes to ``tmp_path``, never to
``results/``.
"""

import csv
import importlib
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "train_utils"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
ce = importlib.import_module("coset_enumerate")

SMALL = "18_4_4_ldpc"
GROSS = "144_12_12_ldpc"


def _weight_four_vectors(n):
    """All C(n, 4) weight-4 vectors of F_2^n as one (C(n, 4), n) uint8 array."""
    comb = np.array(list(itertools.combinations(range(n), 4)), int)
    E = np.zeros((len(comb), n), np.uint8)
    E[np.arange(len(comb))[:, None], comb] = 1
    return E


@pytest.fixture(scope="module")
def small():
    """(setup, G) of 18_4_4_ldpc."""
    return ce.code_data(SMALL)


@pytest.fixture(scope="module")
def found(tmp_path_factory):
    """The 300-trial enumeration of 18_4_4_ldpc at max_weight 4, seed 0."""
    return ce.enumerate_cosets(SMALL, max_weight=4, trials=300, workers=1, seed=0,
                               out_dir=tmp_path_factory.mktemp("enum_small"))


# --------------------------------------------------------------- brute force
def test_rref_matches_reference(small):
    """The vectorised GF(2) elimination agrees with coset_catalogue.gf2_rref."""
    rng = np.random.default_rng(0)
    _, G = small
    for _ in range(5):
        A = rng.integers(0, 2, G.shape).astype(np.uint8)
        R, piv = ce.rref(A)
        R2, piv2 = ce.cc.gf2_rref(A)
        assert np.array_equal(R, R2) and list(piv) == piv2


def test_every_brute_force_orbit_is_found(found, small):
    """Exhaustive weight-4 search: every orbit it finds is found by the sampler too."""
    setup, _ = small
    E = _weight_four_vectors(setup.n)
    assert len(E) == 3060                                   # C(18, 4)
    ids = ce.coset_ids(setup.Lz, setup.weights, E)
    valid = ~(setup.H_z.astype(int) @ E.T.astype(int) % 2).any(0) & (ids != 0)
    brute = {int(found.orbit_of[c]) for c in ids[valid]}
    assert brute                                             # the oracle is not empty
    assert set(found.weights.tolist()) == {4}                # d = 4: nothing lighter
    assert brute <= {int(found.orbit_of[c]) for c in found.cosets}


def test_words_are_valid(found, small):
    """Every returned word: H_z e = 0, nonzero coset, weight within the cap."""
    setup, _ = small
    assert found.words.shape[1] == setup.n and len(found.words) > 0
    assert not (setup.H_z.astype(int) @ found.words.T.astype(int) % 2).any()
    assert (found.cosets != 0).all()
    assert (found.weights <= 4).all()
    assert (found.weights == found.words.sum(1)).all()
    assert [int(c) for c in found.cosets[:5]] == [setup.coset(e) for e in found.words[:5]]
    assert found.orbit_of[0] == -1
    assert (found.orbit_of[found.cosets] >= 0).all()         # every word is in an orbit
    keys = [e.tobytes() for e in found.words]
    assert len(set(keys)) == len(keys)                      # stored once each
    assert len(found.coset_words) == len(set(map(int, found.cosets)))   # lightest per coset
    assert sum(found.orbit_words.values()) == len(keys)


# ------------------------------------------------------------- checkpoints
def test_checkpoint_resume_and_catalogue(tmp_path):
    """Resuming reuses enum.npz; a restart (resume=False) starts over."""
    out = tmp_path / "resume"
    first = ce.enumerate_cosets(SMALL, 4, trials=80, workers=1, seed=1, out_dir=out)
    assert {p.name for p in out.iterdir()} == {"enum.npz", "enum_catalogue.csv",
                                               "enum_lowest.png"}
    assert first.trials == 80 and len(first.words) > 0

    second = ce.enumerate_cosets(SMALL, 4, trials=80, workers=1, seed=1, out_dir=out)
    assert second.trials == first.trials
    assert {e.tobytes() for e in second.words} == {e.tobytes() for e in first.words}
    assert dict(second.orbit_best) == dict(first.orbit_best)

    with open(out / "enum_catalogue.csv") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["orbit", "weight", "n_cosets", "n_words", "support"]
    assert {int(r[0]) for r in rows[1:]} == set(second.orbit_best)
    for r in rows[1:]:
        o = int(r[0])
        weight, index = second.orbit_best[o]
        rep = ce.cc.compact_shift(second.words[index], 3, 3)
        assert int(r[1]) == weight == len(r[4].split())          # weight = support size
        assert r[4].split() == [str(q) for q in np.nonzero(rep)[0]]
        assert int(r[2]) == int((second.orbit_of == o).sum())     # orbit size
        assert int(r[3]) == second.orbit_words[o]                 # words found in orbit

    restarted = ce.enumerate_cosets(SMALL, 4, trials=40, workers=1, seed=2, out_dir=out,
                                    resume=False)
    assert restarted.trials == 40                                # not 80: no resume
    with np.load(out / "enum.npz") as data:
        assert int(data["trials"]) == 40
        assert data["words"].dtype == np.uint8
        assert data["words"].shape == restarted.words.shape == (len(restarted.words), 18)


# ------------------------------------------------------- parallel + gross code
def test_parallel_workers_merge(tmp_path):
    """The ProcessPoolExecutor path is exercised and merges both workers.

    Chunk seeds are fixed by (seed, chunk index) and only the parent merges, so two
    identical runs with --workers 2 give the same set of words.
    """
    runs = [ce.enumerate_cosets(SMALL, 4, trials=40, workers=2, seed=5,
                                out_dir=tmp_path / f"w{n}") for n in range(2)]
    for n, run in enumerate(runs):
        assert run.trials == 40 and len(run.words) > 0
        assert (run.weights <= 4).all() and (run.cosets != 0).all()
        assert (tmp_path / f"w{n}" / "enum_catalogue.csv").exists()
    assert {e.tobytes() for e in runs[0].words} == {e.tobytes() for e in runs[1].words}


@pytest.mark.slow
def test_gross_code_smoke(tmp_path):
    """A few hundred trials on the gross code: distance 12, so every word is >= 12."""
    run = ce.enumerate_cosets(GROSS, max_weight=14, trials=200, workers=1, seed=0,
                              out_dir=tmp_path)
    setup, G = ce.code_data(GROSS)
    assert (setup.n, G.shape[0], len(run.orbits)) == (144, 78, 155)
    assert len(run.words) > 0 and run.trials == 200
    assert (run.weights >= 12).all()                        # no lighter logical exists
    assert (run.weights <= 14).all()
    assert not (setup.H_z.astype(int) @ run.words.T.astype(int) % 2).any()
    assert (run.cosets != 0).all()
    assert (run.orbit_of[run.cosets] >= 0).all()             # every word is in an orbit
