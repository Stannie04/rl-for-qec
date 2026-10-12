"""Direct enumeration of low-weight X-type logical patterns (information-set sampling).

    python src/train_utils/coset_enumerate.py --code 144_12_12_ldpc --trials 3000

`coset_catalogue.py` proves the minimum weight of every X-type logical coset with one
MILP per translation orbit.  This module takes the other side of that trade: it hunts
for as many distinct low-weight patterns as it can, quickly and without a per-coset
proof.

Method (Prange-style information-set decoding, p <= 2).  The words wanted are the `e`
in `C = ker(H_z)` whose coset `Lz e` is nonzero.  Take a basis `G` of `C`
(`gf2_nullspace`) and a random column permutation `pi`, then reduce `G[:, pi]` to
reduced row echelon form `R`.  In the basis `R` a codeword's coefficients are exactly
its values on the pivot columns, so the codewords of weight `<= max_weight` that `R`
exposes are the single rows (one 1 among the pivots) and the XORs of two rows (two of
them) whose reduced mass is small.  XOR and weight do not care about `pi`, so only the
survivors are mapped back through `pi`.  Each trial thus hands over a handful of
candidate words; they are kept when their coset is nonzero (words in the stabiliser
group, coset 0, are discarded -- e.g. the weight-6 X stabilisers of the gross code)
and their weight is within `--max-weight`.

Trials run in chunks, each chunk printing one line and appending the words found so
far to `results/cosets/<code>/enum.npz` (written atomically), so an interrupted run
resumes with `--resume` (the default).  Every distinct word is stored once (a dict
keyed by the raw bytes), the lightest word found per coset is remembered, and the
words are grouped into translation orbits with `coset_catalogue.partition_orbits`:
the orbit of a word is the orbit of its coset.  The final catalogue lists, per orbit
hit, the lightest pattern found and plots it with `coset_catalogue.plot_lowest`.

Limit: this is a search, not a proof.  Finding a weight-`w` pattern in an orbit proves
that the orbit's true minimum is at most `w` -- an upper bound, which is what the
output catalogue reports.  Finding nothing in an orbit proves nothing: that orbit may
well hold light patterns this sampler never sampled.  A pattern found here is only the
lightest one seen, never a claim of minimum weight; use `coset_catalogue.py` when an
exact minimum is needed.  Like it, this works on the *X-type* side (Z-type cosets are
the mirror image under `A <-> B`).

CLI: ``--code --max-weight --trials --minutes --workers --seed --no-resume --out-dir``.
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:                  # sibling module; importing through
    sys.path.insert(0, str(_HERE))              # the package __init__ pulls in torch
import coset_catalogue as cc

ROOT = Path(__file__).resolve().parents[2]
CHUNK_TRIALS = 200                              # trials per chunk (log line + checkpoint)


# --------------------------------------------------------------- GF(2) tools
def rref(A):
    """Reduced row echelon form of a uint8 GF(2) matrix, plus its pivot columns.

    One numpy XOR per column (the pivot row into every other row that has a 1 in that
    column), ~4 ms per trial on the gross code's 78 x 144 basis.
    """
    A = A.copy() % 2
    rows, cols = A.shape
    piv, r = [], 0
    for c in range(cols):
        nz = np.nonzero(A[r:, c])[0]
        if nz.size == 0:
            continue
        p = nz[0] + r
        if p != r:
            A[[r, p]] = A[[p, r]]
        mask = A[:, c] == 1
        mask[r] = False
        A[mask] ^= A[r]
        piv.append(c)
        r += 1
        if r == rows:
            break
    return A, np.array(piv, int)


_CACHE = {}


def code_data(code_name):
    """``(setup, G)``: coset_catalogue's ``code_setup`` plus ``G = gf2_nullspace(H_z)``.

    Cached per code name, so a pool worker builds it once (``code_setup`` is the
    expensive part of starting a worker).
    """
    if code_name not in _CACHE:
        setup = cc.code_setup(code_name)
        _CACHE[code_name] = (setup, cc.gf2_nullspace(setup.H_z))
    return _CACHE[code_name]


def coset_ids(Lz, weights, E):
    """Coset id of every row of ``E`` (0 = a stabiliser word)."""
    return ((Lz.astype(int) @ E.T.astype(int)) % 2).T @ weights


# ------------------------------------------------------------------- search
def trial(B, pi, max_weight):
    """One Prange trial -> candidate words, ``(candidates, n)`` uint8 in qubit order.

    ``B[:, pi]`` is reduced; a word's coefficients in that basis are its values on the
    pivot columns, so the candidates are every single row and every XOR of two rows
    (all pairs, vectorised) whose reduced form is no heavier than ``max_weight``.  Only
    those survivors are mapped back through ``pi`` (module convention: ``q -> pi[q]``).
    """
    R, _ = rref(B[:, pi])
    keep = [R[R.sum(1) <= max_weight]]
    for i in range(R.shape[0] - 1):
        X = R[i] ^ R[i + 1:]                    # upper triangle: the pairs (i, j > i)
        sel = X.sum(1) <= max_weight
        if sel.any():
            keep.append(X[sel])
    C = np.concatenate(keep)
    E = np.zeros((C.shape[0], B.shape[1]), np.uint8)
    E[:, pi] = C
    return E


def enumerate_chunk(code_name, max_weight, trials, seed):
    """``trials`` Prange trials in this process -> ``(words, cosets)``.

    Module level with picklable arguments, so it can run in a ``ProcessPoolExecutor``
    (each chunk gets its own RNG seed from ``np.random.SeedSequence``).  Words are
    deduplicated here already; the parent merges the chunks.
    """
    setup, B = code_data(code_name)
    rng = np.random.default_rng(seed)
    found = {}
    for _ in range(trials):
        E = trial(B, rng.permutation(setup.n), max_weight)
        for e, c in zip(E, coset_ids(setup.Lz, setup.weights, E)):
            if c:                                   # stabiliser words are not patterns
                found.setdefault(e.tobytes(), (e, int(c)))
    words = np.array([v[0] for v in found.values()], np.uint8).reshape(-1, setup.n)
    return words, np.array([v[1] for v in found.values()], int)


def resume_from(path):
    """``(words, trials done)`` of a checkpoint, or empty state if there is none."""
    if not Path(path).exists():
        return None
    with np.load(path) as data:
        return data["words"].astype(np.uint8), int(data["trials"])


def save(path, words, trials):
    """Atomic checkpoint: write ``enum.tmp.npz`` next to ``path``, then rename."""
    tmp = Path(path).with_suffix(".tmp.npz")
    np.savez(tmp, words=words, trials=trials)
    os.replace(tmp, path)


def enumerate_cosets(code_name="144_12_12_ldpc", max_weight=14, trials=20000, minutes=None,
                     workers=2, seed=0, resume=True, out_dir=None, chunk=CHUNK_TRIALS):
    """Sample words of ``ker(H_z)`` with a nonzero coset until the budget runs out.

    Stops after ``trials`` trials or ``minutes`` wall-clock minutes, whichever comes
    first, checkpointing ``enum.npz`` after every chunk of ``chunk`` trials (split over
    ``workers`` processes).  Chunk ``i`` gets ``np.random.SeedSequence(seed).spawn(n)[i]``
    and each worker a child of that, so the trials of a chunk are reproducible and a
    resumed run continues with the trials a fresh run would have made next (at worst it
    replays the tail of the last, partially filled chunk; words are deduplicated).
    Only the files in ``out_dir`` (default ``results/cosets/<code>``) are written:
    ``enum.npz``, ``enum_catalogue.csv`` and ``enum_lowest.png``.  Returns a namespace
    with the words (``(N, n)`` uint8), their ``cosets``/``weights``, ``orbit_of`` and
    ``orbits`` from ``partition_orbits``, the lightest word found per coset
    (``coset_words``), per-orbit counters (``orbit_best``, ``orbit_words``,
    ``orbit_trial`` = trials done when that orbit was first hit) and ``out_dir``.

    Each weight returned is the weight of the lightest word *this search found* in that
    orbit: an upper bound on the true minimum, never a claim of minimum weight.
    """
    setup, _ = code_data(code_name)
    l, m, n, k = setup.l, setup.m, setup.n, setup.k
    orbit_of, orbits = cc.partition_orbits(setup)
    out = Path(out_dir) if out_dir else ROOT / "results" / "cosets" / code_name
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "enum.npz"

    state = resume_from(ckpt) if resume else None
    words, cos = np.zeros((0, n), np.uint8), np.zeros(0, int)
    seen = set()                                            # word bytes -> stored once
    by_weight, orbit_words, orbit_trial = Counter(), Counter(), {}
    coset_words, orbit_best = {}, {}                        # id -> (weight, row index)

    def absorb(new_words, new_cos, at_trial):
        """Merge freshly found words into the state; return (added, new orbits)."""
        nonlocal words, cos
        keep = np.array([j for j, e in enumerate(new_words)
                         if e.tobytes() not in seen], int)
        if not keep.size:
            return 0, 0
        base, W, C = len(words), new_words[keep], new_cos[keep]
        seen.update(e.tobytes() for e in W)
        words, cos = np.concatenate([words, W]), np.concatenate([cos, C])
        new_orbits = 0
        for j, (e, c) in enumerate(zip(W, C)):
            w, g, o = int(e.sum()), base + j, int(orbit_of[c])
            by_weight[w] += 1
            if w < coset_words.get(c, (np.inf, 0))[0]:
                coset_words[c] = (w, g)
            orbit_words[o] += 1
            if w < orbit_best.get(o, (np.inf, 0))[0]:
                if o not in orbit_best:
                    new_orbits += 1
                    orbit_trial[o] = at_trial
                orbit_best[o] = (w, g)
        return len(keep), new_orbits

    done = 0
    if state and state[0].shape[1] == n:                    # continue an earlier run
        done = state[1]
        added = absorb(state[0], coset_ids(setup.Lz, setup.weights, state[0]), done)[0]
        print(f"resumed {added} words / {done} trials from {ckpt}: "
              f"{len(orbit_best)}/{len(orbits)} orbits", flush=True)

    seqs = np.random.SeedSequence(seed).spawn(int(np.ceil(trials / chunk)))
    deadline = None if not minutes else time.time() + 60 * minutes
    pool = (ProcessPoolExecutor(max_workers=workers, initializer=code_data,
                                initargs=(code_name,)) if workers > 1 else None)

    def search(args):
        """Run one chunk's worker tasks, in the pool or (workers == 1) right here."""
        return ([j.result() for j in (pool.submit(enumerate_chunk, *a) for a in args)]
                if pool else [enumerate_chunk(*a) for a in args])

    t0 = time.time()
    while done < trials and (deadline is None or time.time() < deadline):
        ci = done // chunk
        budget = min(chunk, trials - done)
        args = [(code_name, max_weight, c, int(s.generate_state(1)[0]))
                for c, s in zip([budget // workers + (j < budget % workers)
                                 for j in range(workers)], seqs[ci].spawn(workers)) if c]
        found = search(args)
        done += budget
        new_orbits = 0
        for W, C in found:
            new_orbits += absorb(W, C, done)[1]
        print(f"chunk {ci + 1:3d}  trials {done}/{trials}  words {len(words)} "
              f"({', '.join(f'w{w}:{c}' for w, c in sorted(by_weight.items())) or 'none'})  "
              f"cosets {len(coset_words)}/{2 ** k - 1}  orbits {len(orbit_best)}/{len(orbits)} "
              f"(+{new_orbits} new)  {time.time() - t0:.1f}s", flush=True)
        save(ckpt, words, done)
    if pool:                                                # workers are done with it
        pool.shutdown()

    # One row per orbit hit; the weight is only an upper bound on its true minimum.
    rows = sorted(((w, int(np.count_nonzero(orbit_of == o)), cc.compact_shift(words[g], l, m), o)
                   for o, (w, g) in orbit_best.items()), key=lambda r: (r[0], -r[1]))
    with open(out / "enum_catalogue.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["orbit", "weight", "n_cosets", "n_words", "support"])
        for w, size, rep, o in rows:
            wr.writerow([o, w, size, orbit_words[o], " ".join(map(str, np.nonzero(rep)[0]))])
    cc.plot_lowest(l, m, [(w, size, rep) for w, size, rep, _ in rows],
                   out / "enum_lowest.png", max_weight)
    print(f"{done} trials, {len(words)} distinct words, {len(coset_words)} cosets, "
          f"{len(orbit_best)}/{len(orbits)} orbits ({time.time() - t0:.1f}s) -> "
          f"{out / 'enum_catalogue.csv'} + enum_lowest.png", flush=True)
    return SimpleNamespace(code=code_name, l=l, m=m, n=n, k=k, trials=done, words=words,
                           cosets=cos, weights=words.sum(1).astype(int), orbit_of=orbit_of,
                           orbits=orbits, coset_words=coset_words, orbit_best=orbit_best,
                           orbit_words=orbit_words, orbit_trial=orbit_trial, out_dir=out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--code", default="144_12_12_ldpc")
    ap.add_argument("--max-weight", type=int, default=14)
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--minutes", type=float, default=None,
                    help="wall-clock budget in minutes (default: no limit)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true",
                    help="start over instead of continuing an existing enum.npz")
    ap.add_argument("--out-dir", default=None, help="default: results/cosets/<code>")
    args = ap.parse_args()
    enumerate_cosets(args.code, args.max_weight, args.trials, args.minutes, args.workers,
                     args.seed, not args.no_resume, args.out_dir)


if __name__ == "__main__":
    main()
