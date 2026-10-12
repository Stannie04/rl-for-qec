"""Catalogue of X-type logical cosets of a bivariate-bicycle (gross) code.

    python -m src.train_utils.coset_catalogue --code 144_12_12_ldpc

A pure-X error e with H_z e = 0 is a logical coset representative; its coset is
the logical syndrome s = Lz e in F2^k (Lz: Z logicals). There are 2^k - 1
non-trivial cosets (4095 for the gross code). For each one we find the exact
minimum weight with a MILP (HiGHS, zero gap). Translations of the l x m torus
are code automorphisms, so one MILP per translation orbit is enough: the
solution is shifted to the other cosets in the orbit.

Writes results/cosets/<code>/{catalogue.npz, catalogue.csv, lowest_weight.png}, plus
checkpoint.jsonl listing each solved orbit; an interrupted run resumes from it
(one MILP per orbit that is not in the file yet).  The Z-type cosets are the
mirror image (A <-> B swap) and are not drawn.
"""

import argparse
import csv
import json
import multiprocessing
import time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- GF(2) tools
def gf2_rref(M):
    M = (M.copy() % 2).astype(np.uint8)
    pivots, r = [], 0
    for c in range(M.shape[1]):
        rows = np.nonzero(M[r:, c])[0]
        if rows.size == 0:
            continue
        p = rows[0] + r
        M[[r, p]] = M[[p, r]]
        for i in np.nonzero(M[:, c])[0]:
            if i != r:
                M[i] ^= M[r]
        pivots.append(c)
        r += 1
        if r == M.shape[0]:
            break
    return M[:r], pivots


def gf2_nullspace(M):
    R, piv = gf2_rref(M)
    free = [c for c in range(M.shape[1]) if c not in piv]
    N = np.zeros((len(free), M.shape[1]), dtype=np.uint8)
    for i, f in enumerate(free):
        N[i, f] = 1
        for r, p in enumerate(piv):
            N[i, p] = R[r, f]
    return N


def quotient_basis(null_space, row_space):
    """Vectors of null_space that are independent of row_space (and each other)."""
    cur, _ = gf2_rref(row_space)
    rank = cur.shape[0]
    out = []
    for v in null_space:
        cand, _ = gf2_rref(np.vstack([cur, v]))
        if cand.shape[0] > rank:
            out.append(v)
            cur, rank = cand, rank + 1
    return np.array(out, dtype=np.uint8)


def gf2_inv(M):
    k = M.shape[0]
    R, piv = gf2_rref(np.hstack([M, np.eye(k, dtype=np.uint8)]))
    assert piv[:k] == list(range(k)), "singular"
    return R[:, k:]


# ------------------------------------------------------------------- the code
def build_code(cfg):
    l, m = cfg["l"], cfg["m"]
    S_l, S_m = np.roll(np.eye(l, dtype=int), 1, 1), np.roll(np.eye(m, dtype=int), 1, 1)
    x, y = np.kron(S_l, np.eye(m, dtype=int)), np.kron(np.eye(l, dtype=int), S_m)
    z = np.kron(S_l, S_m)

    def poly(terms):
        out = np.zeros_like(x)
        for a, b, c in terms:
            out = (out + np.linalg.matrix_power(x, a) @ np.linalg.matrix_power(y, b)
                   @ np.linalg.matrix_power(z, c)) % 2
        return out.astype(np.uint8)

    A, B = poly(cfg["code_params"]["A"]), poly(cfg["code_params"]["B"])
    H_x, H_z = np.hstack([A, B]), np.hstack([B.T, A.T])
    assert not ((H_x.astype(int) @ H_z.T.astype(int)) % 2).any()
    return l, m, H_x, H_z


def translation_perms(l, m):
    """Qubit permutations for the l*m shifts of the torus (L block then R block)."""
    lm, perms = l * m, []
    for da in range(l):
        for db in range(m):
            p = np.empty(2 * lm, dtype=int)
            for a in range(l):
                for b in range(m):
                    j = ((a + da) % l) * m + (b + db) % m
                    p[a * m + b], p[lm + a * m + b] = j, lm + j
            perms.append(p)
    return perms


# ------------------------------------------------------------ min-weight MILP
def min_weight_rep(H_z, Lz, s, max_weight, time_limit=600):
    """min |e| s.t. H_z e = 0, Lz e = s (mod 2), |e| <= max_weight.

    Returns (e, weight, status): status "optimal", "above" (proved > max_weight)
    or "unknown" (time limit, no proof; e is the best found or None)."""
    n = H_z.shape[1]
    C = np.vstack([H_z, Lz]).astype(int)
    rhs = np.concatenate([np.zeros(H_z.shape[0], int), s.astype(int)])
    slack_max = C.sum(1) // 2                      # C e - 2 k = rhs, 0 <= k <= wt/2
    nr = C.shape[0]
    A = csr_matrix(np.hstack([C, -2 * np.eye(nr, dtype=int)]))
    cap = csr_matrix(np.concatenate([np.ones(n), np.zeros(nr)])[None])
    res = milp(
        c=np.concatenate([np.ones(n), np.zeros(nr)]),
        constraints=[LinearConstraint(A, rhs, rhs),
                     LinearConstraint(cap, 0, max_weight)],
        integrality=np.ones(n + nr),
        bounds=Bounds(np.zeros(n + nr), np.concatenate([np.ones(n), slack_max])),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0},
    )
    if res.status == 2:
        return None, None, "above"
    if res.x is None:
        return None, None, "unknown"
    e = np.round(res.x[:n]).astype(np.uint8)
    return e, int(e.sum()), "optimal" if res.status == 0 else "unknown"


# ------------------------------------------------------------------ catalogue
def _solve_orbit(task):
    """One translation orbit -> (orbit_id, e, w, status) for its representative.

    Module-level and handed only picklable arguments, so the pool is safe under the
    'spawn' start method; capped to one HiGHS thread to avoid oversubscription."""
    orbit_id, H_z, Lz, s, max_weight, time_limit = task
    _highs_one_thread()
    e, w, status = min_weight_rep(H_z, Lz, s, max_weight, time_limit)
    return orbit_id, e, w, status


def _highs_one_thread():
    """Use a single HiGHS thread in this process, for pool workers only (scipy
    hands unknown ``milp`` options to HiGHS verbatim, so ``options={"threads": 1}``
    works).  ``min_weight_rep`` owns its options dict and is not ours to edit, so
    the module-level ``milp`` binding is wrapped instead.  Idempotent; a no-op
    outside a pool child, where there is nothing to oversubscribe."""
    global milp
    if getattr(milp, "one_thread", False) or multiprocessing.parent_process() is None:
        return
    inner = milp

    def capped(*args, **kwargs):
        kwargs["options"] = {**(kwargs.get("options") or {}), "threads": 1}
        return inner(*args, **kwargs)

    capped.one_thread = True
    milp = capped


def code_setup(code_name):
    """Code and coset tools for ``code_name`` (see ``configs/code_config.yml``).

    Fields: ``l``, ``m``, ``k``, ``n = 2*l*m``, the check matrices ``H_x``/``H_z``,
    ``Lz`` (Z logicals) and ``Lx_dual`` (X logicals dual to them, so every logical
    syndrome has a representative), ``perms`` (torus translations) and ``weights``
    (bit values of a syndrome).  Callables: ``coset(e)`` -> int coset id of the
    pure-X, H_z-commuting error ``e``, and ``shifted(e, p)`` -> ``e`` under the
    qubit permutation ``p``.
    """
    cfg = yaml.safe_load(open(ROOT / "configs" / "code_config.yml"))[code_name]
    l, m, H_x, H_z = build_code(cfg)
    Lz = quotient_basis(gf2_nullspace(H_x), H_z)
    k = Lz.shape[0]
    assert k == cfg["k"], (k, cfg["k"])
    # X logicals dual to Lz: Lz @ Lx_dual[i] = unit vector i, so any coset has a rep.
    Lx = quotient_basis(gf2_nullspace(H_z), H_x)
    Lx_dual = (gf2_inv((Lz.astype(int) @ Lx.T.astype(int) % 2).astype(np.uint8)).T.astype(int)
               @ Lx.astype(int) % 2).astype(np.uint8)   # M^-T Lx, M = Lz Lx^T
    weights = 1 << np.arange(k)

    def coset(e):
        return int(((Lz.astype(int) @ e.astype(int)) % 2) @ weights)

    def shifted(e, p):
        out = np.zeros_like(e)
        out[p] = e                                  # qubit q moves to p[q]
        return out

    return types.SimpleNamespace(l=l, m=m, k=k, n=2 * l * m, H_x=H_x, H_z=H_z,
                                 Lz=Lz, Lx_dual=Lx_dual, perms=translation_perms(l, m),
                                 weights=weights, coset=coset, shifted=shifted)


def partition_orbits(setup):
    """(orbit_of, orbits) of the 2^k - 1 non-trivial cosets under ``setup.perms``.

    ``orbit_of[c]`` is the orbit id of coset ``c`` (``orbit_of[0] = -1``: the trivial
    coset is in no orbit), so orbit ``i`` has size ``np.count_nonzero(orbit_of == i)``.
    ``orbits[i]`` is ``(smallest coset id of the orbit, its syndrome array)``, ids
    numbered by increasing smallest coset id.

    Phase 1 of the catalogue, no MILP: any rep will do, e.g. e0 = (bits of c) @
    Lx_dual, and every coset reached by shifting e0 is in the same orbit.  Walking
    the cosets in increasing order labels each orbit by its smallest id -- which is
    also the coset its MILP is solved for, so the numbering is deterministic.
    """
    k, Lx_dual, perms, coset, shifted = (setup.k, setup.Lx_dual, setup.perms,
                                         setup.coset, setup.shifted)
    orbit_of = np.full(1 << k, -1, int)
    orbits = []                                     # orbit id -> (coset, syndrome)
    for c in range(1, 1 << k):
        if orbit_of[c] >= 0:
            continue
        s = np.array([(c >> i) & 1 for i in range(k)])
        e0 = (s @ Lx_dual % 2).astype(np.uint8)
        assert coset(e0) == c
        orbit_of[c] = len(orbits)
        for p in perms:
            orbit_of[coset(shifted(e0, p))] = len(orbits)
        orbits.append((c, s))
    return orbit_of, orbits


def _read_checkpoint(path):
    """{orbit id: latest record} of a JSONL checkpoint written by this module.

    Later lines win, so a re-solved orbit overwrites its earlier record.  A torn
    final line (the run was killed mid-write) is ignored.
    """
    out = {}
    if path is None or not Path(path).exists():
        return out
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            out[rec["orbit"]] = rec
    return out


def _restored(rec, c, n):
    """(e, weight, status) recorded for coset ``c``, or None if it must be solved.

    Only "optimal" (with its representative) and "above" (no representative) are
    proofs.  Anything else -- "unknown", a foreign orbit id, another code's file --
    is re-solved.
    """
    if rec is None or rec.get("coset") != c or rec.get("status") not in ("optimal", "above"):
        return None
    if rec["status"] == "above":                    # recorded without a representative
        return None, None, "above"
    if rec.get("weight") is None or rec.get("rep") is None:
        return None
    rep = np.asarray(rec["rep"], dtype=int)
    if np.any(rep < 0) or np.any(rep >= n):         # rep of another (bigger) code
        return None
    e = np.zeros(n, dtype=np.uint8)
    e[rep] = 1
    return e, int(rec["weight"]), "optimal"


def build_catalogue(code_name, max_weight=14, time_limit=600, workers=1, checkpoint=None):
    """Solve one MILP per translation orbit; maps coset id (1..2^k-1) to ``(rep,
    weight, orbit id, status)``, weight -1 above the cap and -2 unresolved.

    With ``checkpoint`` (a path) every solved orbit is appended as one JSON line and
    flushed, and orbits the file already proves are absorbed without solving, so a
    killed run resumes where it stopped.  A checkpoint is only valid for the code and
    the ``max_weight`` cap it was written with (that is what "above" means), so keep
    one file per run.
    """
    setup = code_setup(code_name)
    l, m, k = setup.l, setup.m, setup.k
    H_z, Lz, Lx_dual = setup.H_z, setup.Lz, setup.Lx_dual
    perms, coset, shifted = setup.perms, setup.coset, setup.shifted
    orbits = partition_orbits(setup)[1]

    # Phase 2: one MILP per orbit, in worker processes if asked for.
    done, t0 = {}, time.time()
    ckpt = None
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        ckpt = open(checkpoint, "a")

    def absorb(orbit_id, e, w, status, save=True):
        """Fill every coset of a solved orbit (rep, weight, orbit id, status).

        ``save`` writes the checkpoint line; a record read back from the file is not
        written again, so the file keeps one line per orbit however often it is used.
        """
        c, s = orbits[orbit_id]
        rep = None if e is None else [int(q) for q in np.nonzero(e)[0]]
        if e is None:                               # no rep: only its coset is needed
            e = (s @ Lx_dual % 2).astype(np.uint8)
        else:
            assert coset(e) == c and not (H_z.astype(int) @ e % 2).any()
        if save and ckpt is not None:
            ckpt.write(json.dumps({"orbit": orbit_id, "coset": c, "weight": w,
                                   "status": status, "rep": rep}) + "\n")
            ckpt.flush()
        for p in perms:
            ep = shifted(e, p)
            done.setdefault(coset(ep), (ep if w is not None else None,
                                        w if w is not None else (-1 if status == "above" else -2),
                                        orbit_id, status))
        print(f"orbit {orbit_id:3d}  coset {c:4d}  "
              f"{'weight %2d' % w if w is not None else '> cap' if status == 'above' else 'UNKNOWN'}  "
              f"{status}  [{len(done)}/{2**k - 1} cosets, {time.time() - t0:.0f}s]", flush=True)

    try:
        saved = _read_checkpoint(checkpoint)
        tasks = []
        for orbit_id, (c, s) in enumerate(orbits):
            restored = _restored(saved.get(orbit_id), c, setup.n)
            if restored is None:
                tasks.append((orbit_id, H_z, Lz, s, max_weight, time_limit))
            else:                                   # already proven: no MILP needed
                absorb(orbit_id, *restored, save=False)
        if workers == 1:                            # in-process: no pool overhead
            for task in tasks:
                absorb(*_solve_orbit(task))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_solve_orbit, task) for task in tasks]
                for fut in as_completed(futures):   # progress in completion order
                    absorb(*fut.result())
    finally:
        if ckpt is not None:
            ckpt.close()
    assert len(done) == 2 ** k - 1
    return l, m, k, done


def compact_shift(e, l, m):
    """Translate so the support's cyclic bounding box is smallest, at the origin."""
    lm = l * m
    pts = np.array([(q % lm // m, q % lm % m) for q in np.nonzero(e)[0]])
    shift = []
    for axis, size in ((0, l), (1, m)):
        occ = np.zeros(size, bool)
        occ[pts[:, axis]] = True
        gaps = [(i, (np.nonzero(np.roll(occ, -i - 1))[0][0] + 1)) for i in np.nonzero(occ)[0]]
        i, g = max(gaps, key=lambda t: t[1])           # biggest empty arc follows i
        shift.append(-(i + g) % size)                   # first occupied site after it -> 0
    da, db = shift
    out = np.zeros_like(e)
    for q in np.nonzero(e)[0]:
        blk, r = divmod(q, lm)
        a, b = divmod(r, m)
        out[blk * lm + ((a + da) % l) * m + (b + db) % m] = 1
    return out


# ------------------------------------------------------------------- plotting
def plot_lowest(l, m, orbits, path, max_weight=None, ncols=10):
    """orbits: list of (weight, size, rep) sorted by weight.  Tiny dots, patterns only.

    One row per weight: a new row starts whenever the weight changes (the rest of
    the previous row is left blank).
    """
    limit = max_weight if max_weight else (orbits[0][0] if orbits else None)
    sel = [o for o in orbits if limit is None or o[0] <= limit]
    rows = []                                          # each row is one weight
    for o in sel:
        if not rows or o[0] != rows[-1][0][0] or len(rows[-1]) == ncols:
            rows.append([])
        rows[-1].append(o)
    if not rows:
        print(f"plot_lowest: nothing to plot ({len(orbits)} orbits, "
              f"max_weight={max_weight}) -> {path} skipped")
        return

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.5 * ncols, 1.1 * nrows + 0.9))
    axes = np.atleast_2d(axes)
    lm = l * m
    gx, gy = np.meshgrid(np.arange(m), np.arange(l))
    for ax in axes.ravel():
        ax.axis("off")                                 # blanks pad each row
    for ax_row, row in zip(axes, rows):
        for ax, (w, size, rep) in zip(ax_row, row):
            ax.scatter(gx, gy, s=1.0, c="#cccccc", lw=0)              # L sites
            ax.scatter(gx + .5, gy + .5, s=1.0, c="#e3e3e3", lw=0)    # R sites
            for q in np.nonzero(rep)[0]:
                blk, r = divmod(q, lm)
                a, b = divmod(r, m)
                ax.scatter(b + .5 * blk, a + .5 * blk, s=9, lw=0,
                           c="#0072B2" if blk == 0 else "#D55E00", zorder=3)
            ax.set_xlim(-.7, m + .2); ax.set_ylim(l + .2, -.7)
            ax.set_aspect("equal")
            ax.set_title(f"w={w}  ×{size}", fontsize=6, pad=1)
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=3, c="#0072B2", label="L qubit"),
                        plt.Line2D([], [], marker="o", ls="", ms=3, c="#D55E00", label="R qubit")],
               loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=2, frameon=False, fontsize=7)
    fig.suptitle(f"Lowest-weight X-type logical cosets ({len(sel)} translation orbits)", fontsize=8)
    footnote = ("One panel per translation orbit of X-type logical cosets: w = minimum weight, "
                "×N = number of cosets in the orbit (same pattern shifted on the torus). "
                "Blue = L qubit, orange = R qubit, grey = unflipped. "
                "Z-type cosets are the mirror image.")
    fig.text(0.5, 0.012, footnote, ha="center", va="bottom", fontsize=6, color="#444444")
    fig.tight_layout(rect=(0, 0.12, 1, 0.97), pad=0.3)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--code", default="144_12_12_ldpc")
    ap.add_argument("--max-weight", type=int, default=14,
                    help="cosets with no rep of at most this weight are only marked '> cap'")
    ap.add_argument("--time-limit", type=float, default=600, help="seconds per MILP")
    ap.add_argument("--workers", type=int, default=2,
                    help="processes solving translation orbits in parallel")
    ap.add_argument("--checkpoint", default=None,
                    help="resume file for solved orbits (default: "
                         "results/cosets/<code>/checkpoint.jsonl)")
    args = ap.parse_args()

    checkpoint = (Path(args.checkpoint) if args.checkpoint
                  else ROOT / "results" / "cosets" / args.code / "checkpoint.jsonl")
    l, m, k, done = build_catalogue(args.code, args.max_weight, args.time_limit, args.workers,
                                    checkpoint=checkpoint)
    out = ROOT / "results" / "cosets" / args.code
    out.mkdir(parents=True, exist_ok=True)

    ids = sorted(done)
    n = 2 * l * m
    np.savez(out / "catalogue.npz", coset=ids,
             rep=np.array([done[c][0] if done[c][0] is not None else np.zeros(n, np.uint8)
                           for c in ids]),
             weight=[done[c][1] for c in ids], orbit=[done[c][2] for c in ids],
             status=[done[c][3] for c in ids])

    by_orbit = {}
    for c in ids:
        rep, w, o, st = done[c]
        by_orbit.setdefault(o, [w, 0, rep])[1] += 1
    found = [(w, size, compact_shift(rep, l, m)) for w, size, rep in by_orbit.values() if w > 0]
    orbits = sorted(found, key=lambda t: (t[0], -t[1]))
    others = {}
    for w, size, _ in by_orbit.values():
        if w < 0:
            others[w] = others.get(w, 0) + size

    with open(out / "catalogue.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["orbit", "weight", "n_cosets", "support"])
        for i, (w, size, rep) in enumerate(orbits):
            wr.writerow([i, w, size, " ".join(map(str, np.nonzero(rep)[0]))])

    hist = {}
    for w, size, _ in orbits:
        o, c = hist.get(w, (0, 0))
        hist[w] = (o + 1, c + size)
    print("weight: (orbits, cosets)", dict(sorted(hist.items())))
    print(f"cosets above cap {args.max_weight}: {others.get(-1, 0)}, unresolved: {others.get(-2, 0)}")
    plot_lowest(l, m, orbits, out / "lowest_weight.png", args.max_weight)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
