"""Step 2 (card BD3, issue #13): weight-1 (single-qubit) shots split by Pauli.

Contract: ``scripts/bp_debug/INTERFACES.md``.  One shot per data qubit carrying a single
Pauli, decoded with plain BP at the repo's settings plus a prior sweep, so the systematic
weight-1 failures can be attributed to a Pauli type and to individual qubits.

Run from the repository root::

    python -m scripts.bp_debug.step2_weight1 [--codes ...] [--quick]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.bp_debug import common

MODULE = "step2_weight1"
RESULTS_DIR = Path("results/bp_debug")
DEFAULT_CODES = ("d7_surface", "288_8_12_tile", "144_12_12_ldpc")
PRIORS = (0.001, 0.01, 0.1)  # on top of the repo's own prior
MAX_ITER = 30  # `BPAgent`'s max_iter
MAX_LISTED = 20  # contract: "first 20" failing qubit indices
QUICK_QUBITS = 8  # `--quick` samples this many qubits instead of all n


def check_degrees(code):
    """(n_x_checks, n_z_checks) per data qubit, read off the parity check matrices."""
    H_x, H_z = common._as_bits(code.H.x), common._as_bits(code.H.z)
    return H_x.sum(axis=0).astype(int), H_z.sum(axis=0).astype(int)


def boundary_qubits(code):
    """Bool (n,): qubits that touch only one stabilizer of a type (code boundary)."""
    n_x, n_z = check_degrees(code)
    return (n_x == 1) | (n_z == 1)


def prior_table(env):
    """Ordered {label: p_prior}: the repo's BP prior first, then the contract's sweep."""
    priors = {"repo": 2 * env.curriculum_error_rate / 3}
    priors.update({f"{p:g}": p for p in PRIORS})
    return priors


def sample_qubits(n, quick):
    """Qubit indices to test; `--quick` spreads a handful over the code, all qubits otherwise."""
    if not quick or n <= QUICK_QUBITS:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, QUICK_QUBITS)).astype(int))


def summarise(failures, qubits, boundary):
    """Contract row: failures / n, first 20 failing qubits, and whether those are boundary qubits."""
    failing = qubits[failures]
    on_boundary = boundary[failing]
    return {
        "n": int(len(qubits)),
        "failures": int(failures.sum()),
        "fraction": float(failures.mean()),
        "failing_qubits": failing[:MAX_LISTED].tolist(),
        "num_failing": int(len(failing)),
        "num_boundary_failing": int(on_boundary.sum()),
        "boundary_failing_qubits": failing[on_boundary][:MAX_LISTED].tolist(),
        "all_failing_are_boundary": bool(on_boundary.all()),
    }


def run_code(code_name, quick=False):
    """Decode the single-qubit shots of one code for every Pauli and prior."""
    config = common.load_config(code_name)
    env = common.make_env(config)
    code = env.code
    priors = prior_table(env)
    qubits = sample_qubits(config.n, quick)
    n_x, n_z = check_degrees(code)
    boundary = boundary_qubits(code)

    paulis = {}
    for pauli in common.PAULIS:
        shots = common.single_qubit_shots(config, pauli)[qubits]
        paulis[pauli] = {
            label: summarise(common.decode_bp(code, shots, p_prior=p, max_iter=MAX_ITER)["failure"], qubits, boundary)
            for label, p in priors.items()
        }
        print(f"{code_name} {pauli}: " + ", ".join(f"{label} {row['failures']}/{row['n']}"
                                                   for label, row in paulis[pauli].items()), flush=True)

    return {
        "n": int(config.n),
        "d": int(code.d),
        "priors": {label: float(p) for label, p in priors.items()},
        "sampled_qubits": qubits.tolist(),
        "check_degrees": {"n_x": n_x.tolist(), "n_z": n_z.tolist()},
        "boundary_qubits": np.flatnonzero(boundary).tolist(),
        "paulis": paulis,
    }


def observations(results):
    """Facts only, at most five bullets (contract)."""
    bullets = []
    codes = results["codes"]
    for name, data in codes.items():
        rows = {pauli: data["paulis"][pauli]["repo"] for pauli in common.PAULIS}
        bullets.append(
            f"`{name}` at the repo prior ({data['priors']['repo']:.6g}): "
            + ", ".join(f"{pauli} fails on {row['failures']}/{row['n']} qubits" for pauli, row in rows.items())
            + "."
        )
    for name, data in codes.items():
        for pauli in common.PAULIS:
            row = data["paulis"][pauli]["repo"]
            if row["failures"] and not row["all_failing_are_boundary"]:
                non_boundary = sorted(set(row["failing_qubits"]) - set(row["boundary_failing_qubits"]))
                bullets.append(
                    f"`{name}` {pauli}: {row['failures'] - row['num_boundary_failing']} of the "
                    f"{row['failures']} failing qubits touch two or more stabilizers of each type "
                    f"(not boundary qubits), e.g. {non_boundary[:5]}."
                )
                break
    spread = []
    for name, data in codes.items():
        for pauli in common.PAULIS:
            counts = {label: data["paulis"][pauli][label]["failures"] for label in data["priors"]}
            if max(counts.values()) != min(counts.values()):
                spread.append(f"`{name}` {pauli} {min(counts.values())}-{max(counts.values())}")
    bullets.append("Prior sweep " + (f"changes the failure count for {', '.join(spread)}" if spread
                                     else "does not change any failure count") + ".")
    return bullets[:5]
def markdown(results):
    """Tables plus the caption and the Observation section."""
    meta = results["meta"]
    first = next(iter(results["codes"].values()))
    quick_note = (f" `--quick`: only qubits {first['sampled_qubits']} sampled." if meta["quick"] else "")
    lines = [
        f"# Step 2: weight-1 (single-qubit) shots, split by Pauli (`{MODULE}`)",
        "",
        f"Caption. Dataset: `common.single_qubit_shots` - one shot per data qubit with a single X, Y or Z error on "
        f"that qubit, so n shots per Pauli per code (n = number of data qubits); the shots are deterministic, no "
        f"sampling is involved. Seed: {meta['seed']} (set for numpy and torch).{quick_note} Decoder: "
        f"`common.decode_bp` with the repo settings (`max_iter={meta['max_iter']}`) and `p_prior` = repo value "
        f"`2*curriculum_error_rate/3` (label `repo`) plus {list(PRIORS)}. Failure = non-zero residual syndrome, or "
        f"residual anticommuting with a logical operator.",
        "",
    ]
    for name, data in results["codes"].items():
        lines += [f"## {name} (n={data['n']}, d={data['d']}, repo prior {data['priors']['repo']:.6g})", "",
                  "| Pauli | prior | failures / n | fraction | failing qubits (first 20) | boundary failing (first 20) |",
                  "|---|---|---|---|---|---|"]
        for pauli in common.PAULIS:
            for label, row in data["paulis"][pauli].items():
                failing = ", ".join(str(q) for q in row["failing_qubits"]) or "-"
                boundary = ", ".join(str(q) for q in row["boundary_failing_qubits"]) or "-"
                lines.append(f"| {pauli} | {label} ({data['priors'][label]:.6g}) | {row['failures']} / {row['n']} | "
                             f"{row['fraction']:.3f} | {failing} | {boundary} |")
        lines += ["", "Check degrees per qubit (number of X / Z stabilizers touching it); a qubit that touches only "
                      "one stabilizer of a type is on the code boundary:", "",
                  "| n_x checks | n_z checks | qubits | boundary | qubit indices (first 20) |", "|---|---|---|---|---|"]
        degrees = list(zip(data["check_degrees"]["n_x"], data["check_degrees"]["n_z"]))
        for degree in sorted(set(degrees)):
            qubits = [q for q, other in enumerate(degrees) if other == degree]
            listed = ", ".join(str(q) for q in qubits[:MAX_LISTED]) + (" ..." if len(qubits) > MAX_LISTED else "")
            lines.append(f"| {degree[0]} | {degree[1]} | {len(qubits)} | {'yes' if min(degree) == 1 else 'no'} | "
                         f"{listed} |")
        lines.append("")
    lines += ["## Observation", ""] + [f"- {bullet}" for bullet in results["observations"]] + [""]
    return "\n".join(lines)


def write_outputs(results, out_dir=None):
    """`results/bp_debug/<module>.json` and `.md` (contract)."""
    out_dir = Path(RESULTS_DIR if out_dir is None else out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{MODULE}.json").write_text(json.dumps(results, indent=2) + "\n")
    (out_dir / f"{MODULE}.md").write_text(markdown(results))
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Weight-1 (single-qubit) BP failures split by Pauli.")
    parser.add_argument("--codes", nargs="+", default=list(DEFAULT_CODES))
    parser.add_argument("--n-per-weight", type=int, default=5000,
                        help="unused here: step 2 uses one shot per qubit (n shots per Pauli)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="sample a few qubits per code (test-sized)")
    args = parser.parse_args(argv)

    common._seed(args.seed)
    results = {
        "meta": {
            "module": MODULE,
            "dataset": "single-qubit shots, one per data qubit (common.single_qubit_shots)",
            "seed": args.seed,
            "quick": bool(args.quick),
            "codes": list(args.codes),
            "n_per_weight": args.n_per_weight,
            "n_per_weight_used": False,
            "max_iter": MAX_ITER,
            "priors": list(PRIORS),
        },
        "codes": {code_name: run_code(code_name, quick=args.quick) for code_name in args.codes},
    }
    results["observations"] = observations(results)
    out_dir = write_outputs(results)
    print(f"wrote {out_dir / (MODULE + '.json')} and {out_dir / (MODULE + '.md')}")
    return results


if __name__ == "__main__":
    main()

