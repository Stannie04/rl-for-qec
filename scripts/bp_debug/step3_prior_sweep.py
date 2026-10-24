"""Step 3: sweep the BP prior `p_prior` on uniform and iid-depolarizing shots (card BD4, issue #13).

Binding contract: ``scripts/bp_debug/INTERFACES.md``.  Run from the repository root::

    python -m scripts.bp_debug.step3_prior_sweep --quick --codes d5_surface

Writes ``results/bp_debug/step3_prior_sweep.json`` and ``.md``.  Part A decodes uniform shots of
weights 1..3, part B iid depolarizing shots at p in {0.001, ..., 0.005}; both sweep the same
seven `p_prior` values.

Contract reading where it is ambiguous: part A uses *uniform* shots, which have no physical
depolarizing rate `p`, so "matched 2p/3 (per-shot p)" is mapped onto the density of that dataset.
A weight-`w` shot has exactly `w` X-type flips among `n` qubits, so the equivalent iid rate is
`p_nominal = 3w/(2n)` (the rate whose X-flip marginal `2p/3` equals `w/n`) and the matched prior
is `w/n`.  Part B uses the generating `p` directly.  The `2p` and `p` priors use the same
`p_nominal`.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.bp_debug.common import decode_bp, load_config, make_env, random_shots, uniform_shots, weights

DEFAULT_CODES = ("d7_surface", "288_8_12_tile", "144_12_12_ldpc")
PART_A_WEIGHTS = (1, 2, 3)
PART_B_P_VALUES = (0.001, 0.002, 0.003, 0.004, 0.005)
# `matched_2p_over_3`, `2p` and `p` are derived from the row's nominal rate; the rest are absolute.
PRIOR_LABELS = ("matched_2p_over_3", "2p", "p", "repo", "0.01", "0.05", "0.1")
FIXED_PRIORS = {"0.01": 0.01, "0.05": 0.05, "0.1": 0.1}
MAX_ITER = 30
DEFAULT_N_PER_WEIGHT = 5000
MIN_N_PER_P = 20000
QUICK_N_PER_WEIGHT = 50
QUICK_N_PER_P = 200
RESULTS_DIR = Path("results/bp_debug")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--codes", nargs="+", default=list(DEFAULT_CODES))
    parser.add_argument("--n-per-weight", type=int, default=DEFAULT_N_PER_WEIGHT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="tiny sizes for tests (< 20 s)")
    return parser.parse_args(argv)


def nominal_p_for_weight(weight, n):
    """Depolarizing rate whose X-flip marginal `2p/3` equals the uniform density `weight/n`."""
    return 1.5 * weight / n


def priors_for(p_nominal, repo_prior):
    """The contract's seven `p_prior` values, keyed by label."""
    return {
        "matched_2p_over_3": 2 * p_nominal / 3,
        "2p": 2 * p_nominal,
        "p": p_nominal,
        "repo": repo_prior,
        **FIXED_PRIORS,
    }


def sweep(code, shots, p_nominal, repo_prior):
    """Failure fraction per prior label for one shot set (both error planes, one BP run each)."""
    rows = {}
    for label, prior in priors_for(p_nominal, repo_prior).items():
        outcome = decode_bp(code, shots, p_prior=prior, max_iter=MAX_ITER)
        failures = int(outcome["failure"].sum())
        rows[label] = {
            "prior": float(prior),
            "failures": failures,
            "n": int(len(shots)),
            "failure_fraction": failures / len(shots),
        }
        print(f"    {label:<17} prior={prior:.6g} failures={failures}/{len(shots)}", flush=True)
    return rows


def run_code(code_name, *, n_per_weight, n_per_p, seed):
    """Part A (uniform weights 1..3) and part B (iid p in PART_B_P_VALUES) for one code."""
    config = load_config(code_name)
    env = make_env(config)
    n = config.n
    repo_prior = 2 * env.curriculum_error_rate / 3
    print(f"  {code_name}: n={n} repo prior={repo_prior:.6g}", flush=True)

    uniform = uniform_shots(config, n_per_weight, max_weight=max(PART_A_WEIGHTS), seed=seed)
    part_a = {}
    for weight in PART_A_WEIGHTS:
        block = uniform[(weight - 1) * n_per_weight: weight * n_per_weight]
        p_nominal = nominal_p_for_weight(weight, n)
        touched = float(weights(block).mean())
        print(f"  uniform weight {weight}: {len(block)} shots, mean qubits touched {touched:.2f}", flush=True)
        part_a[str(weight)] = {
            "weight": weight,
            "p_nominal": p_nominal,
            "n_shots": int(len(block)),
            "mean_qubits_touched": touched,
            "priors": sweep(env.code, block, p_nominal, repo_prior),
        }

    part_b = {}
    for index, p in enumerate(PART_B_P_VALUES):
        shots = random_shots(config, n_per_p, p, seed=seed + index + 1)
        print(f"  iid p={p}: {len(shots)} shots", flush=True)
        part_b[str(p)] = {
            "p": p,
            "n_shots": int(len(shots)),
            "priors": sweep(env.code, shots, p, repo_prior),
        }

    return {
        "n": n,
        "curriculum_error_rate": env.curriculum_error_rate,
        "repo_prior": repo_prior,
        "part_a": part_a,
        "part_b": part_b,
    }


def _prior_cells(priors):
    return [f"{priors[label]['failure_fraction']:.4f}" for label in PRIOR_LABELS]


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def observations(payload):
    """At most five factual bullets read off the sweep."""
    mid_p = PART_B_P_VALUES[len(PART_B_P_VALUES) // 2]
    bullets = []
    for name, entry in list(payload["codes"].items())[:4]:
        iid = {label: entry["part_b"][str(mid_p)]["priors"][label]["failure_fraction"] for label in PRIOR_LABELS}
        unit = {label: entry["part_a"]["1"]["priors"][label]["failure_fraction"] for label in PRIOR_LABELS}
        bullets.append(
            f"`{name}`: at iid p={mid_p:g} the failure fraction spans {iid[min(iid, key=iid.get)]:.4f} "
            f"({min(iid, key=iid.get)}) .. {iid[max(iid, key=iid.get)]:.4f} ({max(iid, key=iid.get)}) over the "
            f"{len(PRIOR_LABELS)} priors, repo prior {iid['repo']:.4f}; at uniform weight 1 it spans "
            f"{min(unit.values()):.4f} .. {max(unit.values()):.4f}."
        )
    total = payload["part_a_weights"][-1] * payload["n_per_weight"] + len(payload["part_b_p_values"]) * payload["n_per_p"]
    bullets.append(
        f"Priors swept per row: matched 2p/3, 2p, p (from the row's nominal rate), the repo value "
        f"2*curriculum/3, and the fixed 0.01/0.05/0.1; {total} shots per (code, prior)."
    )
    return bullets[:5]


def render_markdown(payload):
    caption = (
        f"Dataset: part A `create_dataset_from_uniform_shots` (nominal weight w = w X flips AND w Z flips at "
        f"independent positions, so up to 2w of the n qubits are touched) and part B `create_dataset_from_random_shots` "
        f"(iid depolarizing, per-qubit Pauli rate p); sample sizes {payload['n_per_weight']} shots per (code, weight) and "
        f"{payload['n_per_p']} shots per (code, p); seeds: --seed={payload['seed']} with per-code base seed "
        f"seed + 100*code_index and part B using base + p_index + 1; decoder `common.decode_bp` "
        f"(ldpc.BpDecoder, max_iter={MAX_ITER}, product_sum, parallel, prior passed through unchanged)."
    )
    rows_a, rows_b, rows_repo = [], [], []
    for name, entry in payload["codes"].items():
        for weight in payload["part_a_weights"]:
            block = entry["part_a"][str(weight)]
            rows_a.append([name, weight, f"{block['p_nominal']:.5f}", *_prior_cells(block["priors"]), block["n_shots"]])
        for p in payload["part_b_p_values"]:
            block = entry["part_b"][str(p)]
            rows_b.append([name, p, *_prior_cells(block["priors"]), block["n_shots"]])
        rows_repo.append([name, entry["n"], f"{entry['curriculum_error_rate']:.6g}", f"{entry['repo_prior']:.6g}"])

    lines = [
        "# Step 3: BP prior sweep",
        "",
        caption,
        "",
        "## Part A: uniform shots, weights 1..3",
        "",
        "Cell = BP failure fraction. `matched 2p/3 = w/n`, `2p = 3w/n`, `p = 3w/(2n)` (see module docstring).",
        "",
        *_table(["code", "w", "nominal p", *PRIOR_LABELS, "n shots"], rows_a),
        "",
        "## Part B: iid depolarizing shots",
        "",
        "Cell = BP failure fraction. `matched 2p/3 = 2p/3`, `2p` and `p` use the row's p.",
        "",
        *_table(["code", "p", *PRIOR_LABELS, "n shots"], rows_b),
        "",
        "## Repo prior per code",
        "",
        *_table(["code", "n", "curriculum_error_rate", "repo prior = 2*curriculum/3"], rows_repo),
        "",
        "## Observation",
        "",
        *[f"- {bullet}" for bullet in observations(payload)],
        "",
    ]
    return "\n".join(lines)


def write_outputs(payload):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "step3_prior_sweep.json").write_text(json.dumps(payload, indent=2) + "\n")
    (RESULTS_DIR / "step3_prior_sweep.md").write_text(render_markdown(payload))


def main(argv=None):
    args = parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    n_per_weight = min(args.n_per_weight, QUICK_N_PER_WEIGHT) if args.quick else args.n_per_weight
    n_per_p = min(args.n_per_weight, QUICK_N_PER_P) if args.quick else max(args.n_per_weight, MIN_N_PER_P)
    print(
        f"step3 prior sweep: codes={args.codes} n_per_weight={n_per_weight} n_per_p={n_per_p} "
        f"seed={args.seed} quick={args.quick}",
        flush=True,
    )

    payload = {
        "module": "step3_prior_sweep",
        "quick": args.quick,
        "seed": args.seed,
        "n_per_weight": n_per_weight,
        "n_per_p": n_per_p,
        "part_a_weights": list(PART_A_WEIGHTS),
        "part_b_p_values": list(PART_B_P_VALUES),
        "prior_labels": list(PRIOR_LABELS),
        "max_iter": MAX_ITER,
        "codes": {
            name: run_code(name, n_per_weight=n_per_weight, n_per_p=n_per_p, seed=args.seed + 100 * index)
            for index, name in enumerate(args.codes)
        },
    }

    write_outputs(payload)
    print(f"wrote {RESULTS_DIR}/step3_prior_sweep.json and {RESULTS_DIR}/step3_prior_sweep.md", flush=True)
    return payload


if __name__ == "__main__":
    main()
