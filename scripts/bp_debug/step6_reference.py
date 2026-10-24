"""Step 6: MWPM reference and logical-check cross-check (card BD7, issue #13).

Contract: ``scripts/bp_debug/INTERFACES.md``.  For the surface codes, compare the
failure rate of pymatching MWPM (the reference oracle for these codes) against plain
BP, both scored with ``common.independent_logical_failure``, and cross-check that
independent check against the repo's own ``has_logical_error`` evaluation on the
repo ``MWPMAgent`` prediction.

Deliberate reading of the contract (step 6 names ``d7_surface`` and ``d5_surface``,
the global flag default is the three-code list): the ``--codes`` default here is the
step-6 scope, ``d7_surface d5_surface``.  ``pymatching.Matching(H)`` needs columns of
weight <= 2, which only the surface codes have (tile/gross H columns have weight 3),
so those codes cannot be run through this MWPM reference at all.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.bp_debug.common import (
    decode_bp,
    independent_logical_failure,
    load_config,
    make_env,
    random_shots,
    syndromes,
    uniform_shots,
)

RESULTS_DIR = Path("results/bp_debug")
DEFAULT_CODES = ("d7_surface", "d5_surface")
IID_RATES = (0.001, 0.003, 0.005)
UNIFORM_WEIGHTS = (1, 2, 3)
BP_MAX_ITER = 30  # repo `BPAgent` setting
QUICK_SHOTS = 100  # per dataset, for `--quick`
EXAMPLE_COUNT = 3


def _check_matrix(matrix):
    """0/1 uint8 copy of a parity check matrix, the way the repo hands it to pymatching."""
    return np.array(matrix.cpu(), dtype=np.uint8)


def mwpm_failure(code, shots):
    """pymatching MWPM failure flags, scored with the independent logical check.

    X errors are decoded from the z-syndrome with H_z, Z errors from the x-syndrome
    with H_x - exactly the `pymatching.Matching(H)` construction of the repo
    `MWPMAgent`.
    """
    import pymatching

    x_syndrome, z_syndrome = syndromes(code, shots)
    x_pred = pymatching.Matching(_check_matrix(code.H.z)).decode_batch(z_syndrome)
    z_pred = pymatching.Matching(_check_matrix(code.H.x)).decode_batch(x_syndrome)
    return independent_logical_failure(code, shots, x_pred, z_pred)


def repo_mwpm_path(config, shots):
    """The repo's own `MWPMAgent` evaluation path.

    Returns ``(repo_failure, independent_failure, examples)``.  ``repo_failure`` is
    ``not classical_inference(...)`` (the repo's `is_syndrome_free and not
    has_logical_error` verdict); ``independent_failure`` is
    `common.independent_logical_failure` on the *same* prediction, so the two can be
    compared shot by shot.  ``examples`` are up to EXAMPLE_COUNT indices where they
    disagree.
    """
    from src.train_utils.inference import classical_inference, get_agent_and_inference

    n = config.n
    env = make_env(config, shots)
    agent, _ = get_agent_and_inference(config, env, "mwpm")

    repo_failure = np.zeros(len(shots), dtype=bool)
    independent_failure = np.zeros(len(shots), dtype=bool)
    examples = []
    for i in range(len(shots)):
        obs, info = env.reset()
        error_pred, _ = agent.select_action(obs)
        error_pred = np.asarray(error_pred.cpu().numpy(), dtype=np.int64)
        # `select_action` returns X error indices first, then Z indices offset by n_data.
        x_pred = np.zeros((1, n), dtype=np.uint8)
        z_pred = np.zeros((1, n), dtype=np.uint8)
        x_pred[0, error_pred[error_pred < n]] = 1
        z_pred[0, error_pred[error_pred >= n] - n] = 1

        independent_failure[i] = independent_logical_failure(env.code, shots[i:i + 1], x_pred, z_pred)[0]
        # `state=` reuses the reset already done above, so `classical_inference` sees
        # the same shot and re-derives the same prediction.
        repo_failure[i] = not classical_inference(agent, env, state=(obs, info))

        if repo_failure[i] != independent_failure[i] and len(examples) < EXAMPLE_COUNT:
            examples.append(i)
    return repo_failure, independent_failure, examples


def evaluate_dataset(config, code, shots, p_prior):
    """MWPM (pymatching and repo path) and plain BP on one dataset."""
    start = time.time()
    bp = decode_bp(code, shots, p_prior=p_prior, max_iter=BP_MAX_ITER)
    mwpm = mwpm_failure(code, shots)
    repo_mwpm, repo_independent, examples = repo_mwpm_path(config, shots)
    return {
        "n": int(len(shots)),
        "bp_failures": int(bp["failure"].sum()),
        "bp_failure_fraction": float(bp["failure"].mean()),
        "bp_non_converged": int((~(bp["converged_x"] & bp["converged_z"])).sum()),
        "mwpm_failures": int(mwpm.sum()),
        "mwpm_failure_fraction": float(mwpm.mean()),
        "repo_mwpm_failures": int(repo_mwpm.sum()),
        "repo_mwpm_failure_fraction": float(repo_mwpm.mean()),
        "repo_vs_independent_disagreements": int((repo_mwpm != repo_independent).sum()),
        "repo_vs_independent_disagreement_examples": examples,
        "seconds": round(time.time() - start, 2),
    }


def run(codes, n_per_weight, seed, quick=False):
    """Collect the step-6 payload: per code, per iid rate and per uniform weight."""
    n = min(n_per_weight, QUICK_SHOTS) if quick else n_per_weight
    payload = {
        "codes": list(codes),
        "seed": int(seed),
        "quick": bool(quick),
        "n_per_weight": int(n_per_weight),
        "shots_per_dataset": int(n),
        "iid_rates": [float(p) for p in IID_RATES],
        "uniform_weights": [int(w) for w in UNIFORM_WEIGHTS],
        "bp_max_iter": BP_MAX_ITER,
        "bp_prior_convention": "2*curriculum_error_rate/3 (repo BPAgent, depolarizing)",
        "bp_prior": {},
        "iid": {},
        "uniform": {},
    }

    for code_name in codes:
        config = load_config(code_name)
        env = make_env(config)
        code = env.code
        p_prior = 2 * env.curriculum_error_rate / 3
        payload["bp_prior"][code_name] = float(p_prior)
        print(f"[{code_name}] n={config.n} prior={p_prior:.6g}", flush=True)

        iid = {}
        for p in IID_RATES:
            shots = random_shots(config, n, p, seed=seed)
            result = evaluate_dataset(config, code, shots, p_prior)
            iid[f"{p:g}"] = result
            print(f"  iid p={p:g} n={result['n']} mwpm={result['mwpm_failures']} "
                  f"bp={result['bp_failures']} repo_mwpm={result['repo_mwpm_failures']} "
                  f"disagreements={result['repo_vs_independent_disagreements']}", flush=True)
        payload["iid"][code_name] = iid

        # `uniform_shots` returns weights 1..max_weight concatenated, `n` shots per weight.
        shots = uniform_shots(config, n, max_weight=max(UNIFORM_WEIGHTS), seed=seed)
        uniform = {}
        for w in UNIFORM_WEIGHTS:
            result = evaluate_dataset(config, code, shots[(w - 1) * n: w * n], p_prior)
            uniform[str(w)] = result
            print(f"  uniform w={w} n={result['n']} mwpm={result['mwpm_failures']} "
                  f"bp={result['bp_failures']} repo_mwpm={result['repo_mwpm_failures']} "
                  f"disagreements={result['repo_vs_independent_disagreements']}", flush=True)
        payload["uniform"][code_name] = uniform

    payload["observations"] = observations(payload)
    return payload


def observations(payload):
    """Up to 5 bullets of facts read off the payload (no speculation)."""
    bullets = []
    iid_cells = [(code, key, result) for code, table in payload["iid"].items() for key, result in table.items()]
    uniform_cells = [(code, key, result) for code, table in payload["uniform"].items() for key, result in table.items()]
    cells = iid_cells + uniform_cells
    if not cells:
        return bullets

    total_shots = sum(result["n"] for _, _, result in cells)
    disagreements = sum(result["repo_vs_independent_disagreements"] for _, _, result in cells)
    bullets.append(
        f"Repo `has_logical_error` evaluation and the independent logical check disagree on "
        f"{disagreements} of {total_shots} MWPM predictions over all {len(cells)} datasets."
    )

    if uniform_cells:
        mwpm_failures = sum(result["mwpm_failures"] for _, _, result in uniform_cells)
        bp_failures = sum(result["bp_failures"] for _, _, result in uniform_cells)
        n = sum(result["n"] for _, _, result in uniform_cells)
        bullets.append(
            f"On the uniform weight-{min(UNIFORM_WEIGHTS)}..{max(UNIFORM_WEIGHTS)} shots MWPM fails "
            f"{mwpm_failures} times and plain BP fails {bp_failures} times ({n} shots in total)."
        )
        weight1 = [(code, result) for code, key, result in uniform_cells if key == "1"]
        if weight1:
            bullets.append(
                f"Weight-1 shots (one X and one Z flip at independent positions, "
                f"{weight1[0][1]['n']} shots per code): MWPM fails "
                f"{sum(r['mwpm_failures'] for _, r in weight1)} times, BP fails "
                f"{sum(r['bp_failures'] for _, r in weight1)} times."
            )

    for code, table in payload["iid"].items():
        lowest = table[f"{min(IID_RATES):g}"]
        bullets.append(
            f"[{code}] iid p={min(IID_RATES):g} (n={lowest['n']}): MWPM fails {lowest['mwpm_failures']}, "
            f"BP fails {lowest['bp_failures']} (non-converged BP shots: {lowest['bp_non_converged']})."
        )
    return bullets[:5]


def _table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def _dataset_caption(payload):
    n = payload["shots_per_dataset"]
    rates = ", ".join(f"{p:g}" for p in payload["iid_rates"])
    weights = payload["uniform_weights"]
    return (
        f"Dataset: iid depolarizing shots at p in {{{rates}}} (`random_shots`, {n} shots per p) and uniform "
        f"shots of nominal weight {min(weights)}-{max(weights)} (`uniform_shots`, {n} shots per weight; weight w "
        "means w X flips AND w Z flips at independent random positions, so up to 2w qubits are touched). "
        f"Codes: {', '.join(payload['codes'])}. Seed {payload['seed']}. "
        "Decoders: pymatching MWPM on the check matrices (`Matching(H_z)` for X errors, `Matching(H_x)` for Z), "
        "the repo `MWPMAgent` path (`classical_inference`, i.e. `has_logical_error`), and plain BP "
        f"(`ldpc.BpDecoder`, prior 2*curriculum_error_rate/3, max_iter={payload['bp_max_iter']}). MWPM and BP are "
        "scored with `common.independent_logical_failure`; `disagreements` counts shots where the repo verdict "
        "differs from that check on the same MWPM prediction."
    )


def _result_rows(table):
    rows = []
    for code, per_key in table.items():
        for key, result in per_key.items():
            rows.append([
                code, key, result["n"], result["mwpm_failures"], f"{result['mwpm_failure_fraction']:.4f}",
                result["bp_failures"], f"{result['bp_failure_fraction']:.4f}", result["bp_non_converged"],
                result["repo_mwpm_failures"], result["repo_vs_independent_disagreements"],
                ",".join(str(i) for i in result["repo_vs_independent_disagreement_examples"]) or "-",
            ])
    return rows


def render_markdown(payload):
    """`results/bp_debug/step6_reference.md`."""
    columns = ["n", "MWPM fails", "MWPM frac", "BP fails", "BP frac", "BP non-conv",
               "repo MWPM fails", "disagreements", "disagreement shot indices"]
    lines = [
        "# Step 6: MWPM reference and logical-check cross-check",
        "",
        _dataset_caption(payload),
        "",
        "## iid depolarizing shots",
        "",
    ]
    lines += _table(_result_rows(payload["iid"]), ["code", "p"] + columns)
    lines += ["", f"## Uniform shots, weights {min(payload['uniform_weights'])}"
                  f"-{max(payload['uniform_weights'])}", ""]
    lines += _table(_result_rows(payload["uniform"]), ["code", "weight"] + columns)
    lines += ["", "## Observation", ""]
    lines += [f"- {bullet}" for bullet in payload["observations"]]
    lines.append("")
    return "\n".join(lines)


def write_results(payload, results_dir=RESULTS_DIR):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "step6_reference.json"
    md_path = results_dir / "step6_reference.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    md_path.write_text(render_markdown(payload))
    return json_path, md_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Step 6: MWPM reference and logical-check cross-check.")
    parser.add_argument("--codes", nargs="+", default=list(DEFAULT_CODES))
    parser.add_argument("--n-per-weight", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="tiny sizes for tests (< 20 s)")
    args = parser.parse_args(argv)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    payload = run(args.codes, args.n_per_weight, args.seed, quick=args.quick)
    json_path, md_path = write_results(payload)
    print(f"Wrote {json_path} and {md_path}", flush=True)
    return payload


if __name__ == "__main__":
    main()
