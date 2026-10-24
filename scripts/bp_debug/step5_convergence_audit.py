"""Step 5: convergence / failure-class audit (card BD6, issue #13).

Repo-default plain BP (``common.decode_bp`` with ``p_prior = 2*curriculum_error_rate/3`` and
``max_iter=30``) on iid depolarizing shots at ``p=0.003``; the failing shots are split into

  (a) not converged - effective convergence (``effective_convergence``) is False, i.e. a plane has
      the ``ldpc`` ``converge`` flag False *and* a non-free residual syndrome,
  (b) converged, residual syndrome-free, but the residual anticommutes with a logical operator
      (``common.independent_logical_failure``) - a genuine BP logical failure,
  (c) converged, residual syndrome-free, the independent check says success but the repo's
      ``classical_inference`` says failure - the repo evaluation is the odd one out,
  (d) other (in practice empty, see the convergence cross-checks below).

"Failing" is the union of the two evaluations (independent check and repo evaluation): a shot
counts as failing when either of them says so, so neither evaluation is assumed correct.

Reading of the contract (INTERFACES.md, step 5): the repo ``classical_inference`` path is only
re-run on that failing subset, which is a few percent of the shots.  For the other shots the repo
verdict is derived from the residual state ``shot ^ prediction`` - that is exactly the state
``classical_inference`` evaluates, because it flips the predicted qubits of the code loaded with
the shot.  The repo path is still run on every failing shot, so the derivation is cross-checked
where it matters.

Run from the repository root::

    python -m scripts.bp_debug.step5_convergence_audit --quick --codes d5_surface
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.bp_debug import common

MODULE = "step5_convergence_audit"
RESULTS_DIR = Path("results/bp_debug")

DEFAULT_CODES = ("d7_surface", "288_8_12_tile", "144_12_12_ldpc")
P = 0.003  # iid depolarizing physical error rate of the audit (fixed by the contract)
MAX_ITER = 30  # repo `BPAgent` setting
CLASSES = ("a_not_converged", "b_converged_logical", "c_evaluation_bug", "d_other")
EXAMPLES = 3
QUICK_SHOTS = 250


def repo_failure(code, shots, x_pred, z_pred):
    """Repo success criterion per shot, from the residual ``shot ^ prediction``.

    ``classical_inference`` applies the predicted flips to the code loaded with the shot, so the
    state it evaluates is the residual; ``is_syndrome_free() and not has_logical_error()`` on that
    state is the repo's verdict.  Returns bool (N,) failure flags.
    """
    residual_x = np.asarray(shots[:, 0], dtype=np.uint8) ^ np.asarray(x_pred, dtype=np.uint8)
    residual_z = np.asarray(shots[:, 1], dtype=np.uint8) ^ np.asarray(z_pred, dtype=np.uint8)

    failures = np.empty(len(shots), dtype=bool)
    for i in range(len(shots)):
        code.set_error_pattern(residual_x[i], residual_z[i])
        failures[i] = not (bool(code.is_syndrome_free()) and not code.has_logical_error())
    return failures


def repo_path_failure(config, shots):
    """Repo path verdict per shot: ``get_agent_and_inference(config, env, "bp")`` + inference.

    A fresh env serves ``shots`` in order, so verdict ``i`` belongs to shot ``i``.
    """
    from src.train_utils.inference import get_agent_and_inference  # pulls in the agent stack

    env = common.make_env(config, shots)
    agent, inference = get_agent_and_inference(config, env, "bp")
    return np.array([not inference(agent, env) for _ in range(len(shots))], dtype=bool)


def effective_convergence(converged_x, converged_z, x_free, z_free):
    """Per-plane convergence: the ``ldpc`` ``converge`` flag OR a syndrome-free residual.

    Defensive reading of the flag: a plane whose hard decision already satisfies its syndrome has
    converged even if ``ldpc`` reports otherwise, and such a shot is not a class-(a) failure.
    Measured on the audit shots the two agree exactly (the
    ``n_not_converged_*_but_*_syndrome_free`` counters are 0), so a plane that lands here only via
    ``*_free`` is a flag/syndrome mismatch worth reporting rather than assuming away.
    """
    return (converged_x | x_free) & (converged_z | z_free)


def classify(failing, converged, syndrome_free, logical_failure, repo_failure):
    """Partition the failing shots; precedence a > b > c > d.  Returns {class: bool mask}.

    ``converged`` must be the effective convergence from ``effective_convergence``.
    """
    a = failing & ~converged
    b = failing & converged & syndrome_free & logical_failure
    c = failing & converged & syndrome_free & ~logical_failure & repo_failure
    d = failing & ~(a | b | c)

    masks = dict(zip(CLASSES, (a, b, c, d)))
    # The four masks must be disjoint and cover the failing set exactly.
    total = a.astype(np.uint8) + b + c + d
    assert np.array_equal(total, failing.astype(np.uint8)), (
        "failure classes must partition the failing shots"
    )
    return masks


def audit_code(code_name, n_shots, seed):
    """Run the audit for one code; returns the JSON-able record."""
    config = common.load_config(code_name, "bp")
    shots = common.random_shots(config, n_shots, P, seed=seed)
    env = common.make_env(config, shots)
    code = env.code

    p_prior = 2 * env.curriculum_error_rate / 3  # repo `BPAgent` prior for depolarizing noise
    print(f"[{code_name}] decoding {len(shots)} iid shots at p={P} with plain BP (prior {p_prior:.2e})", flush=True)
    started = time.time()
    decoded = common.decode_bp(code, shots, p_prior=p_prior, max_iter=MAX_ITER)
    print(f"[{code_name}] decoded in {time.time() - started:.1f}s", flush=True)

    converged_x, converged_z = decoded["converged_x"], decoded["converged_z"]
    x_pred, z_pred = decoded["x_pred"], decoded["z_pred"]

    residual_x = np.asarray(shots[:, 0], dtype=np.uint8) ^ x_pred
    residual_z = np.asarray(shots[:, 1], dtype=np.uint8) ^ z_pred
    # `common.syndromes` names by check type: the X-check syndrome comes first and is the one the
    # Z-error plane (`z_pred`, `converged_z`) must satisfy; the Z-check syndrome (second) is the
    # X-error plane's (`x_pred`, `converged_x`).
    x_check, z_check = common.syndromes(code, np.stack([residual_x, residual_z], axis=1))
    x_free, z_free = ~z_check.any(axis=1), ~x_check.any(axis=1)
    syndrome_free = x_free & z_free
    converged = effective_convergence(converged_x, converged_z, x_free, z_free)

    failure_independent = decoded["failure"]  # independent check: syndrome or logical
    logical_failure = failure_independent & syndrome_free
    failure_repo = repo_failure(code, shots, x_pred, z_pred)
    failing = failure_independent | failure_repo
    masks = classify(failing, converged, syndrome_free, logical_failure, failure_repo)

    # The repo's own path on the failing subset only (the contract asks for it; `--quick` stays fast).
    n_failing = int(failing.sum())
    if n_failing:
        print(f"[{code_name}] re-running the repo classical_inference path on {n_failing} failing shots", flush=True)
        repo_subset = repo_path_failure(config, shots[failing])
    else:
        repo_subset = np.zeros(0, dtype=bool)
    disagreements = int((repo_subset != failure_repo[failing]).sum())

    return {
        "code_name": code_name,
        "n": int(code.n_data),
        "k": int(code.k),
        "d": int(code.d),
        "n_shots": int(len(shots)),
        "p": P,
        "p_prior": float(p_prior),
        "max_iter": MAX_ITER,
        "n_failing": n_failing,
        "failing_fraction": float(failing.mean()),
        "n_failure_independent": int(failure_independent.sum()),
        "n_failure_repo": int(failure_repo.sum()),
        "n_failure_repo_subset": int(repo_subset.sum()),
        "n_repo_path_disagreements": disagreements,
        "n_repo_only": int((failure_repo & ~failure_independent).sum()),  # repo fails, check passes
        "n_check_only": int((failure_independent & ~failure_repo).sum()),  # check fails, repo passes
        "convergence": {
            "n_not_converged_x": int((~converged_x).sum()),
            "n_not_converged_z": int((~converged_z).sum()),
            # Cross-check of the `converge` flag: a plane flagged as not converged must show a
            # non-free residual syndrome, so both counters are expected to be 0 - an `ldpc`
            # flag/syndrome mismatch would land here.  `effective_convergence` treats a plane as
            # converged when the flag says so OR its syndrome is free, so a mismatch cannot push a
            # shot into class (a).
            "n_not_converged_x_but_x_syndrome_free": int((~converged_x & x_free).sum()),
            "n_not_converged_z_but_z_syndrome_free": int((~converged_z & z_free).sum()),
        },
        "classes": {
            name: {
                "count": int(mask.sum()),
                "fraction_of_failing": float(mask.sum() / n_failing) if n_failing else 0.0,
                "fraction_of_shots": float(mask.mean()),
                "examples": np.flatnonzero(mask)[:EXAMPLES].tolist(),
            }
            for name, mask in masks.items()
        },
    }


CLASS_LABELS = {
    "a_not_converged": "(a) not converged",
    "b_converged_logical": "(b) converged, logical",
    "c_evaluation_bug": "(c) repo eval mismatch",
    "d_other": "(d) other",
}

def render_markdown(results):
    """Markdown tables + caption + a facts-only Observation section."""
    lines = [
        "# Step 5: convergence / failure-class audit",
        "",
        f"Dataset: iid depolarizing shots, `p={results['p']}`, `{results['n_shots']}` shots per code, "
        f"seed `{results['seed']}`. Decoder: repo-default plain BP (`common.decode_bp`, "
        f"`p_prior=2*curriculum_error_rate/3`, `max_iter={results['max_iter']}`). \"Failing\" = union of the "
        "independent check and the repo evaluation; (a) not effectively converged (flag False *and* "
        "syndrome non-free), (b) converged + syndrome-free + "
        "independent logical failure, (c) converged + syndrome-free + independent success but repo failure, "
        "(d) other.",
        "",
        "| code | n | shots | failing | frac | " + " | ".join(CLASS_LABELS[name] for name in CLASSES) + " |",
        "|---|---|---|---|---|" + "---|" * len(CLASSES),
    ]
    for record in results["codes"].values():
        cells = [
            f"{record['classes'][name]['count']} ({record['classes'][name]['fraction_of_failing']:.0%})"
            for name in CLASSES
        ]
        lines.append(
            f"| {record['code_name']} | {record['n']} | {record['n_shots']} | {record['n_failing']} | "
            f"{record['failing_fraction']:.3f} | " + " | ".join(cells) + " |"
        )

    lines += [
        "",
        "## Failure-class examples (shot indices)",
        "",
        "| code | class | count | example shot indices |",
        "|---|---|---|---|",
    ]
    for record in results["codes"].values():
        for name in CLASSES:
            entry = record["classes"][name]
            if entry["count"]:
                lines.append(
                    f"| {record['code_name']} | {CLASS_LABELS[name]} | {entry['count']} | {entry['examples']} |"
                )

    lines += [
        "",
        "## Diagnostics",
        "",
        "| code | independent failures | repo failures | repo-only | check-only | repo path re-decoded | "
        "repo path disagreements | non-converged x | non-converged z | flag/syndrome mismatch x | "
        "flag/syndrome mismatch z |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for record in results["codes"].values():
        convergence = record["convergence"]
        lines.append(
            f"| {record['code_name']} | {record['n_failure_independent']} | {record['n_failure_repo']} | "
            f"{record['n_repo_only']} | {record['n_check_only']} | {record['n_failing']} | "
            f"{record['n_repo_path_disagreements']} | {convergence['n_not_converged_x']} | "
            f"{convergence['n_not_converged_z']} | "
            f"{convergence['n_not_converged_x_but_x_syndrome_free']} | "
            f"{convergence['n_not_converged_z_but_z_syndrome_free']} |"
        )

    lines += ["", "## Observation", ""]
    for bullet in observations(results):
        lines.append(f"- {bullet}")
    return "\n".join(lines) + "\n"


def observations(results):
    """Facts only, at most five bullets, derived from the numbers."""
    bullets = []
    for record in results["codes"].values():
        classes = record["classes"]
        if record["n_failing"]:
            main = max(CLASSES, key=lambda name: classes[name]["count"])
            bullets.append(
                f"`{record['code_name']}`: {record['n_failing']}/{record['n_shots']} shots fail "
                f"({record['failing_fraction']:.3f}); the largest class is {CLASS_LABELS[main]} with "
                f"{classes[main]['count']} shots ({classes[main]['fraction_of_failing']:.0%} of the failing set)."
            )
        else:
            bullets.append(f"`{record['code_name']}`: no failing shot in {record['n_shots']} shots.")
    for record in results["codes"].values():
        convergence = record["convergence"]
        bullets.append(
            f"`{record['code_name']}`: BP did not converge on {convergence['n_not_converged_x']} x-plane and "
            f"{convergence['n_not_converged_z']} z-plane decodes out of {record['n_shots']}; "
            f"{convergence['n_not_converged_x_but_x_syndrome_free']} x and "
            f"{convergence['n_not_converged_z_but_z_syndrome_free']} z of those are flag/syndrome mismatches "
            "(flag False while the residual syndrome is free, hence effectively converged); "
            f"{record['n_repo_only']} shots fail the repo evaluation only and {record['n_check_only']} the "
            "independent check only."
        )
    total_disagreements = sum(record["n_repo_path_disagreements"] for record in results["codes"].values())
    total_subset = sum(record["n_failing"] for record in results["codes"].values())
    bullets.append(
        f"Repo path re-run on all {total_subset} failing shots: {total_disagreements} verdict(s) differ from the "
        "residual-state evaluation used for the classification."
    )
    return bullets[:5]


def write_outputs(results, out_dir):
    """Write `step5_convergence_audit.json` and `.md` into `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{MODULE}.json"
    md_path = out_dir / f"{MODULE}.md"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    md_path.write_text(render_markdown(results))
    return json_path, md_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Step 5: convergence / failure-class audit.")
    parser.add_argument("--codes", nargs="+", default=list(DEFAULT_CODES))
    parser.add_argument("--n-per-weight", type=int, default=5000, help="shots per code (iid p=0.003)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="tiny sizes for tests (< 20 s)")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args(argv)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    n_shots = QUICK_SHOTS if args.quick else args.n_per_weight
    results = {
        "module": MODULE,
        "p": P,
        "seed": args.seed,
        "n_shots": n_shots,
        "max_iter": MAX_ITER,
        "prior": "2*curriculum_error_rate/3 (repo BPAgent default)",
        "quick": bool(args.quick),
        "codes": {code_name: audit_code(code_name, n_shots, args.seed) for code_name in args.codes},
    }

    json_path, md_path = write_outputs(results, args.out_dir)
    print(f"wrote {json_path} and {md_path}", flush=True)
    return results


if __name__ == "__main__":
    main()
