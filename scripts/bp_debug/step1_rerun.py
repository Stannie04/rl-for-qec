"""Step 1: rerun the per-weight baseline for every decoder (card BD2, issue #13).

Contract: ``scripts/bp_debug/INTERFACES.md``.  Run from the repository root::

    python -m scripts.bp_debug.step1_rerun --quick --codes d5_surface

For every code this reruns the repository's own evaluation path
(``get_agent_and_inference`` + ``classical_inference``) for ``bp``, ``bp_osd``, ``bp_lsd`` and
``mwpm`` on uniform shots of nominal weight 1..5, and decodes the same shots with
``common.decode_bp`` under the repository's settings (``p_prior = 2*curriculum_error_rate/3``,
``max_iter=30``).  Both pipelines' per-weight failure fractions and their per-shot
disagreement counts are written to ``results/bp_debug/step1_rerun.json`` and
``results/bp_debug/step1_rerun.md``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.bp_debug.common import decode_bp, load_config, make_env, uniform_shots

DEFAULT_CODES = ("d7_surface", "288_8_12_tile", "144_12_12_ldpc")
REPO_DECODERS = ("bp", "bp_osd", "bp_lsd", "mwpm")
# `common.decode_bp` implements the three BP variants; MWPM has no counterpart there.
COMMON_DECODERS = ("bp", "bp_osd", "bp_lsd")
MAX_WEIGHT = 5
MAX_ITER = 30
QUICK_N_PER_WEIGHT = 5
RESULTS_DIR = Path("results/bp_debug")
JSON_PATH = RESULTS_DIR / "step1_rerun.json"
MD_PATH = RESULTS_DIR / "step1_rerun.md"
WEIGHTS = tuple(range(1, MAX_WEIGHT + 1))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes", nargs="+", default=list(DEFAULT_CODES))
    parser.add_argument("--n-per-weight", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="tiny sizes for tests (< 20 s)")
    return parser.parse_args(argv)


def repo_failures(config, shots, agent_name):
    """Repository path: `get_agent_and_inference` + `classical_inference`, one env per decoder.

    `classical_inference` consumes `env.shots` in order via `env.reset`, so the env is fresh
    per decoder and the shots are served in the order `uniform_shots` produced them.
    """
    from src.train_utils.inference import get_agent_and_inference

    env = make_env(config, shots)
    agent, inference = get_agent_and_inference(config, env, agent_name)
    return np.array([not inference(agent, env) for _ in shots], dtype=bool)


def weight_labels(n_per_weight):
    """Nominal weight per row: `create_dataset_from_uniform_shots` writes weight blocks 1..5.

    The nominal weight is not recoverable from the shot itself (weight w places w X flips and w
    Z flips independently, so up to 2w qubits are touched), only from the row layout.
    """
    return np.repeat(np.arange(1, MAX_WEIGHT + 1), n_per_weight)


def per_weight(failures, labels):
    """`failures` split into one boolean array per nominal weight 1..MAX_WEIGHT."""
    return [failures[labels == weight] for weight in WEIGHTS]


def summarise(failures, labels):
    """Per-weight failure counts, sample sizes and fractions for the `.md`/`.json` tables."""
    groups = per_weight(failures, labels)
    return {
        "failures": [int(group.sum()) for group in groups],
        "n": [int(group.size) for group in groups],
        "failure_fraction": [float(group.mean()) if group.size else float("nan") for group in groups],
    }


def run_code(code_name, n_per_weight, seed):
    """Both pipelines on uniform weights 1..5 for one code; returns the per-code result dict."""
    config = load_config(code_name)
    shots = uniform_shots(config, n_per_weight, max_weight=MAX_WEIGHT, seed=seed)
    labels = weight_labels(n_per_weight)

    env = make_env(config)  # no shots: this only needs `code` and the curriculum prior
    p_prior = 2 * env.curriculum_error_rate / 3

    repo, common, repo_errors = {}, {}, {}
    for agent_name in REPO_DECODERS:
        print(f"  repo   {agent_name}", flush=True)
        try:
            repo[agent_name] = repo_failures(config, shots, agent_name)
        except ValueError as error:
            # PROBE: `MWPMAgent` builds `pymatching.Matching(H)`, which only accepts graphlike
            # check matrices (<= 2 ones per column) - it cannot be constructed at all for the
            # tile and LDPC codes.  Recorded as data instead of aborting the whole rerun.
            print(f"    {agent_name} unavailable: {error}", flush=True)
            repo_errors[agent_name] = f"{type(error).__name__}: {error}"
    for decoder in COMMON_DECODERS:
        print(f"  common {decoder}", flush=True)
        common[decoder] = decode_bp(
            env.code, shots, p_prior=p_prior, max_iter=MAX_ITER, decoder=decoder
        )["failure"]

    return {
        "n_data": int(config.n),
        "curriculum_error_rate": float(env.curriculum_error_rate),
        "p_prior": float(p_prior),
        "n_per_weight": int(n_per_weight),
        "weights": list(WEIGHTS),
        "repo": {name: summarise(failures, labels) for name, failures in repo.items()},
        "repo_errors": repo_errors,
        "common": {name: summarise(failures, labels) for name, failures in common.items()},
        "disagreements": {
            name: {
                "per_weight": [int((repo[name] != common[name])[labels == w].sum()) for w in WEIGHTS],
                "total": int((repo[name] != common[name]).sum()),
                "n": int(len(shots)),
            }
            for name in COMMON_DECODERS
            if name in repo
        },
    }


def _fraction_string(entry):
    return "/".join(f"{fraction:.3f}" for fraction in entry["failure_fraction"])


def observations(payload):
    """At most 5 factual bullets; no speculation about causes or fixes."""
    bullets = [
        "Dataset: `create_dataset_from_uniform_shots` puts w X flips AND w Z flips at independent "
        "random positions, so a weight-w shot touches up to 2w of the n_data qubits; the nominal "
        "weight label comes from the weight-major row layout, not from the shot.",
    ]
    for code_name, code in list(payload["results"].items())[:3]:
        parts = ", ".join(
            f"{decoder} {_fraction_string(code['repo'][decoder])}" for decoder in REPO_DECODERS
            if decoder in code["repo"]
        )
        errors = "; ".join(
            f"{decoder} unavailable ({message})" for decoder, message in code["repo_errors"].items()
        )
        bullets.append(
            f"{code_name} (n_data={code['n_data']}, p_prior={code['p_prior']:.2e}): repo failure "
            f"fraction by weight 1..{MAX_WEIGHT} - {parts}."
            + (f" Repo path not constructible: {errors}." if errors else "")
        )
    bullets.append(
        "Repo vs common per-shot disagreements: "
        + "; ".join(
            f"{code_name} {decoder} "
            f"{code['disagreements'][decoder]['total']}/{code['disagreements'][decoder]['n']}"
            for code_name, code in payload["results"].items()
            for decoder in COMMON_DECODERS
        )
        + "."
    )
    return bullets[:5]


def caption(payload):
    return (
        f"Dataset: `create_dataset_from_uniform_shots`, {payload['n_per_weight']} shots per nominal "
        f"weight for weights 1..{MAX_WEIGHT} ({payload['n_per_weight'] * MAX_WEIGHT} shots per code), "
        f"seed {payload['seed']}; codes: {', '.join(payload['codes'])}; "
        f"{'quick (tiny) run' if payload['quick'] else 'full run'}. "
        f"repo = `get_agent_and_inference` + `classical_inference`; "
        f"common = `common.decode_bp` with `p_prior=2*curriculum_error_rate/3`, `max_iter={MAX_ITER}`."
    )


def render_markdown(payload):
    lines = [
        "# Step 1: rerun baseline per weight",
        "",
        caption(payload),
        "",
        "## Failure fraction (code x weight x decoder)",
        "",
        "| code | weight | decoder | repo failures/n | repo failure fraction | common failures/n "
        "| common failure fraction | disagreements |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for code_name, code in payload["results"].items():
        for decoder in REPO_DECODERS:
            if decoder in code["repo_errors"]:
                lines.append(f"| {code_name} | - | {decoder} | error | - | - | - | - |")
                continue
            repo = code["repo"][decoder]
            for i, weight in enumerate(WEIGHTS):
                if decoder in code["common"]:
                    common = code["common"][decoder]
                    common_cells = (
                        f"{common['failures'][i]}/{common['n'][i]}",
                        f"{common['failure_fraction'][i]:.3f}",
                        str(code["disagreements"][decoder]["per_weight"][i]),
                    )
                else:
                    common_cells = ("-", "-", "-")
                lines.append(
                    f"| {code_name} | {weight} | {decoder} | {repo['failures'][i]}/{repo['n'][i]} | "
                    f"{repo['failure_fraction'][i]:.3f} | {common_cells[0]} | {common_cells[1]} | "
                    f"{common_cells[2]} |"
                )
    lines += [
        "",
        "## Pipeline agreement (repo vs common, per shot)",
        "",
        "| code | decoder | disagreements | n |",
        "| --- | --- | --- | --- |",
    ]
    for code_name, code in payload["results"].items():
        for decoder in COMMON_DECODERS:
            if decoder in code["repo_errors"]:
                lines.append(f"| {code_name} | {decoder} | repo path unavailable | - |")
                continue
            entry = code["disagreements"][decoder]
            lines.append(f"| {code_name} | {decoder} | {entry['total']} | {entry['n']} |")
    lines += ["", "## Observation", ""]
    lines += [f"- {bullet}" for bullet in observations(payload)]
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    if args.n_per_weight < 1:
        raise SystemExit("--n-per-weight must be >= 1")

    n_per_weight = QUICK_N_PER_WEIGHT if args.quick else args.n_per_weight
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    payload = {
        "step": "step1_rerun",
        "quick": bool(args.quick),
        "seed": int(args.seed),
        "n_per_weight": int(n_per_weight),
        "max_weight": MAX_WEIGHT,
        "max_iter": MAX_ITER,
        "codes": list(args.codes),
        "repo_pipeline": "src.train_utils.inference.get_agent_and_inference + classical_inference",
        "common_pipeline": "scripts.bp_debug.common.decode_bp",
        "p_prior_convention": "2 * curriculum_error_rate / 3",
        "repo_decoders": list(REPO_DECODERS),
        "common_decoders": list(COMMON_DECODERS),
        "results": {},
    }
    for code_name in args.codes:
        print(f"[{code_name}] {n_per_weight} shots per weight", flush=True)
        payload["results"][code_name] = run_code(code_name, n_per_weight, args.seed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    MD_PATH.write_text(render_markdown(payload))
    print(f"wrote {JSON_PATH} and {MD_PATH}", flush=True)
    return payload


if __name__ == "__main__":
    main()

