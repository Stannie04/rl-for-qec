"""Step 7 of the BP debug investigation: stratified iid failure check (card BD8).

The binding contract is ``scripts/bp_debug/INTERFACES.md``, section "Step 7".
Run from the repository root::

    python -m scripts.bp_debug.step7_stratified --quick --codes d5_surface

iid depolarizing shots are decoded with ``common.decode_bp`` using the repo's BP
settings, then binned by the number of X-type and Z-type flips ``(n_x, n_z)`` so
the per-bin failure fractions can be compared with the directly measured LER.
"""

import argparse
import json
import os

import numpy as np
import torch

from scripts.bp_debug.common import decode_bp, load_config, make_env, random_shots

MODULE_NAME = "step7_stratified"
DEFAULT_CODES = ("d7_surface", "288_8_12_tile", "144_12_12_ldpc")
P_VALUES = (0.001, 0.002, 0.003, 0.004, 0.005)
DEFAULT_N_PER_P = 200000
QUICK_N_PER_P = 2000
MAX_ITER = 30
MAX_BIN_WEIGHT = 4  # keep the (n_x, n_z) bins while n_x + n_z <= 4
OVERFLOW_BIN = "5+"  # every heavier shot lands here
FINE_P = 0.001  # the p of the per-(n_x, n_z) table
FINE_BINS = ("(1,0)", "(0,1)", "(1,1)", "(2,0)", "(0,2)", "(1,2)", "(2,1)")

# Ordered bin labels: lightest first, overflow last.
BIN_ORDER = tuple(
    f"({n_x},{n_z})"
    for total in range(MAX_BIN_WEIGHT + 1)
    for n_x in range(total + 1)
    for n_z in (total - n_x,)
) + (OVERFLOW_BIN,)


def p_key(p):
    """Stable string key for a physical error rate (``0.001`` -> ``"0.001"``)."""
    return f"{p:g}"


def _seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def flip_counts(shots):
    """(n_x, n_z) per shot: X-type and Z-type flips (a Y error counts in both)."""
    shots = np.asarray(shots)
    return shots[:, 0].sum(axis=1).astype(int), shots[:, 1].sum(axis=1).astype(int)


def bin_labels(n_x, n_z):
    """Bin label per shot: ``"(n_x,n_z)"`` while ``n_x + n_z <= 4``, else the ``"5+"`` bin."""
    return np.array(
        [OVERFLOW_BIN if a + b > MAX_BIN_WEIGHT else f"({a},{b})" for a, b in zip(n_x, n_z)]
    )


def summarise_bins(labels, failure):
    """Per bin: shots, share of all shots, failures and failure fraction (None when empty)."""
    failure = np.asarray(failure, dtype=bool)
    total = len(failure)
    bins = {}
    for label in BIN_ORDER:
        mask = labels == label
        count = int(mask.sum())
        if count == 0:
            bins[label] = {"shots": 0, "share": 0.0, "failures": 0, "failure_fraction": None}
            continue
        failures = int(failure[mask].sum())
        bins[label] = {
            "shots": count,
            "share": count / total,
            "failures": failures,
            "failure_fraction": failures / count,
        }
    return bins


def reconstructed_ler(bins):
    """`sum_bins (bin share) * (bin failure fraction)`; equals the direct LER by construction."""
    return float(
        sum(
            stats["share"] * stats["failure_fraction"]
            for stats in bins.values()
            if stats["failure_fraction"] is not None
        )
    )


def run_code(code_name, n_per_p, seed, p_values=P_VALUES, progress=print):
    """Stratify `n_per_p` iid shots per p for one code; returns the per-code payload."""
    config = load_config(code_name)
    env = make_env(config)
    code = env.code
    p_prior = 2 * env.curriculum_error_rate / 3  # the repo's `bp` prior for depolarizing noise

    per_p = {}
    for p in p_values:
        shots = random_shots(config, n_per_p, p, seed=seed)
        decoded = decode_bp(code, shots, p_prior=p_prior, max_iter=MAX_ITER)
        n_x, n_z = flip_counts(shots)
        bins = summarise_bins(bin_labels(n_x, n_z), decoded["failure"])
        direct_ler = float(decoded["failure"].mean())
        per_p[p_key(p)] = {
            "n_shots": int(len(shots)),
            "failures": int(decoded["failure"].sum()),
            "direct_ler": direct_ler,
            "reconstructed_ler": reconstructed_ler(bins),
            "bins": bins,
        }
        progress(
            f"  {code_name} p={p_key(p)}: LER {direct_ler:.6f} "
            f"({per_p[p_key(p)]['failures']}/{len(shots)})"
        )

    return {
        "code_name": code_name,
        "n": int(code.n_data),
        "curriculum_error_rate": float(env.curriculum_error_rate),
        "p_prior": float(p_prior),
        "per_p": per_p,
    }


def run(codes, n_per_p, seed, quick=False, p_values=P_VALUES, progress=print):
    """Full result payload: one `run_code` per code, keyed by code name."""
    progress(f"{MODULE_NAME}: {len(codes)} codes x {len(p_values)} p x {n_per_p} shots, seed {seed}")
    return {
        "module": MODULE_NAME,
        "seed": seed,
        "n_per_p": n_per_p,
        "quick": bool(quick),
        "p_values": list(p_values),
        "codes": {name: run_code(name, n_per_p, seed, p_values, progress) for name in codes},
    }


def _frac(stats, digits=5):
    """`failures/shots (fraction)` for one bin, or `-` when the bin is empty."""
    if stats["failure_fraction"] is None:
        return "-"
    return f"{stats['failures']}/{stats['shots']} ({stats['failure_fraction']:.{digits}f})"


def _caption(result):
    p_values = ", ".join(p_key(p) for p in result["p_values"])
    return (
        "Dataset: iid depolarizing shots (`create_dataset_from_random_shots`) at p in "
        f"{{{p_values}}}; {result['n_per_p']} shots per p per code; seed {result['seed']}; "
        f"codes {', '.join(result['codes'])}; BP via `common.decode_bp` with the repo settings "
        f"(`p_prior = 2*curriculum_error_rate/3`, `max_iter={MAX_ITER}`)."
    )


def _code_section(code, p_keys):
    lines = [
        f"## {code['code_name']} (n={code['n']}, curriculum_error_rate="
        f"{code['curriculum_error_rate']:.6g}, p_prior={code['p_prior']:.6g})",
        "",
        "| bin | " + " | ".join(f"p={p}" for p in p_keys) + " |",
        "|" + "---|" * (len(p_keys) + 1),
    ]
    for label in BIN_ORDER:
        cells = [_frac(code["per_p"][p]["bins"][label]) for p in p_keys]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "| p | shots | failures | direct LER | reconstructed LER |",
        "|---|---|---|---|---|",
    ]
    for p in p_keys:
        stats = code["per_p"][p]
        lines.append(
            f"| {p} | {stats['n_shots']} | {stats['failures']} | {stats['direct_ler']:.6f} | "
            f"{stats['reconstructed_ler']:.6f} |"
        )
    lines.append("")
    return lines


def _fine_section(result):
    lines = [
        f"## Failure fraction by `(n_x, n_z)` at p={p_key(FINE_P)}",
        "",
        "| code | bin | shots | failures | failure fraction |",
        "|---|---|---|---|---|",
    ]
    for code_name, code in result["codes"].items():
        stats_p = code["per_p"].get(p_key(FINE_P))
        if stats_p is None:
            continue
        for label in FINE_BINS:
            stats = stats_p["bins"][label]
            fraction = "-" if stats["failure_fraction"] is None else f"{stats['failure_fraction']:.5f}"
            lines.append(
                f"| {code_name} | {label} | {stats['shots']} | {stats['failures']} | {fraction} |"
            )
    lines.append("")
    return lines


def _observation_section(result):
    p0 = p_key(FINE_P)
    codes = result["codes"]

    gaps = [
        abs(stats["direct_ler"] - stats["reconstructed_ler"])
        for code in codes.values()
        for stats in code["per_p"].values()
    ]
    single = "; ".join(
        f"{name} (1,0) {_frac(code['per_p'][p0]['bins']['(1,0)'], 4)} vs "
        f"(0,1) {_frac(code['per_p'][p0]['bins']['(0,1)'], 4)}"
        for name, code in codes.items()
    )
    worst = []
    for name, code in codes.items():
        stats_p = code["per_p"][p0]
        label = max(BIN_ORDER, key=lambda bin_label: stats_p["bins"][bin_label]["failures"])
        failures = stats_p["bins"][label]["failures"]
        share = failures / stats_p["failures"] if stats_p["failures"] else float("nan")
        worst.append(f"{name} {label} {failures}/{stats_p['failures']} ({share:.1%})")
    overflow = "; ".join(
        f"{name} {code['per_p'][p0]['bins'][OVERFLOW_BIN]['share']:.2%} of shots, "
        f"{_frac(code['per_p'][p0]['bins'][OVERFLOW_BIN], 4)}"
        for name, code in codes.items()
    )
    zero = "; ".join(
        f"{name} {code['per_p'][p0]['bins']['(0,0)']['shots']} shots / "
        f"{code['per_p'][p0]['bins']['(0,0)']['failures']} failures"
        for name, code in codes.items()
    )

    bullets = [
        f"Reconstructed LER matches the direct LER to within {max(gaps):.2e} for every code and p, so the "
        f"`{OVERFLOW_BIN}` bin plus the `(n_x, n_z)` bins cover all shots.",
        f"At p={p0} the single-flip bins fail at: {single}.",
        f"At p={p0} the bin holding the most failures per code (share of that code's failures): "
        + "; ".join(worst) + ".",
        f"At p={p0} the `{OVERFLOW_BIN}` bin holds {overflow}.",
        f"At p={p0} the zero-flip bin (0,0) holds {zero}.",
    ]
    return ["## Observation", ""] + [f"- {bullet}" for bullet in bullets] + [""]


def render_markdown(result):
    """Tables plus the Observation section for `results/bp_debug/<module>.md`."""
    p_keys = [p_key(p) for p in result["p_values"]]
    lines = [
        f"# Step 7: stratified iid failure check (`{MODULE_NAME}`)",
        "",
        _caption(result),
        "",
        f"Bins are the flip counts `(n_x, n_z)` (`n_x = shots[:,0].sum()`, `n_z = shots[:,1].sum()`); every "
        f"shot with `n_x + n_z > {MAX_BIN_WEIGHT}` falls into the `{OVERFLOW_BIN}` bin. Cells are "
        "`failures/shots (failure fraction)`.",
        "",
    ]
    for code in result["codes"].values():
        lines += _code_section(code, p_keys)
    lines += _fine_section(result)
    lines += _observation_section(result)
    return "\n".join(lines) + "\n"


def write_outputs(result, out_dir):
    """Write `<out_dir>/step7_stratified.json` and `.md`; returns both paths."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{MODULE_NAME}.json")
    md_path = os.path.join(out_dir, f"{MODULE_NAME}.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(result))
    return json_path, md_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--codes", nargs="+", default=list(DEFAULT_CODES))
    parser.add_argument(
        "--n-per-weight",
        type=int,
        default=None,
        help=f"shots per p (default {DEFAULT_N_PER_P}; {QUICK_N_PER_P} with --quick)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help=f"tiny run: {QUICK_N_PER_P} shots per p")
    parser.add_argument("--out-dir", default=os.path.join("results", "bp_debug"))
    return parser.parse_args(argv)


def resolve_n_per_p(args):
    """Shots per p: an explicit `--n-per-weight` wins, else the `--quick`/full default."""
    if args.n_per_weight is not None:
        return args.n_per_weight
    return QUICK_N_PER_P if args.quick else DEFAULT_N_PER_P


def main(argv=None):
    args = parse_args(argv)
    _seed_everything(args.seed)

    result = run(args.codes, resolve_n_per_p(args), args.seed, quick=args.quick)
    json_path, md_path = write_outputs(result, args.out_dir)
    print(f"wrote {json_path} and {md_path}")
    return result


if __name__ == "__main__":
    main()

