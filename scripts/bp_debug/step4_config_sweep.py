"""Step 4: BP configuration sweep (card BD5, issue #13).

One factor at a time around the repo's plain-BP default (product_sum, parallel
schedule, ``ms_scaling_factor`` 1.0, ``max_iter`` 30) on iid depolarizing shots at
``p=0.003`` with the prior fixed at ``2p/3``.  The binding contract is
``scripts/bp_debug/INTERFACES.md``; run from the repository root as
``python -m scripts.bp_debug.step4_config_sweep``.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.bp_debug.common import decode_bp, load_config, make_env, random_shots

DEFAULT_CODES = ("d7_surface", "288_8_12_tile", "144_12_12_ldpc")
P_EVAL = 0.003
PRIOR = 2 * P_EVAL / 3  # the repo's depolarizing convention, held fixed across the sweep
BASE = {"max_iter": 30, "bp_method": "product_sum", "ms_scaling_factor": 1.0, "schedule": "parallel"}

MAX_ITER_GRID = (10, 30, 100, 500)
BP_METHOD_GRID = ("product_sum", "minimum_sum")
MS_SCALING_GRID = (0.5, 0.625, 0.75, 0.9, 1.0)
SCHEDULE_GRID = ("parallel", "serial")

OUT_DIR = Path("results/bp_debug")
MODULE_NAME = "step4_config_sweep"


def sweep_settings():
    """``(factor, setting, overrides)`` triples, one factor at a time around ``BASE``."""
    settings = []
    for value in MAX_ITER_GRID:
        settings.append(("max_iter", f"max_iter={value}", {"max_iter": value}))
    for value in BP_METHOD_GRID:
        settings.append(("bp_method", f"bp_method={value}", {"bp_method": value}))
    for value in MS_SCALING_GRID:
        # ms_scaling_factor only applies to minimum_sum, so pin the method there.
        settings.append(
            (
                "ms_scaling_factor",
                f"ms_scaling_factor={value} (minimum_sum)",
                {"bp_method": "minimum_sum", "ms_scaling_factor": value},
            )
        )
    for value in SCHEDULE_GRID:
        settings.append(("schedule", f"schedule={value}", {"schedule": value}))
    return settings


def valid_settings(kwargs):
    """True when ldpc 2.4.1 accepts ``kwargs`` for ``BpDecoder`` (invalid values raise ValueError)."""
    import ldpc

    probe = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    try:
        ldpc.BpDecoder(probe, error_rate=0.1, **kwargs)
        return True
    except ValueError:
        return False


def sweep_code(code_name, n_shots, seed):
    """One-factor-at-a-time rows for a single code (iid shots at ``P_EVAL``)."""
    config = load_config(code_name)
    code = make_env(config).code
    shots = random_shots(config, n_shots, P_EVAL, seed=seed)

    rows = []
    for factor, setting, overrides in sweep_settings():
        kwargs = {**BASE, **overrides}
        row = {"factor": factor, "setting": setting, **kwargs}
        if not valid_settings(kwargs):
            row["skipped"] = True
            print(f"[{code_name}] skip {setting}: not supported by ldpc 2.4.1")
        else:
            result = decode_bp(code, shots, p_prior=PRIOR, **kwargs)
            converged = result["converged_x"] & result["converged_z"]
            row["skipped"] = False
            row["failure_fraction"] = float(result["failure"].mean())
            row["non_converged_fraction"] = float((~converged).mean())
            print(
                f"[{code_name}] {setting}: failure={row['failure_fraction']:.4f} "
                f"non_converged={row['non_converged_fraction']:.4f}"
            )
        rows.append(row)
    return rows


def run(codes, n_shots, seed):
    """Sweep every code; returns the machine-readable payload (no file IO)."""
    return {
        "meta": {
            "codes": list(codes),
            "n_shots": n_shots,
            "seed": seed,
            "p_eval": P_EVAL,
            "prior": PRIOR,
            "base": BASE,
            "grid": {
                "max_iter": list(MAX_ITER_GRID),
                "bp_method": list(BP_METHOD_GRID),
                "ms_scaling_factor": list(MS_SCALING_GRID),
                "schedule": list(SCHEDULE_GRID),
            },
        },
        "results": {code_name: sweep_code(code_name, n_shots, seed) for code_name in codes},
    }


def _value(rows, setting, key):
    """Value of ``key`` for ``setting``, or None when the setting is missing/skipped."""
    row = next((row for row in rows if row["setting"] == setting), None)
    return None if row is None or row.get("skipped") else row[key]


def _compare(results, setting_a, setting_b, key="failure_fraction"):
    """``code a->b`` strings for two settings, skipping codes with missing values."""
    out = []
    for code_name, rows in results.items():
        a, b = _value(rows, setting_a, key), _value(rows, setting_b, key)
        if a is not None and b is not None:
            out.append(f"{code_name} {a:.4f}->{b:.4f}")
    return ", ".join(out)


def observations(payload):
    """At most 5 fact-only bullets computed from the sweep."""
    results = payload["results"]
    bullets = []

    base = {c: _value(rows, "max_iter=30", "failure_fraction") for c, rows in results.items()}
    bullets.append(
        "- Default (product_sum, parallel, ms_scaling_factor=1.0, max_iter=30): failure fraction "
        + ", ".join(f"{c}={v:.4f}" for c, v in base.items() if v is not None)
        + "."
    )

    bullets.append("- max_iter 30->500 failure: " + _compare(results, "max_iter=30", "max_iter=500") + ".")
    bullets.append(
        "- bp_method product_sum->minimum_sum failure: "
        + _compare(results, "bp_method=product_sum", "bp_method=minimum_sum")
        + "."
    )
    bullets.append(
        "- schedule parallel->serial failure: " + _compare(results, "schedule=parallel", "schedule=serial") + "."
    )

    ms = []
    for code_name, rows in results.items():
        values = [
            row["failure_fraction"]
            for row in rows
            if row["factor"] == "ms_scaling_factor" and not row.get("skipped")
        ]
        if values:
            ms.append(f"{code_name} {min(values):.4f}..{max(values):.4f}")
    bullets.append("- ms_scaling_factor (minimum_sum, 0.5..1.0) failure range: " + ", ".join(ms) + ".")
    return bullets[:5]


def _table(payload):
    lines = [
        "| Code | Factor | Setting | Failure fraction | Non-converged fraction |",
        "|---|---|---|---|---|",
    ]
    for code_name, rows in payload["results"].items():
        for row in rows:
            failure = "skipped" if row.get("skipped") else f"{row['failure_fraction']:.4f}"
            non_converged = "skipped" if row.get("skipped") else f"{row['non_converged_fraction']:.4f}"
            lines.append(f"| {code_name} | {row['factor']} | {row['setting']} | {failure} | {non_converged} |")
    return "\n".join(lines)


def _markdown(payload):
    meta = payload["meta"]
    caption = (
        f"Dataset: iid depolarizing shots at p={meta['p_eval']}, {meta['n_shots']} shots per code, seed={meta['seed']}; "
        f"prior fixed at 2p/3={meta['prior']:.4f}; one factor at a time around the default "
        "(max_iter=30, bp_method=product_sum, ms_scaling_factor=1.0, schedule=parallel). "
        "Failure fraction = fraction of shots failing the independent logical check; non-converged fraction = "
        "fraction of shots where either the X or the Z BP plane did not converge."
    )

    skipped = [f"{c}/{r['setting']}" for c, rows in payload["results"].items() for r in rows if r.get("skipped")]
    if skipped:
        validity = "Combos skipped as unsupported by ldpc 2.4.1: " + ", ".join(skipped) + "."
    else:
        validity = (
            "Verified with `help(ldpc.BpDecoder)` and by constructing decoders on ldpc 2.4.1: `bp_method` accepts "
            "`product_sum` and `minimum_sum`; `schedule` accepts `parallel` and `serial` (`serial_relative` also "
            "exists); `ms_scaling_factor` accepts any float. Every contract value is valid, so no combos were skipped."
        )

    lines = [
        "# Step 4: BP configuration sweep",
        "",
        caption,
        "",
        "## Validity of ldpc 2.4.1 options",
        "",
        validity,
        "",
        "## Results",
        "",
        _table(payload),
        "",
        "## Observation",
        "",
    ]
    lines.extend(observations(payload))
    lines.append("")
    return "\n".join(lines)


def write_outputs(payload, out_dir=OUT_DIR):
    """Write ``<module>.json`` and ``<module>.md`` under ``out_dir``; returns both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{MODULE_NAME}.json"
    md_path = out_dir / f"{MODULE_NAME}.md"
    json_path.write_text(json.dumps(payload, indent=1))
    md_path.write_text(_markdown(payload))
    return json_path, md_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Step 4: BP configuration sweep (issue #13).")
    parser.add_argument("--codes", nargs="+", default=list(DEFAULT_CODES))
    parser.add_argument("--n-per-weight", type=int, default=5000, help="iid shots per code")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="tiny sizes for tests (< 20 s)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    n_shots = min(args.n_per_weight, 200) if args.quick else args.n_per_weight
    payload = run(args.codes, n_shots, args.seed)
    json_path, md_path = write_outputs(payload, OUT_DIR)
    print(f"wrote {json_path} and {md_path}")
    return payload


if __name__ == "__main__":
    main()
