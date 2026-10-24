"""Observation O8 check: do the weight-1 BP failures coincide with identical columns in H? (issue #13)

For X errors the decoder uses H_z, for Z errors H_x. Two qubits with identical (non-zero) columns give BP the same
likelihood for either of them, a degenerate tie. Compare the set of such qubits with the step-2 failing sets.
"""
import json
import numpy as np
from scripts.bp_debug import common

res = json.load(open("results/bp_debug/step2_weight1.json"))["codes"]
for code_name in ["d7_surface", "288_8_12_tile", "144_12_12_ldpc"]:
    cfg = common.load_config(code_name, agent_name="sac")
    env = common.make_env(cfg)
    code = env.code if hasattr(env, "code") else env.qec_code
    H = {"X": common._as_bits(code.H.z), "Z": common._as_bits(code.H.x)}
    for pauli, M in H.items():
        cols = [M[:, j].tobytes() for j in range(M.shape[1])]
        dup = {j for j, c in enumerate(cols) if M[:, j].any() and cols.count(c) > 1}
        fail = set(res[code_name]["paulis"][pauli]["repo"]["failing_qubits"])
        print(f"{code_name} {pauli}: dup-column qubits={len(dup)} failing={len(fail)} "
              f"both={len(dup & fail)} dup_not_failing={sorted(dup - fail)[:8]} failing_not_dup={sorted(fail - dup)[:8]}")
