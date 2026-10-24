"""Observation O1 figure: weight-1 BP failure fraction per Pauli vs the F1 ratio (issue #13)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/bp_debug/step2_weight1.json"))
rows = {"d7_surface": (12, 12, 24, 49, 0.33), "288_8_12_tile": (20, 20, 40, 288, 0.11), "144_12_12_ldpc": (0, 0, 0, 144, 0.0)}
fig, ax = plt.subplots(figsize=(7, 4))
w = 0.2
for i, (c, (x, z, y, n, ratio)) in enumerate(rows.items()):
    f = [x / n, y / n, z / n]
    mean = sum(f) / 3
    for j, (lab, v) in enumerate(zip("XYZ", f)):
        ax.bar(i + (j - 1) * w, v, w, color=["C0", "C1", "C2"][j], label=lab if i == 0 else None)
    ax.plot([i - 0.4, i + 0.4], [mean, mean], "k--", lw=1, label="mean over X,Y,Z" if i == 0 else None)
    ax.plot(i + 0.33, ratio, "r*", ms=11, label="LER / P(>=1 error) from F1" if i == 0 else None)
ax.set_xticks(range(3)); ax.set_xticklabels(rows)
ax.set_ylabel("fraction of single-qubit errors that BP fails")
ax.set_title("O1: weight-1 BP failures explain the F1 ratio")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("results/bp_debug/o1_weight1_vs_f1.png", dpi=140)
