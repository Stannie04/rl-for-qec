"""Observation O2 figure: failure classes at p=0.003 and the F1-style ratio (issue #13)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = 0.003
rows = {"d7_surface": (49, 242, 5000), "288_8_12_tile": (288, 388, 5000), "144_12_12_ldpc": (144, 0, 5000)}
fig, ax = plt.subplots(figsize=(7, 4))
for i, (c, (n, fail, shots)) in enumerate(rows.items()):
    any_err = 1 - (1 - p) ** n
    ax.bar(i - 0.2, any_err, 0.4, color="lightgray", label="P(>=1 error)" if i == 0 else None)
    ax.bar(i + 0.2, fail / shots, 0.4, color="C3", label="BP fails, all non-converged" if i == 0 else None)
    ax.text(i + 0.2, fail / shots + 0.01, f"{fail / shots / any_err:.2f}" + " of any-error", ha="center", fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels(rows)
ax.set_ylabel("per-shot probability, p=0.003, 5000 shots")
ax.set_title("O2: every BP failure is non-convergence")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("results/bp_debug/o2_failure_class.png", dpi=140)
