"""Observation O5 figure: BP failure vs evaluated p for several priors (issue #13)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ps = [0.001, 0.002, 0.003, 0.004, 0.005]
cur = {
    "d7_surface": {"matched 2p/3": [0.0159, 0.0343, 0.0481, 0.0626, 0.0750], "prior 0.1": [0.0159, 0.0343, 0.0481, 0.0626, 0.0750]},
    "288_8_12_tile": {"matched 2p/3": [0.0283, 0.0537, 0.0762, 0.0945, 0.1281], "prior 0.1": [0.0283, 0.0539, 0.0767, 0.0949, 0.1287]},
}
fig, ax = plt.subplots(figsize=(6.5, 4))
for (c, d), col in zip(cur.items(), ["C0", "C1"]):
    ax.plot(ps, d["matched 2p/3"], "o-", color=col, label=f"{c}, matched prior 2p/3")
    ax.plot(ps, d["prior 0.1"], "x--", color=col, label=f"{c}, prior 0.1 (up to 150x off)")
ax.set_xlabel("evaluated p"); ax.set_ylabel("BP shot failure fraction"); ax.set_title("O5: the prior does not matter")
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig("results/bp_debug/o5_prior_sweep.png", dpi=140)
