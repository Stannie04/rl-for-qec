"""Observation O3 figure: MWPM vs BP on iid shots and uniform weight 1-3 (issue #13)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ps = [0.001, 0.003, 0.005]
bp = {"d7_surface": [0.0162, 0.0484, 0.0756], "d5_surface": [0.0108, 0.0364, 0.0532]}
w = [1, 2, 3]
bpw = {"d7_surface": [0.4368, 0.7448, 0.9072], "d5_surface": [0.5416, 0.8698, 0.9594]}
mw = {"d7_surface": [0, 0, 0], "d5_surface": [0, 0, 0.2248]}
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
for c, col in zip(bp, ["C0", "C1"]):
    a.plot(ps, bp[c], "o-", color=col, label=f"BP {c}")
    a.plot(ps, [1e-4] * 3, "x--", color=col, label=f"MWPM {c} (0 / 5000, plotted at 1e-4)")
    b.plot(w, bpw[c], "o-", color=col, label=f"BP {c}")
    b.plot(w, mw[c], "x--", color=col, label=f"MWPM {c}")
a.set_yscale("log"); a.set_xlabel("p"); a.set_ylabel("shot failure fraction"); a.set_title("iid depolarizing, 5000 shots")
b.set_xlabel("nominal weight w (w X + w Z flips)"); b.set_ylabel("shot failure fraction"); b.set_title("uniform shots")
a.legend(fontsize=7); b.legend(fontsize=7); fig.tight_layout(); fig.savefig("results/bp_debug/o3_mwpm_vs_bp.png", dpi=140)
