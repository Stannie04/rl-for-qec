"""Observation O4 figure: BP config sweep, failure fraction vs max_iter and ms_scaling_factor (issue #13)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

it = [10, 30, 100, 500]
ms = [0.5, 0.625, 0.75, 0.9, 1.0]
d = {"d7_surface": ([0.0484] * 4, [0.0522, 0.0484, 0.0484, 0.0484, 0.0484]),
     "288_8_12_tile": ([0.0776] * 4, [0.1612, 0.0784, 0.0780, 0.0776, 0.0774])}
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
for (c, (y1, y2)), col in zip(d.items(), ["C0", "C1"]):
    a.plot(it, y1, "o-", color=col, label=c)
    b.plot(ms, y2, "o-", color=col, label=c)
a.set_xscale("log"); a.set_ylim(0, 0.12); a.set_xlabel("max_iter"); a.set_ylabel("shot failure fraction, p=0.003")
a.set_title("50x more iterations: no change")
b.set_xlabel("min-sum ms_scaling_factor"); b.set_ylim(0, 0.18); b.set_title("min-sum scaling: no gain, worse at 0.5")
a.legend(); b.legend(); fig.tight_layout(); fig.savefig("results/bp_debug/o4_config_sweep.png", dpi=140)
