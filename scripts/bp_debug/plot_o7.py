"""Observation O7 figure: where the BP failures come from, by (n_x, n_z) bin class, vs p (issue #13)."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

text = open("results/bp_debug/step7_stratified.md").read()
ps = [0.001, 0.002, 0.003, 0.004, 0.005]
single = {(0, 1), (1, 0), (1, 1)}  # at most one qubit hit per plane
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, code in zip(axes, ["d7_surface", "288_8_12_tile"]):
    sec = text.split(f"## {code}")[1].split("\n## ")[0]
    classes = {"one flip per plane: (0,1) (1,0) (1,1)": [0] * 5, "two or more flips in a plane": [0] * 5}
    for m in re.finditer(r"^\| (\(\d,\d\)|5\+) \|(.*)$", sec, re.M):
        b = m.group(1)
        key = tuple(map(int, b.strip("()").split(","))) if b != "5+" else None
        for i, c in enumerate(m.group(2).split("|")[:5]):
            f = re.search(r"(\d+)/(\d+)", c)
            if not f or key == (0, 0):
                continue
            k = "one flip per plane: (0,1) (1,0) (1,1)" if key in single else "two or more flips in a plane"
            classes[k][i] += int(f.group(1))
    tot = [a + b for a, b in zip(*classes.values())]
    bottom = [0] * 5
    for k, v in classes.items():
        share = [x / t for x, t in zip(v, tot)]
        ax.bar(range(5), share, bottom=bottom, label=k)
        bottom = [a + b for a, b in zip(bottom, share)]
    ax.set_xticks(range(5)); ax.set_xticklabels(ps); ax.set_xlabel("p"); ax.set_title(code)
axes[0].set_ylabel("share of all BP failures"); axes[0].legend(fontsize=7, loc="lower left")
fig.suptitle("O7: low-weight shots dominate BP failures at small p"); fig.tight_layout()
fig.savefig("results/bp_debug/o7_failure_share.png", dpi=140)
