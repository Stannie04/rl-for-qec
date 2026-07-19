import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
import torch
from src.environment import QLDPCEnv
from src.train_utils.datasets import load_shots
from src.train_utils.inference import get_agent_and_inference
from collections import Counter, defaultdict
from matplotlib.colors import LogNorm
import seaborn as sns

def plot_results(results, config, path, title=None, with_error_bars=False):

    plt.style.use("seaborn-v0_8-whitegrid")

    colors = {"BP": "#0072B2","SL": "#D55E00", "SAC": "#009E73", "MWPM": "#0072B2","No Agent": "#999999", "Router": "#D55E00", "Optimal Router": "#E69F00"}
    linestyles = {"BP": "-", "NBP": ":", "CGNN": "-", "MWPM": "-", "No Agent": "-", "Router": "-", "Optimal Router": "-"}
    markers = {"BP": "o","NBP": "s", "CGNN": "^", "MWPM": "o", "No Agent": "o", "Router": "^", "Optimal Router": "^"}

    title = title or f"Logical Error Rate vs Physical Error Rate for {config.code_name}"

    fig, ax = plt.subplots(figsize=(8, 6))

    for agent_name, agent_results in results.items():
        error_rates = sorted(agent_results.keys())
        logical_error_rates = [agent_results[e][0] for e in error_rates]
        std = [agent_results[e][1] for e in error_rates]

        if agent_name in ["BP", "MWPM", "No Agent", "Router", "Optimal Router"]:
            color = colors[agent_name]
            linestyle = "-"
            marker = markers[agent_name]
        else:
            family = "SL" if agent_name.startswith("SL") else "SAC"
            method = "NBP" if "NBP" in agent_name else "CGNN"

            color = colors[family]
            linestyle = linestyles[method]
            marker = markers[method]

        ax.errorbar(
            error_rates,
            logical_error_rates,
            yerr=std,  # <-- error bars here
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.5,
            markersize=7,
            capsize=4,  # width of error bar caps
            capthick=1.5,
            label=agent_name
        )


    ax.set_yscale("log")
    ax.set_xlabel(r"Physical error rate $p$", fontsize=14)
    ax.set_ylabel(r"Logical error rate $p_L$", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.tick_params(labelsize=12)

    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(fontsize=12)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed
    window: size of the smoothing window '''

    return savgol_filter(y,window,poly) if window > 0 else y


def get_confidence_bounds(results, window=100):
    """
    Calculates smoothed mean and 95% confidence intervals for the results.
    """

    # Convert to list of arrays
    results = [np.array(r) for r in results]

    if len(results) == 1:
        return smooth(results[0], window=window), np.zeros_like(results[0])

    # Find minimum length
    min_len = min(len(r) for r in results)

    # Truncate all runs
    results = np.array([r[:min_len] for r in results])

    mean = smooth(np.mean(results, axis=0), window=window)
    ci95 = 1.96 * np.std(results, axis=0) / np.sqrt(len(results))

    return mean, smooth(ci95, window=window)


def render_example_environment(config):
    env = QLDPCEnv(config)

    for l in env.code.logical_x:
        initial_x, initial_z = env.code.get_logical_state()
        num = len(torch.argwhere(l == 1).flatten())
        for i, j in enumerate(torch.argwhere(l == 1).flatten()):
            print(f"{i+1} / {num}")
            env.code.flip(j)
            env.code.update_graph(env.curriculum_error_rate)

            current_x, current_z = env.code.get_logical_state()
            x_changed = not torch.equal(initial_x, current_x)
            z_changed = not torch.equal(initial_z, current_z)
            print(f"Flipped qubit {j.item()}: Logical X changed: {x_changed}, Logical Z changed: {z_changed}")


            print(f"Error free: {env.code.is_error_free()}\n")

        env.render()

        env.reset()


def render_mistakes(config):
    agent_name = "bp"

    mistakes = load_shots(config, dataset_type="mistakes", noise_model="bit_flip", agent_name=agent_name)
    env = QLDPCEnv(config, mistakes)
    agent, _ = get_agent_and_inference(config, env, agent_name)

    for idx, shot in enumerate(mistakes):
        print(f"Rendering mistake {idx + 1}/{len(mistakes)}")
        obs, info = env.reset_with_error_pattern(shot[0], shot[1])

        # Evaluate the agent on this mistake
        error_pred = agent.select_action(obs, evaluate=True)[0]
        print(f"Action: {error_pred}")
        if len(error_pred) > 0:
            # env.code.render_subgraph()
            env.render()
            for a in error_pred:
                env.step(a)
            env.code.render_subgraph()


def plot_jaccard_heatmap(agent_names, jaccard):

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        jaccard,
        cmap="Reds",
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(np.arange(len(agent_names)))
    ax.set_yticks(np.arange(len(agent_names)))

    ax.set_xticklabels(agent_names, rotation=45, ha="right")
    ax.set_yticklabels(agent_names)

    # annotate cells
    for i in range(len(agent_names)):
        for j in range(len(agent_names)):
            ax.text(
                j,
                i,
                f"{jaccard[i,j]:.2f}",
                ha="center",
                va="center",
                color="white" if jaccard[i,j] < 0.6 else "black",
                fontsize=10,
            )

    cbar = plt.colorbar(im)
    cbar.set_label("Jaccard similarity")

    ax.set_title("Pairwise Jaccard Similarity of Mistake Sets")

    plt.tight_layout()
    plt.show()


def plot_oracle_mistakes(all_mistakes):
    """
    Plot the number of oracle mistakes for every pair of agents.

    Oracle(A,B) is assumed to be correct whenever either model is correct,
    so the remaining mistakes are those made by both models.
    """

    def mistake_counter(mistakes):
        """Count occurrences of each mistake."""
        mistakes = np.ascontiguousarray(mistakes)
        flat = mistakes.reshape(mistakes.shape[0], -1)
        return Counter(row.tobytes() for row in flat)

    new_names = {
        "bp": "BP",
        "sac_nbp_big": "SAC+NBP",
        "sac_tanner_big": "SAC+CGNN",
        "sl_nbp_big": "SL+NBP",
        "sl_tanner_big": "SL+CGNN",
    }
    all_mistakes = {new_names.get(k, k): v for k, v in all_mistakes.items()}

    agent_names = list(all_mistakes.keys())
    counters = {
        name: mistake_counter(mistakes)
        for name, mistakes in all_mistakes.items()
    }

    n = len(agent_names)
    oracle = np.zeros((n, n), dtype=int)

    for i, a in enumerate(agent_names):
        for j, b in enumerate(agent_names):

            if i == j:
                oracle[i, j] = len(all_mistakes[a])
                continue

            # Multiset intersection
            shared = 0
            keys = counters[a].keys() & counters[b].keys()
            for k in keys:
                shared += min(counters[a][k], counters[b][k])

            oracle[i, j] = shared

    sns.heatmap(
        oracle,
        cmap="viridis",
        norm=LogNorm(vmin=max(1, oracle.min()), vmax=oracle.max()),
        annot=True,
        fmt=",",
        xticklabels=agent_names,
        yticklabels=agent_names,
        linewidths=0.8,
        linecolor="white",
        square=True,
        cbar_kws={
            "label": "Remaining mistakes",
            "shrink": 0.8
        }
    )

    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig("results/oracle_heatmap.png", dpi=300)