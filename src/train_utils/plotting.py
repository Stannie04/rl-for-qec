import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
import torch
from src.environment import QECEnv
from src.train_utils.datasets import load_shots
from src.train_utils.inference import get_agent_and_inference
from collections import Counter, defaultdict
from matplotlib.colors import LogNorm
import seaborn as sns
import os
import json
from tqdm import tqdm

def plot_results(results, config, path, title=None, with_error_bars=False):

    # Dump the results to a json file for later analysis
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json_path = os.path.splitext(path)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    title = title or f"Logical Error Rate vs Physical Error Rate for {config.code_name}"
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    for agent_name, agent_results in results.items():
        error_rates = sorted(agent_results.keys())

        if with_error_bars:
            logical_error_rates = [agent_results[e][0] for e in error_rates]
            std = [agent_results[e][1] for e in error_rates]
        else:
            logical_error_rates = [agent_results[e] for e in error_rates]
            std = None

        agent_config = config.plot_config[agent_name]

        ax.errorbar(
            error_rates,
            logical_error_rates,
            yerr=std,  # <-- error bars here
            color=agent_config["color"],
            linestyle=agent_config["linestyle"],
            marker=agent_config["marker"],
            linewidth=1.5,
            markersize=7,
            capsize=4,  # width of error bar caps
            capthick=1.5,
            label=agent_config["label"]
        )


    ax.set_yscale("log")
    ax.set_xlabel(r"Physical error rate $p$", fontsize=14)
    ax.set_ylabel(r"Logical error rate $p_L$", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.tick_params(labelsize=12)

    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(fontsize=12)

    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    env = QECEnv(config)
    for type, correction in [(env.code.logical.x, 0), (env.code.logical.z, env.code.n_data)]:

        for l in type:
            print(f"Number of qubits: {len(torch.argwhere(l).flatten())}")

            for i, j in enumerate(torch.argwhere(l).flatten()):
                env.code.flip(j.item() + correction)
                env.code.update_graph(env.curriculum_error_rate)

            env.render()
            env.reset()


def render_mistakes(config):
    agent_name = "rl_nbp"

    mistakes = load_shots(config, dataset_type="mistakes", agent_name=agent_name)

    env = QECEnv(config, mistakes)
    agent, _ = get_agent_and_inference(config, env, agent_name)

    for idx, shot in enumerate(mistakes):
        correct_actions = np.where(shot > 0)[1] + (np.where(shot > 0)[0] * env.code.n_data)
        actions = []

        obs, info = env.reset_with_error_pattern(shot[0], shot[1])
        done = info["syndrome_free"]
        while not done:
        # Evaluate the agent on this mistake
            error_pred, _ = agent.select_action(obs, evaluate=True)
            actions.append(error_pred.item())
            obs, reward, terminated, truncated, info = env.step(error_pred)
            done = terminated or truncated

        print(f"Correct {correct_actions} | Taken: {actions}")



def plot_oracle_mistakes(config, all_mistakes, agents):
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

    all_mistakes = {agents.get(k, k): v for k, v in all_mistakes.items()}

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
    os.makedirs(f"results/mistake_distributions/{config.code_name}/{config.noise_model}", exist_ok=True)
    plt.savefig(f"results/mistake_distributions/{config.code_name}/{config.noise_model}/oracle_heatmap.png", dpi=300)