import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
import torch
from src.environment import QECEnv
from src.train_utils.datasets import load_shots
from src.train_utils.inference import get_agent_and_inference, inference
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


def split_inference_between_shot_types(config, agent_name, unique_shots, task="failures", checkpoint=None):
    x_patterns = unique_shots[:, 0, :]
    z_patterns = unique_shots[:, 1, :]
    unique_x, inverse_x = np.unique(x_patterns, axis=0, return_inverse=True)
    unique_z, inverse_z = np.unique(z_patterns, axis=0, return_inverse=True)

    x_shots = np.zeros((len(unique_x), 2, config.n), dtype=np.int8)
    z_shots = np.zeros((len(unique_z), 2, config.n), dtype=np.int8)

    x_shots[:, 0, :] = unique_x
    z_shots[:, 1, :] = unique_z

    combined_failures = inference(agent_name, config, np.concatenate([x_shots, z_shots], axis=0), task=task, checkpoint=checkpoint).reshape(-1)

    n_x = len(unique_x)
    x_fail_unique = combined_failures[:n_x]
    z_fail_unique = combined_failures[n_x:]

    # Map the results back to the original shots
    x_failures = x_fail_unique[inverse_x]
    z_failures = z_fail_unique[inverse_z]
    full_failures = x_failures | z_failures
    return full_failures, x_failures, z_failures





def get_weighted_ler(failures, counts):
    full_failures = np.asarray(failures, dtype=bool)
    failed_counts = counts[full_failures].sum(0)
    total_counts = counts.sum(0)
    return np.divide(failed_counts, total_counts, out=np.zeros(len(total_counts), dtype=float), where=total_counts != 0)


def get_weighted_ler_on_eval_set(config, agent, checkpoint, eval_set):
    all_shots, counts = eval_set
    agent.save(checkpoint)
    failures, _, _ = split_inference_between_shot_types(config, "rl_hpo", all_shots, checkpoint=checkpoint, task="failures")
    lers = get_weighted_ler(failures, counts)
    floor = 0.5 / np.maximum(counts.sum(0), 1)
    return float(np.log10(np.maximum(lers, floor)).mean()) # Every rate counts equally in relative terms



def plot_oracle_mistakes(config, all_mistakes):
    """
    Plot the number of oracle mistakes for every pair of agents.

    Oracle(A,B) is assumed to be correct whenever either model is correct,
    so the remaining mistakes are those made by both models.
    """

    all_shots, counts = load_shots(config, dataset_type="random_labelled")

    num_shots = len(all_shots)
    packed_shots = np.packbits(all_shots.reshape(num_shots, -1), axis=1)
    shot_keys = [shot.tobytes() for shot in packed_shots]

    # Select the highest evaluation error rate.
    error_rates = list(config.evaluation_error_rates)
    rate_idx = int(np.argmax(error_rates))
    selected_rate = error_rates[rate_idx]

    # Build a set of mistake keys for each available agent.
    agent_names = [name for name, mistakes in all_mistakes.items()if mistakes is not None]
    mistake_keys = {}

    for name in agent_names:
        mistakes = np.asarray(all_mistakes[name])
        if len(mistakes) == 0:
            mistake_keys[name] = set()
            continue
        packed_mistakes = np.packbits(mistakes.reshape(len(mistakes), -1), axis=1)
        mistake_keys[name] = {shot.tobytes() for shot in packed_mistakes}

    # Compute each agent's failure mask on the evaluation dataset.
    failure_masks = {name: np.fromiter((key in mistake_keys[name] for key in shot_keys),dtype=bool,count=num_shots) for name in agent_names}

    n = len(agent_names)
    oracle = np.zeros((n, n), dtype=float)

    # Compute weighted LER for every pair.
    for i, a in enumerate(agent_names):
        for j, b in enumerate(agent_names):
            # The oracle fails only if both agents fail.
            failures = failure_masks[a] & failure_masks[b]

            ler = get_weighted_ler(failures, counts)
            oracle[i, j] = ler[rate_idx]

    # Plot weighted LER values, not raw mistake counts.
    plt.figure(figsize=(9, 7))

    sns.heatmap(
        oracle,
        cmap="viridis",
        annot=True,
        fmt=".2e",
        xticklabels=agent_names,
        yticklabels=agent_names,
        linewidths=0.8,
        linecolor="white",
        square=True,
        cbar_kws={
            "label": f"Weighted LER (p = {selected_rate:g})",
            "shrink": 0.8
        }
    )

    plt.xlabel("")
    plt.ylabel("")
    plt.title(f"Oracle weighted LER at p = {selected_rate:g}")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0)

    plt.tight_layout()

    output_dir = (
        f"results/mistake_distributions/"
        f"{config.code_name}/{config.noise_model}"
    )
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(
        f"{output_dir}/oracle_heatmap.png",
        dpi=300
    )
    plt.close()