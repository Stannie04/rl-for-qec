from __future__ import annotations
import time
import numpy as np
from tqdm import tqdm
import torch

from src.agents import RouterAgent, RouterNetwork
from src.environment import QECEnv
from src.read_config import ConfigParser
from src.train_utils.datasets import create_dataset_from_random_shots,  create_dataset_from_uniform_shots, create_dataset_from_random_shots_labelled
from src.train_utils.inference import get_agent_and_inference, inference
from src.train_utils.plotting import plot_results


def evaluate_agent(config: ConfigParser, step, best_model_ler, agent_name=None, checkpoint_dir=None):
    # Evaluate the logical error rate of the agent over a number of episodes.
    shots = create_dataset_from_uniform_shots(config, num_samples_per_error=config.eval_episodes_per_weight, max_error=config.max_evaluation_error)

    eval_env = QECEnv(config, shots)
    agent, inference_fn = get_agent_and_inference(config, eval_env, agent_name)

    logical_failures = 0
    total_shots = len(shots)

    # LER per error weight
    successes_per_weight = [0] * config.max_evaluation_error

    for i in range(config.max_evaluation_error):
        error_weight = i + 1
        successes = [inference_fn(agent, eval_env) for _ in tqdm(range(config.eval_episodes_per_weight), desc=f"Evaluating agent at error weight {error_weight}", leave=False)]
        successes_per_weight[error_weight - 1] = sum(successes)
        logical_failures += config.eval_episodes_per_weight - successes_per_weight[error_weight - 1]

    # NOTE This is not actual LER, since we are weighting the error weights equally, but it is a good proxy for model performance.
    ler = logical_failures / total_shots

    # LER for weights 1..4
    acc_per_weight = [
        successes_per_weight[i] / config.eval_episodes_per_weight
        for i in range(config.max_evaluation_error)
    ]

    eval_string = "_".join([f"{ler:.2f}" for ler in acc_per_weight])
    print(eval_string)

    if checkpoint_dir is not None:
        agent.save(f"{checkpoint_dir}/eval_{step}_{eval_string}.pt")
        if ler < best_model_ler:
            best_model_ler = ler

    return ler, best_model_ler


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


def per_to_ler_agents(config, agents, plot_file=None):

    results = {}

    for i in range(config.n_repetitions):
        all_shots, counts = create_dataset_from_random_shots_labelled(config, config.post_training_evaluation_episodes, config.evaluation_error_rates)

        for agent_name in agents:
            failures, _, _ = split_inference_between_shot_types(config, agent_name, all_shots, task="failures")
            logical_error_rates = get_weighted_ler(failures, counts)
            key = f"{agent_name}_{i+1}" if config.n_repetitions > 1 else agent_name
            results[key] = {config.evaluation_error_rates[i]: logical_error_rates[i] for i in range(len(config.evaluation_error_rates))}
            print(f"\nResults for {agent_name}: {results[key]}\n")

    if plot_file is not None:
        plot_results(results, config, plot_file)


def generalizability_evaluation(config, agents):

    results = {}
    codes = ["288_2_12_toric"]
    for code in codes:
        config = ConfigParser("configs", config.agent_name, code, run_name=config.wandb_run_name, verbose=config.verbose)
        results[code] = per_to_ler_agents(config, agents)

    return results


def post_train_evaluation(config):
    # Evaluate the agent at the end of training and print results to console.
    per_to_ler_agents(config, config.agents_to_evaluate, plot_file=f"results/per_to_ler/{config.code_name}_bptest.png")
