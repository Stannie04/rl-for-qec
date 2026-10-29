from __future__ import annotations
import time
import numpy as np
from tqdm import tqdm
import torch

from src.agents import RouterAgent, RouterNetwork
from src.environment import QECEnv
from src.read_config import ConfigParser
from src.train_utils.datasets import create_dataset_from_random_shots,  create_dataset_from_uniform_shots, create_dataset_from_random_shots_labelled, load_shots
from src.train_utils.inference import get_agent_and_inference, inference
from src.train_utils.plotting import plot_results, plot_oracle_mistakes, get_weighted_ler, split_inference_between_shot_types
from src.train_utils.code_analysis import compute_all_mistake_overlaps


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


def evaluate_from_dataset(config, plot_file=None):
    all_shots, counts = load_shots(config, dataset_type="random_labelled")
    mistakes = {agent: load_shots(config, dataset_type="mistakes", agent_name=agent) for agent in config.agents_to_evaluate}

    num_shots = len(all_shots)
    packed_shots = np.packbits(all_shots.reshape(num_shots, -1), axis=1)

    results = {}
    for agent_name in config.agents_to_evaluate:
        agent_mistakes = mistakes[agent_name]
        if agent_mistakes is None:
            continue
        if len(agent_mistakes) == 0:
            failures = np.zeros(num_shots, dtype=bool)
        else:
            packed_mistakes = np.packbits(agent_mistakes.reshape(len(agent_mistakes), -1), axis=-1)
            packed_mistakes_bytes = {shot.tobytes() for shot in packed_mistakes}
            failures = np.fromiter((shot.tobytes() in packed_mistakes_bytes for shot in tqdm(packed_shots, desc=f"Evaluating {agent_name}", leave=False)), dtype=bool, count=num_shots)

        ler = get_weighted_ler(failures, counts)

        results[agent_name] = {config.evaluation_error_rates[i]: ler[i] for i in range(len(config.evaluation_error_rates))}
        print(f"\nResults for {agent_name}: {results[agent_name]}")

    # compute_all_mistake_overlaps(config, mistakes)
    plot_oracle_mistakes(config, mistakes)

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
    # per_to_ler_agents(config, config.agents_to_evaluate, plot_file=f"results/per_to_ler/{config.code_name}.png")
    evaluate_from_dataset(config, plot_file=f"results/per_to_ler/{config.code_name}.png")