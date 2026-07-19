from __future__ import annotations
import time
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable

from src.agents import SACAgent, CGNNEncoder, RouterAgent
from src.environment import QLDPCEnv
from src.read_config import ConfigParser
from src.train_utils.datasets import create_dataset_from_random_shots,  create_dataset_from_uniform_shots, create_dataset_from_random_shots_labelled
from src.train_utils.inference import get_agent_and_inference, parallel_inference
from src.train_utils.plotting import plot_results


def benchmark_env(config):
    start = time.time()
    env = QLDPCEnv(config)

    if config.verbose:
        env.render(mode="edge_info")

    agent = SACAgent(env, config)
    end = time.time()
    print(f"Initialization took {end - start:.5f} seconds")

    obs, info = env.reset()

    step_times = []
    agent_times = []
    buffer_times = []
    train_times = []
    loop_times = []
    for _ in tqdm(range(10_000), desc="Benchmarking environment and agent"):

        loop_start = time.time()
        action, _ = agent.select_action(obs)
        end = time.time()
        agent_times.append(end - loop_start)

        start = time.time()
        next_obs, reward, terminated, truncated, info = env.step(action)
        end = time.time()
        step_times.append(end - start)

        start = time.time()
        agent.replay_buffer.push(obs, action, reward, next_obs, terminated or truncated)
        end = time.time()
        buffer_times.append(end - start)

        start = time.time()
        agent.train_step()
        loop_end = time.time()
        train_times.append(loop_end - start)

        obs = next_obs
        loop_times.append(loop_end - loop_start)


    t = PrettyTable(["Component", "Avg Time (s)", "it/s"])

    t.add_row(["Environment Step", f"{np.mean(step_times[1:]):.5f}", f"{1/np.mean(step_times[1:]):.2f}"])
    t.add_row(["Agent Action Selection", f"{np.mean(agent_times[1:]):.5f}", f"{1/np.mean(agent_times[1:]):.2f}"])
    t.add_row(["Buffer Push", f"{np.mean(buffer_times[1:]):.5f}", f"{1/np.mean(buffer_times[1:]):.2f}"])
    t.add_row(["Agent Training Step", f"{np.mean(train_times[1:]):.5f}", f"{1/np.mean(train_times[1:]):.2f}"])
    t.add_row(["Total Loop", f"{np.mean(loop_times[1:]):.5f}", f"{1/np.mean(loop_times[1:]):.2f}"])

    print(t)



def evaluate_agent(config: ConfigParser, step, best_model_ler, agent_name=None, checkpoint_dir=None):
    # Evaluate the logical error rate of the agent over a number of episodes.

    num_samples_per_error = 1000
    max_error = 4

    shots = create_dataset_from_uniform_shots(
        config,
        num_samples_per_error=num_samples_per_error,
        max_error=max_error,
        noise_model="bit_flip",
    )

    eval_env = QLDPCEnv(config, shots)
    agent, _ = get_agent_and_inference(config, eval_env, agent_name)

    logical_failures = 0
    total_shots = len(shots)

    # LER per error weight
    successes_per_weight = [0] * max_error

    for i in range(max_error):
        error_weight = i + 1

        for _ in tqdm(range(num_samples_per_error), desc=f"Evaluating agent at error weight {error_weight}", leave=False):
            obs, info = eval_env.reset()
            done = info["error_free"]

            while not done:
                action, _ = agent.select_action(obs, evaluate=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated

            if info["error_free"]:
                successes_per_weight[error_weight - 1] += 1

    ler = logical_failures / total_shots

    # LER for weights 1..4
    acc_per_weight = [
        successes_per_weight[i] / num_samples_per_error
        for i in range(max_error)
    ]

    print(f"{acc_per_weight[0]}, {acc_per_weight[1]}, {acc_per_weight[2]}, {acc_per_weight[3]}")

    if checkpoint_dir is not None:
        filename = (
            f"{checkpoint_dir}/eval_{step}_"
            f"{acc_per_weight[0]:.2f}_"
            f"{acc_per_weight[1]:.2f}_"
            f"{acc_per_weight[2]:.2f}_"
            f"{acc_per_weight[3]:.2f}.pt"
        )
        agent.save(filename)

        if ler < best_model_ler:
            best_model_ler = ler

    return ler, best_model_ler


def per_to_ler_router(config):
    results = {"BP": {}, "Neural BP": {}, "Router": {}, "Optimal Router": {}}


    for error_rate in [0.01, 0.0075, 0.005, 0.0025, 0.001]:
        shots = create_dataset_from_random_shots(config, config.post_training_evaluation_episodes, error_rate, noise_model="bit_flip")

        moe_env = QLDPCEnv(config, shots)
        bp_env = QLDPCEnv(config, shots)
        neural_env = QLDPCEnv(config, shots)

        router = RouterAgent(config, moe_env, router_checkpoint="checkpoints/evaluate_cps/router.pt")
        bp_agent, bp_inference = get_agent_and_inference(config, bp_env, "bp")
        neural_agent, neural_inference = get_agent_and_inference(config, neural_env, "sac_nbp_big")

        logical_failures_bp, logical_failures_neural, logical_failures_both, logical_failures_router = 0, 0, 0, 0

        for _ in tqdm(bp_env.shots, desc=f"Evaluating at error rate {error_rate}", leave=False):

            obs, info = moe_env.reset()

            if info["error_free"]:
                # If the environment is already error-free, we can skip this episode since both agents will succeed.
                _, _ = bp_env.reset()
                _, _ = neural_env.reset()
                continue  # Skip if the environment is already error-free

            router_pred, _ = router.select_action(obs)

            # Since the environments are reset with the same shots, we can evaluate both agents on the same error instance.
            bp_success = bp_inference(bp_agent, bp_env)
            neural_success = neural_inference(neural_agent, neural_env)

            logical_failures_bp += not bp_success
            logical_failures_neural += not neural_success
            logical_failures_both += not (bp_success or neural_success)
            logical_failures_router += not (bp_success, neural_success)[router_pred]

        results["BP"][error_rate] = logical_failures_bp / config.post_training_evaluation_episodes
        results["Neural BP"][error_rate] = logical_failures_neural / config.post_training_evaluation_episodes
        results["Router"][error_rate] = logical_failures_router / config.post_training_evaluation_episodes
        results["Optimal Router"][error_rate] = logical_failures_both / config.post_training_evaluation_episodes

    return results


def per_to_ler_agents(config, agents):

    error_rates = [0.01, 0.0075, 0.005, 0.0025, 0.001]
    results = {}

    for i in range(config.n_repetitions):
        all_shots = create_dataset_from_random_shots_labelled(config, config.post_training_evaluation_episodes, error_rates, noise_model="bit_flip")

        for agent_name in agents:

            shots, counts = all_shots
            failures = parallel_inference(agent_name, config, shots, task="failures", repetition=i+1)
            logical_error_rates = sum(counts[np.where(failures)]) / config.post_training_evaluation_episodes

            results[f"{agent_name}_{i+1}"] = {error_rates[i]: logical_error_rates[i] for i in range(len(error_rates))}
            print(f"\nResults for {agent_name}: {results[f'{agent_name}_{i+1}']}\n")

    return results



def generalizability_evaluation(config, agents):

    results = {}
    codes = ["288_2_12_toric"]
    for code in codes:
        config = ConfigParser("configs", config.agent_name, code, run_name=config.wandb_run_name, verbose=config.verbose)
        results[code] = per_to_ler_agents(config, agents)

    return results


def post_train_evaluation(config):
    # Evaluate the agent at the end of training and print results to console.
    agents = ["rl_cgnn", "sl_cgnn", "sl_nbp", "rl_nbp", "bp", "bp_osd"]


    results = per_to_ler_agents(config, agents)
    print(results)
    plot_results(results, config,f"results/{config.agent_name}_{config.code_name}_agents_evaluation.png")

    results = per_to_ler_router(config)
    plot_results(results, config,f"results/{config.agent_name}_{config.code_name}_moe_evaluation.png")

    generalizability_evaluation(config, agents)