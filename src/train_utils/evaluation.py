from __future__ import annotations
import time
import torch
import numpy as np
from sympy import logcombine
from tqdm import tqdm
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from src.agents import RandomAgent, SilentAgent, DQNAgent, SACAgent, BPAgent, BPOSDAgent
from src.environment import QLDPCEnv
from src.read_config import ConfigParser
from src.train_utils.datasets import create_dataset_from_random_shots, load_shots, create_dataset_from_uniform_shots


def get_agent(config, env, agent_name):
    if type(agent_name) == str:
        match agent_name:
            case "sac":
                agent = SACAgent(env, config)
                checkpoint_name = f"{config.agent_name}_{config.code_name}" if config.wandb_run_name is None else f"{config.agent_name}_{config.code_name}_{config.wandb_run_name}"
                checkpoint = torch.load(f"checkpoints/{checkpoint_name}.pt", map_location=config.device)
                agent.actor.load_state_dict(checkpoint["actor"])
                agent.critic1.load_state_dict(checkpoint["critic1"])
                agent.critic2.load_state_dict(checkpoint["critic2"])
                return agent
            case "bp":
                return BPAgent(env, config)
            case "bp_osd":
                return BPOSDAgent(env, config)
            case "static":
                return SilentAgent(env, config)
            case _:
                raise NotImplementedError
    else:
        return agent_name



def render_example_environment(config):
    env = QLDPCEnv(config)

    for l in env.code.logical_x:
        for j in torch.argwhere(l == 1).flatten():
            env.code.flip(j)
            env.code.update_graph(env.curriculum_error_rate)

            print(f"Logical error: {env.code.has_logical_error()}")
            print(f"Error free: {env.code.is_error_free()}\n")

            env.render()

        env.reset()


def render_evaluation_episode(config, model_checkpoint, max_episode_steps=100):

    env = QLDPCEnv(config)
    agent = DQNAgent(env, config, evaluation_mode=True)
    agent.model.load_state_dict(torch.load(model_checkpoint, map_location=config.device))

    obs, info = env.reset()
    env.render()

    for step in range(max_episode_steps):
        action, _ = agent.model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step+1}: Reward={reward}, Terminated={terminated}, Truncated={truncated}, Info={info}")
        print(f"Action taken: {action}")
        env.render()

        if terminated or truncated:
            print(f"Episode finished after {step+1} steps with reward {reward} and info {info}")
            return

    print("Episode finished without termination or truncation.")


def run_baselines(config):

    silent_agent = SilentAgent(config)
    random_agent = RandomAgent(config)

    results = {}
    for agent, name in [(silent_agent, "Silent Agent"), (random_agent, "Random Agent")]:
        env = QLDPCEnv(config)

        total_rewards = []
        for i in tqdm(range(10_000), desc=f"Evaluating {name}"):
            obs, info = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.select_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

            total_rewards.append(total_reward)

        results[name] = total_rewards

    return results


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
    # Evaluate the logical error rate of the agent over a number of episodes, without exploration noise.

    # print("\n", "="*20, f"Evaluating Agent: {agent_name.upper()}", "="*20, "\n")

    # NOTE: we create 10 times the number of samples we need for evaluation to ensure we have enough unique episodes, since some episodes may be duplicates due to the random sampling process.
    # The sampling process is fast enough that this does not cause a significant slowdown.
    shots = create_dataset_from_uniform_shots(config, num_samples_per_error=1000, max_error=4, noise_model="bit_flip")
    # shots = load_shots(config, dataset_type="mistakes", noise_model="bit_flip", agent_name="bp")
    num_shots = len(shots)
    logical_failures = 0
    successes = 0

    eval_env = QLDPCEnv(config, shots)
    agent = get_agent(config, eval_env, agent_name)

    for _ in tqdm(eval_env.shots, desc="Evaluating agent", leave=False):
        obs, info = eval_env.reset()
        done = info["error_free"]
        episode_return = 0.0
        episode_length = 0

        while not done:

            action, probs = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_length += 1
            episode_return += reward.item()

        if info["error_free"]:
            successes += 1
        else: # NOTE: This fails both for logical errors and for truncation.
            logical_failures += 1

    ler = float(logical_failures) / num_shots
    print(f"Logical Error Rate: {ler:.4f}\n")

    if checkpoint_dir is not None:
        agent.save(f"{checkpoint_dir}/eval_{step}_ler_{ler}.pt")

        if ler < best_model_ler:
            best_model_ler = ler
            agent.save(f"{checkpoint_dir}/best.pt")
            print(f"New best model saved with LER {best_model_ler:.4f}")

            best_model_ler = ler

    return ler, best_model_ler


def post_train_evaluation(config):
    # Evaluate the agent at the end of training and print results to console.
    eval_agent_results = evaluate_agent(config, agent_name=config.agent_name, log_progress=True)

    config.wandb_run_name = "llr"
    config.use_neural_bp = False
    baseline_results = evaluate_agent(config, agent_name=config.agent_name, log_progress=True)

    bp_results = evaluate_agent(config, agent_name="bp", log_progress=True)
    bp_osd_results = evaluate_agent(config, agent_name="bp_osd", log_progress=True)

    print("\nFinal Evaluation Results:")
    for key, value in eval_agent_results.items():
        print(f"{key}: {value:.4f}")

    print("\nBP Baseline Results:")
    for key, value in bp_results.items():
        print(f"{key}: {value:.4f}")

    print("\nBP+OSD Baseline Results:")
    for key, value in bp_osd_results.items():
        print(f"{key}: {value:.4f}")


    # Plot results
    plot = plt.figure(figsize=(10, 6))
    plt.plot(list(eval_agent_results.keys()), list(eval_agent_results.values()), marker='o', label=f"{config.agent_name.upper()}")
    plt.plot(list(bp_results.keys()), list(bp_results.values()), marker='o', label="BP")
    plt.plot(list(bp_osd_results.keys()), list(bp_osd_results.values()), marker='o', label="BP+OSD")
    plt.yscale('log')
    plt.xlabel("Physical Error Rate")
    plt.ylabel("Logical Error Rate")
    plt.title(f"Logical Error Rate vs Physical Error Rate for {config.code_name}")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(f"results/{config.agent_name}_{config.code_name}_evaluation.png")
