"""Module for creating synthetic datasets for training."""

import os
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter

from src.environment import QLDPCEnv
from src.agents import SACAgent, BPAgent, BPOSDAgent


def load_shots(config, dataset_type="random", noise_model="bit_flip"):
    match dataset_type:
        case "random": shots = np.load(f"datasets/{config.code_name}/random_{noise_model}.npy", allow_pickle=True)
        case "uniform": shots = np.load(f"datasets/{config.code_name}/uniform_{noise_model}.npy", allow_pickle=True)
        case "nonzero": shots = np.load(f"datasets/{config.code_name}/nonzero_{noise_model}.npy", allow_pickle=True)
        case "mistakes":
            shots = []
            for agent_name in config.moe_experts:
                mistakes = np.load(f"datasets/{config.code_name}/mistakes_{agent_name}_{noise_model}.npy", allow_pickle=True)
                shots.append(mistakes)
            shots = np.concatenate(shots, axis=0)

            # Order by number of errors to speed up training
            num_errors = np.sum(shots[:, 0, :], axis=1) + np.sum(shots[:, 1, :], axis=1)
            sorted_indices = np.argsort(num_errors)
            shots = shots[sorted_indices]

        case _: raise ValueError(f"Unknown dataset type: {dataset_type}")

    return shots


def create_dataset_from_random_shots(config, num_samples, error_rate, noise_model="bit_flip", save=False):

    print(f"Creating dataset of {num_samples} random shots for code {config.code_name}")
    physical_error_rate = error_rate / 2 if noise_model == "depolarizing" else error_rate

    shots = np.zeros((num_samples, 2, config.n), dtype=np.int8)
    shots[:, 0, :] = np.random.rand(num_samples, config.n) < physical_error_rate

    if noise_model == "depolarizing":
        shots[:, 1, :] = np.random.rand(num_samples, config.n) < physical_error_rate

    if save:
        os.makedirs(f"datasets/{config.code_name}", exist_ok=True)
        np.save(f"datasets/{config.code_name}/random_{noise_model}.npy", shots)

    return shots

def create_dataset_from_uniform_shots(config, num_samples_per_error, max_error, noise_model="bit_flip", save=False):

    print(f"Creating dataset of {num_samples_per_error} uniform random shots for code {config.code_name}")
    shots = np.zeros((num_samples_per_error * max_error, 2, config.n), dtype=np.int8)

    for num_errors in tqdm(range(1, max_error + 1), desc="Generating uniform random shots", leave=False):
        for i in range(num_samples_per_error):
            idx = (num_errors-1)*num_samples_per_error + i
            error_indices = np.random.choice(config.n, num_errors, replace=False)
            shots[idx, 0, error_indices] = 1

            if noise_model == "depolarizing":
                z_error_indices = np.random.choice(config.n, num_errors, replace=False)
                shots[idx, 1, z_error_indices] = 1

    if save:
        os.makedirs(f"datasets/{config.code_name}", exist_ok=True)
        np.save(f"datasets/{config.code_name}/uniform_{noise_model}.npy", shots)

    return shots


def create_dataset_from_nonzero_shots(config, num_samples, error_rate, noise_model="bit_flip", save=False):

    print(f"Creating dataset of {num_samples} nonzero random shots for code {config.code_name}")

    physical_error_rate = error_rate / 2 if noise_model == "depolarizing" else error_rate
    error_types = [0, 1] if noise_model == "depolarizing" else [0]

    for error_type in error_types:
        shots = np.zeros((num_samples, 2, config.n), dtype=np.int8)
        shots[:, error_type, :] = np.random.rand(num_samples, config.n) < physical_error_rate

        # add exactly one extra 1 per sample
        rows = np.arange(num_samples)
        cols = np.random.randint(0, config.n, size=num_samples)
        shots[rows, error_type, cols] = 1

    if save:
        os.makedirs(f"datasets/{config.code_name}", exist_ok=True)
        np.save(f"datasets/{config.code_name}/nonzero_{noise_model}.npy", shots)

    return shots

def create_dataset_from_expert_mistakes(config, agent_name, shot_type="uniform", noise_model="bit_flip", save=False):

    env = QLDPCEnv(config)
    match agent_name:
        case "sac":
            agent = SACAgent(env, config)
            checkpoint = torch.load(f"checkpoints/{config.agent_name}_{config.code_name}.pt", map_location=config.device)
            agent.actor.load_state_dict(checkpoint["actor"])
            agent.critic1.load_state_dict(checkpoint["critic1"])
            agent.critic2.load_state_dict(checkpoint["critic2"])
        case "sac_finetuned":
            agent = SACAgent(env, config)
            checkpoint = torch.load(f"checkpoints/{config.agent_name}_{config.code_name}_finetuned.pt", map_location=config.device)
            agent.actor.load_state_dict(checkpoint["actor"])
            agent.critic1.load_state_dict(checkpoint["critic1"])
            agent.critic2.load_state_dict(checkpoint["critic2"])
        case "bp": agent = BPAgent(env, config)
        case "bp_osd": agent = BPOSDAgent(env, config)
        case _: raise NotImplementedError

    dataset = []
    shots = load_shots(config, dataset_type=shot_type, noise_model=noise_model)
    for error_pattern_x, error_pattern_z in tqdm(shots, desc=f"Creating dataset for {agent_name}", leave=False):
        obs, info = env.reset_with_error_pattern(error_pattern_x, error_pattern_z)

        done = info["error_free"]
        while not done:
            action, _ = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if not info["error_free"]:
            dataset.append([error_pattern_x, error_pattern_z])

    print(f"Collected {len(dataset)} samples of expert mistakes for agent {agent_name} on code {config.code_name}.")

    if save:
        os.makedirs(f"datasets/{config.code_name}", exist_ok=True)
        np.save(f"datasets/{config.code_name}/mistakes_{agent_name}_{noise_model}.npy", np.array(dataset))


def create_all_datasets(config):
    # create_dataset_from_random_shots(config, num_samples=int(1e6), noise_model="bit_flip", error_rate=config.curriculum_error_rate, save=True)
    # create_dataset_from_nonzero_shots(config, num_samples=int(1e6), noise_model="bit_flip", error_rate=config.curriculum_error_rate, save=True)
    # create_dataset_from_uniform_shots(config, num_samples_per_error=int(1e5), max_error=4, noise_model="bit_flip", save=True)

    for agent_name in config.moe_experts:
        create_dataset_from_expert_mistakes(config, agent_name, shot_type="uniform", noise_model="bit_flip", save=True)
