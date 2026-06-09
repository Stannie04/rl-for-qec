"""Module for creating synthetic datasets for training."""

import os

import torch
import numpy as np
from tqdm import tqdm

from src.environment import QLDPCEnv
from src.agents import SACAgent, BPAgent, BPOSDAgent


import numpy as np
from tqdm import tqdm


def create_dataset_from_random_shots(config, num_samples):
    env = QLDPCEnv(config)
    shots = []
    for _ in tqdm(range(num_samples), desc="Sampling shots", leave=False):
        obs, info = env.reset()
        shots.append((env.code.x_errors.cpu().numpy(), env.code.z_errors.cpu().numpy()))

    os.makedirs(f"datasets/{config.code_name}", exist_ok=True)
    np.save(f"datasets/{config.code_name}/random_shots.npy", np.array(shots))
    print(f"Collected {len(shots)} shots for code {config.code_name}.")

    return shots


def create_dataset_from_expert_mistakes(config, agent_name):

    env = QLDPCEnv(config)
    match agent_name:
        case "sac":
            agent = SACAgent(env, config)
            checkpoint = torch.load(f"checkpoints/{config.agent_name}_{config.code_name}.pt", map_location=config.device)
            agent.actor.load_state_dict(checkpoint["actor"])
            agent.critic1.load_state_dict(checkpoint["critic1"])
            agent.critic2.load_state_dict(checkpoint["critic2"])
        case "bp":
            agent = BPAgent(env, config)
        case "bp_osd":
            agent = BPOSDAgent(env, config)
        case _:
            raise NotImplementedError

    dataset = []
    shots = load_shots(config)
    for error_pattern_x, error_pattern_z in tqdm(shots, desc=f"Creating dataset for {agent_name}", leave=False):

        env.code.set_error_pattern(error_pattern_x, error_pattern_z)
        obs = env.observation

        done = False
        while not done:
            action, _ = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if not env.code.is_error_free():
            dataset.append([error_pattern_x, error_pattern_z])

    print(f"Collected {len(dataset)} samples of expert mistakes for agent {agent_name} on code {config.code_name}.")

    os.makedirs(f"datasets/{config.code_name}", exist_ok=True)
    np.save(f"datasets/{config.code_name}/mistakes_{agent_name}.npy", np.array(dataset))



def create_all_datasets(config):
    create_dataset_from_random_shots(config, num_samples=int(1e3))

    # for agent_name in ["sac", "bp", "bp_osd"]:
    for agent_name in ["sac"]:
        create_dataset_from_expert_mistakes(config, agent_name)


def load_mistakes(config):
    all_mistakes = []

    for agent_name in config.moe_experts:
        mistakes = np.load(f"datasets/{config.code_name}/mistakes_{agent_name}.npy", allow_pickle=True)
        all_mistakes.append(mistakes)

    return all_mistakes


def load_shots(config):
    shots = np.load(f"datasets/{config.code_name}/random_shots.npy", allow_pickle=True)
    return shots