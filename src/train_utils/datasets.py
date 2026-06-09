"""Module for creating synthetic datasets for training."""

import torch
import numpy as np
from tqdm import tqdm

from src.environment import QLDPCEnv
from src.agents import SACAgent, BPAgent, BPOSDAgent


import numpy as np
from tqdm import tqdm


def sample_shots(config, num_samples):
    env = QLDPCEnv(config)
    shots = []
    for _ in tqdm(range(num_samples), desc="Sampling shots", leave=False):
        obs, info = env.reset()
        shots.append((env.code.x_errors.cpu().numpy(), env.code.z_errors.cpu().numpy()))
    np.save(f"datasets/shots_{config.code_name}.npy", np.array(shots))
    print(f"Collected {len(shots)} shots for code {config.code_name}.")

    return shots


def create_dataset_from_expert_mistakes(config, agent_name, shots):

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
    np.save(f"datasets/mistakes_{agent_name}_{config.code_name}.npy", np.array(dataset))



def create_all_datasets(config):
    shots = sample_shots(config, num_samples=int(1e5))

    # for agent_name in ["sac", "bp", "bp_osd"]:
    for agent_name in ["sac"]:
        create_dataset_from_expert_mistakes(config, agent_name, shots)


def analyze_all_datasets(config):

    sac_mistakes = np.load(f"datasets/mistakes_sac_{config.code_name}.npy", allow_pickle=True)
    bp_mistakes = np.load(f"datasets/mistakes_bp_{config.code_name}.npy", allow_pickle=True)
    bp_osd_mistakes = np.load(f"datasets/mistakes_bp_osd_{config.code_name}.npy", allow_pickle=True)

    for agent_name, mistakes in [("SAC", sac_mistakes), ("BP", bp_mistakes), ("BP+OSD", bp_osd_mistakes)]:

        values, counts = np.unique(mistakes[:, 0, :].sum(axis=-1), return_counts=True)
        print(f"\n\n{agent_name} Mistakes Distribution:")
        for v, c in zip(values, counts):
            print(f"  {v} errors: {c} samples")
        print(f"\nTotal samples: {len(mistakes)}")
        # Check if the other agents make the same mistakes
        for other_agent_name, other_mistakes in [("SAC", sac_mistakes), ("BP", bp_mistakes), ("BP+OSD", bp_osd_mistakes)]:
            if other_agent_name == agent_name:
                continue
            overlap = sum(any(np.array_equal(m, om) for om in other_mistakes) for m in mistakes)
            print(f"  Overlap with {other_agent_name}: {overlap} samples ({overlap/len(mistakes)*100:.2f}%)")


def load_mistakes(config):
    all_mistakes = []

    for agent_name in config.moe_experts:
        mistakes = np.load(f"datasets/mistakes_{agent_name}_{config.code_name}.npy", allow_pickle=True)
        all_mistakes.append(mistakes)

    return all_mistakes