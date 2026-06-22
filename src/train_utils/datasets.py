"""Module for creating synthetic datasets for training."""

import os
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter
from itertools import combinations
from math import comb

from src.environment import QLDPCEnv
from src.agents import SACAgent, BPAgent, BPOSDAgent
from src.train_utils.curriculum import CurriculumScheduler


def load_shots(config, dataset_type="random", noise_model="bit_flip", agent_name=None, num_epochs=None):
    match dataset_type:
        case "random": shots = np.load(f"datasets/{config.code_name}/random_{noise_model}.npy", allow_pickle=True)
        case "uniform": shots = np.load(f"datasets/{config.code_name}/uniform_{noise_model}.npy", allow_pickle=True)
        case "nonzero": shots = np.load(f"datasets/{config.code_name}/nonzero_{noise_model}.npy", allow_pickle=True)
        case "all": shots = np.load(f"datasets/{config.code_name}/all_{noise_model}.npy", allow_pickle=True)
        case "moe": shots = np.load(f"datasets/{config.code_name}/moe_{noise_model}.npy", allow_pickle=True)
        case "mistakes":
            shots = np.load(f"datasets/{config.code_name}/mistakes_{agent_name}_{noise_model}_all.npy", allow_pickle=True)

            # If num_epochs is specified, repeat the dataset to match the number of epochs
            # Shape of shots is (num_samples * num_epochs, 2, n)
            if num_epochs is not None:
                shots = np.tile(shots, (num_epochs, 1, 1))

            # Order by number of errors to speed up training
            num_errors = np.sum(shots[:, 0, :], axis=1) + np.sum(shots[:, 1, :], axis=1)
            sorted_indices = np.argsort(num_errors)
            shots = shots[sorted_indices]

        case _: raise ValueError(f"Unknown dataset type: {dataset_type}")

    return shots


def create_dataset_from_moe_shots(config, noise_model="bit_flip", save=False):
    hard_shots = load_shots(config, dataset_type="mistakes", noise_model=noise_model, agent_name="bp")

    # Note that this does also sample some hard shots, though the vast majority will be easy shots. The small imbalance this creates is not a problem.
    easy_shots = create_dataset_from_uniform_shots(config, num_samples_per_error=int(len(hard_shots)/4), max_error=4, noise_model="bit_flip", save=False)

    all_shots = np.concatenate((hard_shots, easy_shots), axis=0)
    np.random.shuffle(all_shots)

    if save:
        os.makedirs(f"datasets/{config.code_name}", exist_ok=True)
        np.save(f"datasets/{config.code_name}/moe_{noise_model}.npy", all_shots)

    return all_shots


def create_dataset_from_curriculum(config, num_samples, noise_model="bit_flip", with_mistakes=False, save=False):
    """"Dataset consisting of random nonzero shots, with error rates increasing order to match the curriculum used during training."""

    curriculum = CurriculumScheduler(config)
    error_rates = curriculum.error_rates_for_steps(range(num_samples))
    error_types = [0, 1] if noise_model == "depolarizing" else [0]

    for error_type in error_types:
        shots = np.zeros((num_samples, 2, config.n), dtype=np.int8)

        shots[:, error_type, :] = np.random.rand(num_samples, config.n) < error_rates[:, None]

        # add exactly one extra 1 per sample
        rows = np.arange(num_samples)
        cols = np.random.randint(0, config.n, size=num_samples)
        shots[rows, error_type, cols] = 1

    # Inject the mistakes made by BP into the curriculum dataset, to ensure the agent learns to handle them even at higher error rates.
    if with_mistakes:
        bp_mistakes = load_shots(config, dataset_type="mistakes", noise_model=noise_model, agent_name="bp")
        num_mistakes = len(bp_mistakes)
        if num_mistakes > 0:
            indices_to_replace = np.random.choice(num_samples, size=min(num_mistakes, num_samples), replace=False)
            shots[indices_to_replace] = bp_mistakes[:len(indices_to_replace)]


    if save:
        os.makedirs(f"datasets/{config.code_name}", exist_ok=True)
        np.save(f"datasets/{config.code_name}/curriculum_{noise_model}.npy", shots)

    return shots


def create_dataset_from_random_shots(config, num_samples, error_rate, noise_model="bit_flip", save=False):

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
            config.use_neural_bp = False

            agent = SACAgent(env, config)
            checkpoint = torch.load(f"checkpoints/{config.agent_name}_{config.code_name}_llr.pt", map_location=config.device)
            agent.actor.load_state_dict(checkpoint["actor"])
            agent.critic1.load_state_dict(checkpoint["critic1"])
            agent.critic2.load_state_dict(checkpoint["critic2"])
        case "sac_finetuned":
            config.use_neural_bp = True

            agent = SACAgent(env, config)
            checkpoint = torch.load(f"checkpoints/{config.agent_name}_{config.code_name}_neural_bp.pt", map_location=config.device)
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
        np.save(f"datasets/{config.code_name}/mistakes_{agent_name}_{noise_model}_{shot_type}.npy", np.array(dataset))


def create_dataset_from_pretrained_encoder_mistakes(config, model, shot_type="uniform", noise_model="bit_flip", save=False):
    dataset = []
    # shots = load_shots(config, dataset_type=shot_type, noise_model=noise_model)
    # shots = create_dataset_from_uniform_shots(config, num_samples_per_error=int(1e4), max_error=4, noise_model=noise_model, save=True)
    shots = load_shots(config, dataset_type=shot_type, noise_model=noise_model)
    env = QLDPCEnv(config, shots)
    for _ in tqdm(env.shots):
        obs, info = env.reset()
        true_indices = torch.nonzero(env.code.x_errors.float()).flatten()

        with torch.no_grad():
            error_pred = model(obs)

        predicted_indices = torch.nonzero(error_pred > 0.5).flatten()
        pred_sorted = torch.sort(predicted_indices).values
        true_sorted = torch.sort(true_indices).values
        if not torch.equal(pred_sorted, true_sorted):
            dataset.append([env.code.x_errors.cpu().numpy().astype(np.int8), env.code.z_errors.cpu().numpy().astype(np.int8)])

    print(f"Collected {len(dataset)} samples of pretrained encoder mistakes for code {config.code_name}.")

    if save:
        os.makedirs(f"datasets/{config.code_name}", exist_ok=True)
        np.save(f"datasets/{config.code_name}/mistakes_finetuned_encoder_early_{noise_model}.npy", np.array(dataset))


def create_dataset_from_all_permutations(config, noise_model="bit_flip", save=False):
    num_errors = [3, 4]
    num_samples = sum(comb(config.n, weight) for weight in num_errors)

    print(f"Creating dataset of all {num_samples} permutations in {config.code_name}")
    shots = np.zeros((num_samples, 2, config.n), dtype=np.int8)

    i = 0
    with tqdm(total=num_samples, leave=False) as pbar:
        for weight in num_errors:
            for error_indices in combinations(range(config.n), weight):
                shots[i, 0, list(error_indices)] = 1
                i += 1
                pbar.update(1)

    if save:
        os.makedirs(f"datasets/{config.code_name}", exist_ok=True)
        np.save(f"datasets/{config.code_name}/all_{noise_model}.npy", shots)

def create_all_datasets(config):
    # create_dataset_from_random_shots(config, num_samples=int(1e6), noise_model="bit_flip", error_rate=config.curriculum_end_error_rate, save=True)
    # create_dataset_from_nonzero_shots(config, num_samples=int(1e6), noise_model="bit_flip", error_rate=config.curriculum_end_error_rate, save=True)
    # create_dataset_from_uniform_shots(config, num_samples_per_error=int(1e5), max_error=4, noise_model="bit_flip", save=True)
    # create_dataset_from_all_permutations(config, noise_model="bit_flip", save=True)
    create_dataset_from_moe_shots(config, save=True)
    #
    # for agent_name in config.moe_experts:
    #     create_dataset_from_expert_mistakes(config, agent_name, shot_type="all", noise_model="bit_flip", save=True)
