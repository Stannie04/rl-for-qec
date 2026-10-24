"""Module for creating synthetic datasets for training."""

import os
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from collections import Counter, defaultdict
from itertools import combinations
from math import comb

from src.environment import QECEnv
from src.train_utils.curriculum import CurriculumScheduler
from src.train_utils.inference import get_agent_and_inference, inference


def load_shots(config, dataset_type="random", agent_name=None, num_epochs=None):
    match dataset_type:
        case "random": shots = np.load(f"datasets/{config.code_name}/random_{config.noise_model}.npy", allow_pickle=True)
        case "uniform": shots = np.load(f"datasets/{config.code_name}/uniform_{config.noise_model}.npy", allow_pickle=True)
        case "nonzero": shots = np.load(f"datasets/{config.code_name}/nonzero_{config.noise_model}.npy", allow_pickle=True)
        case "all": shots = np.load(f"datasets/{config.code_name}/all_{config.noise_model}.npy", allow_pickle=True)
        case "moe": shots = np.load(f"datasets/{config.code_name}/moe_{config.noise_model}.npy", allow_pickle=True)
        case "moe_labelled": shots = torch.load(f"datasets/{config.code_name}/moe_{config.noise_model}_labelled.pt", map_location=torch.device('cpu'))
        case "mistakes":
            shots = np.load(f"datasets/{config.code_name}/mistakes/{agent_name}_{config.noise_model}_uniform.npy", allow_pickle=True)

            # If num_epochs is specified, repeat the dataset to match the number of epochs
            # Shape of shots is (num_samples * num_epochs, 2, n)
            if num_epochs is not None:
                shots = np.tile(shots, (num_epochs, 1, 1))

            # Order by number of errors to speed up training
            num_errors = np.sum(shots[:, 0, :], axis=1) + np.sum(shots[:, 1, :], axis=1)
            sorted_indices = np.argsort(num_errors)
            shots = shots[sorted_indices]

        case _: raise ValueError(f"Unknown dataset type: {dataset_type}")

    print(f"Loaded {len(shots)} number of shots")
    return shots


def create_dataset_from_moe_shots(config, save=False):
    hard_shots = load_shots(config, dataset_type="mistakes", agent_name="bp")
    # Note that this does also sample some hard shots, though the vast majority will be easy shots. The small imbalance this creates is not a problem.
    easy_shots = create_dataset_from_uniform_shots(config, num_samples_per_error=int(len(hard_shots)/4), max_error=4, save=False)

    all_shots = np.concatenate((hard_shots, easy_shots), axis=0)
    np.random.shuffle(all_shots)

    if save:
        os.makedirs(f"datasets/{config.code_name}", exist_ok=True)
        np.save(f"datasets/{config.code_name}/moe_{config.noise_model}.npy", all_shots)

    return all_shots


def create_dataset_from_moe_shots_labelled(config, save=False):
    shots = load_shots(config, dataset_type="uniform")
    np.random.shuffle(shots)

    # Build labels.
    mistakes = {expert: load_shots(config, dataset_type="mistakes", agent_name=expert) for expert in config.moe_experts}
    labels = np.zeros((len(shots), len(config.moe_experts)), dtype=np.int8)
    mistake_sets = [
        {np.ascontiguousarray(shot).tobytes() for shot in expert_mistakes}
        for expert_mistakes in mistakes.values()
    ]
    for i, shot in tqdm(enumerate(shots)):
        shot_key = np.ascontiguousarray(shot).tobytes()
        for j, mistake_set in enumerate(mistake_sets):
            labels[i, j] = shot_key in mistake_set
    labels = torch.as_tensor(labels, dtype=torch.float32).cpu()

    # Build observations.
    env = QECEnv(config, shots=shots)
    edge_index = env.code.data.edge_index
    observations = [env.reset()[0].x.detach().clone() for _ in tqdm(env.shots, desc="Collecting observations for MOE dataset", leave=False)]
    dataset = {"x": torch.stack(observations), "edge_index": edge_index, "labels": labels}

    if save:
        os.makedirs(f"datasets/{config.code_name}", exist_ok=True)
        torch.save(dataset, f"datasets/{config.code_name}/moe_{config.noise_model}_labelled.pt")

    return shots, labels


def create_dataset_from_curriculum(config, num_samples, specialized=False, save=False):
    """"Dataset consisting of random nonzero shots, with error rates increasing order to match the curriculum used during training."""

    curriculum = CurriculumScheduler(config)
    error_rates = curriculum.error_rates_for_steps(range(num_samples))

    has_error = np.random.rand(num_samples, config.n) < error_rates[:, None]

    empty = ~has_error.any(axis=1)
    while empty.any():
        rows = np.flatnonzero(empty)
        has_error[rows] = np.random.rand(len(rows), config.n) < error_rates[rows, None]
        empty = ~has_error.any(axis=1)

    shots = np.zeros((num_samples, 2, config.n), dtype=np.int8)

    if config.noise_model == "depolarizing":
        # Here, 0 = X, 1 = Z, 2 = Y
        paulis = np.random.randint(0, 3, size=(num_samples, config.n))
        shots[:, 0, :] = has_error & (paulis != 1) # X errors
        shots[:, 1, :] = has_error & (paulis != 0) # Z errors

    elif config.noise_model == "bit_flip":
        shots[:, 0, :] = has_error # X errors only

    else:
        raise ValueError(f"Unknown noise model: {config.noise_model}")

    # Inject the mistakes made by BP into the curriculum dataset, to ensure the agent learns to handle them even at higher error rates.
    if specialized:
        bp_mistakes = load_shots(config, dataset_type="mistakes", agent_name="bp")
        num_mistakes = len(bp_mistakes)
        if num_mistakes > 0:
            indices_to_replace = np.random.choice(num_samples, size=min(num_mistakes, num_samples), replace=False)
            shots[indices_to_replace] = bp_mistakes[:len(indices_to_replace)]


    if save:
        os.makedirs(f"datasets/{config.code_name}", exist_ok=True)
        np.save(f"datasets/{config.code_name}/curriculum_{config.noise_model}.npy", shots)

    # Ensure that both X and Z errors are present in the dataset for depolarizing noise
    if config.noise_model == "depolarizing":
        assert np.any(shots[:, 0, :]) and np.any(shots[:, 1, :]), "Curriculum dataset must contain both X and Z errors for depolarizing noise."

    return shots


def create_dataset_from_random_shots(config, num_samples, error_rate, save=False):

    shots = np.zeros((num_samples, 2, config.n), dtype=np.int8)

    if config.noise_model == "depolarizing":
        # Each physical qubit gets a non-identity Pauli with
        # probability p, then X, Y, or Z is selected uniformly.
        has_error = (np.random.rand(num_samples, config.n) < error_rate)
        paulis = np.random.randint(0, 3, size=(num_samples, config.n))

        # Mapping: 0 = X, 1 = Z, 2 = Y.
        shots[:, 0, :] = has_error & (paulis != 1)
        shots[:, 1, :] = has_error & (paulis != 0)

    elif config.noise_model == "bit_flip":
        # Bit-flip channel: X with probability p, otherwise I.
        shots[:, 0, :] = (np.random.rand(num_samples, config.n) < error_rate)

    else:
        raise ValueError(f"Unsupported noise model: {config.noise_model}")

    if save:
        os.makedirs(f"datasets/{config.code_name}", exist_ok=True)
        np.save(f"datasets/{config.code_name}/random_{config.noise_model}.npy", shots)

    return shots


def create_dataset_from_random_shots_labelled(config, num_samples, error_rates, save=False):

    shot_counts = defaultdict(lambda: np.zeros(len(error_rates), dtype=np.int64))
    packed_shots = {}

    for i, error_rate in enumerate(error_rates):
        shots = create_dataset_from_random_shots(config, num_samples, error_rate, save=False)

        packed = np.packbits(shots.reshape(num_samples, -1), axis=-1)
        unique, counts = np.unique(packed, axis=0, return_counts=True)
        weights = np.sum(shots, axis=(1, 2))
        avg_weight, max_weight, median_weight = np.mean(weights), np.max(weights), np.median(weights)
        print(f"Error rate {error_rate}: {len(unique)} unique shots out of {num_samples} samples | Mean: {avg_weight:.2f}, Max: {max_weight}, Median: {median_weight}")

        for shot, count in zip(unique, counts):
            key = shot.tobytes()
            shot_counts[key][i] = count

            # Store packed representation once
            if key not in packed_shots:
                packed_shots[key] = shot

    keys = list(packed_shots.keys())

    unique_packed = np.stack([packed_shots[k] for k in keys])
    unique_flat = np.unpackbits(unique_packed,axis=1,count=2 * config.n,)
    unique_shots = unique_flat.reshape(-1, 2, config.n)

    counts = np.stack([shot_counts[k] for k in keys])

    print(f"\nTotal unique shots across all error rates: {len(unique_shots)}")
    return unique_shots, counts


def _reorder_shots_for_workers(shots, num_workers):
    num_shots = shots.shape[0]

    if num_shots % num_workers != 0:
        raise ValueError(f"Number of shots ({num_shots}) must be divisible by number of workers ({num_workers}).")

    idx = np.arange(num_shots).reshape(-1, num_workers).T.ravel()

    return shots[idx]


def create_dataset_from_uniform_shots(config, num_samples_per_error, max_error, reorder=False, save=False):

    shots = np.zeros((num_samples_per_error * max_error, 2, config.n), dtype=np.int8)

    for num_errors in tqdm(range(1, max_error + 1), desc="Generating uniform random shots", leave=False):
        for i in range(num_samples_per_error):
            idx = (num_errors-1)*num_samples_per_error + i
            error_indices = np.random.choice(config.n, num_errors, replace=False)
            shots[idx, 0, error_indices] = 1

            if config.noise_model == "depolarizing":
                z_error_indices = np.random.choice(config.n, num_errors, replace=False)
                shots[idx, 1, z_error_indices] = 1

    if reorder:
        shots = _reorder_shots_for_workers(shots, config.num_workers)

    if save:
        os.makedirs(f"datasets/{config.code_name}", exist_ok=True)
        np.save(f"datasets/{config.code_name}/uniform_{config.noise_model}.npy", shots)

    return shots


def create_dataset_from_nonzero_shots(config, num_samples, error_rate, save=False):

    physical_error_rate = error_rate / 2 if config.noise_model == "depolarizing" else error_rate
    error_types = [0, 1] if config.noise_model == "depolarizing" else [0]

    for error_type in error_types:
        shots = np.zeros((num_samples, 2, config.n), dtype=np.int8)
        shots[:, error_type, :] = np.random.rand(num_samples, config.n) < physical_error_rate

        # add exactly one extra 1 per sample
        rows = np.arange(num_samples)
        cols = np.random.randint(0, config.n, size=num_samples)
        shots[rows, error_type, cols] = 1

    if save:
        os.makedirs(f"datasets/{config.code_name}", exist_ok=True)
        np.save(f"datasets/{config.code_name}/nonzero_{config.noise_model}.npy", shots)

    return shots


def create_dataset_from_expert_mistakes(config, agent_name, shot_type="uniform", save=False):

    shots = load_shots(config, dataset_type=shot_type)
    dataset = inference(agent_name, config, shots, task="mistakes")

    print(f"Collected {len(dataset)} samples of expert mistakes for agent {agent_name} on code {config.code_name}.")

    if save:
        os.makedirs(f"datasets/{config.code_name}/mistakes", exist_ok=True)
        np.save(f"datasets/{config.code_name}/mistakes/{agent_name}_{config.noise_model}_{shot_type}.npy", np.array(dataset))

    return dataset


def create_dataset_from_pretrained_encoder_mistakes(config, model, shot_type="uniform", encoder_type="nbp", save=False):
    dataset = []
    # shots = load_shots(config, dataset_type=shot_type, noise_model=noise_model)
    # shots = create_dataset_from_uniform_shots(config, num_samples_per_error=int(1e4), max_error=4, noise_model=noise_model, save=True)
    shots = load_shots(config, dataset_type=shot_type)
    env = QECEnv(config, shots)
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
        os.makedirs(f"datasets/{config.code_name}/mistakes", exist_ok=True)
        np.save(f"datasets/{config.code_name}/mistakes/finetuned_{encoder_type}_early_{config.noise_model}.npy", np.array(dataset))

    return dataset


def create_dataset_from_all_permutations(config, num_errors=[3,4], save=False):
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
        np.save(f"datasets/{config.code_name}/all_{config.noise_model}.npy", shots)

    return shots


def create_all_datasets(config):
    # create_dataset_from_random_shots(config, num_samples=int(1e6), error_rate=config.curriculum_end_error_rate, save=True)
    # create_dataset_from_nonzero_shots(config, num_samples=int(1e6), error_rate=config.curriculum_end_error_rate, save=True)
    # create_dataset_from_uniform_shots(config, num_samples_per_error=int(1e5), max_error=5, reorder=True, save=True)

    # for agent_name in ["rl_nbp"]:
    #     create_dataset_from_expert_mistakes(config, agent_name, shot_type="uniform", save=True)

    create_dataset_from_moe_shots_labelled(config, save=True)

    # create_dataset_from_all_permutations(config, save=True)
    # create_dataset_from_moe_shots(config, save=True)
    # create_dataset_from_random_shots_labelled(config, num_samples=int(1e7), error_rates=config.curriculum_end_error_rate, save=False)
