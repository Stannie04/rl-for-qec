from utils import *
import sys
import json
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_configs(model_config_file, code_config_file, agent_name, code_name):
    model_config = json.load(open(model_config_file))
    code_config = json.load(open(code_config_file))
    return model_config[agent_name], code_config[code_name]

if __name__ == '__main__':

    agent_name = sys.argv[1] if len(sys.argv) > 1 else "dqn"
    code_name = sys.argv[2] if len(sys.argv) > 2 else "18_2_3_toric"

    model_config, code_config = get_configs("model_configs.json", "code_configs.json", agent_name, code_name)

    # optimize_hyperparameters(code_config)

    ldpc = np.load("results/new_results/dqn_lengths_ldpc.npy")
    toric = np.load("results/new_results/dqn_lengths_toric.npy")
    baseline = np.load("results/new_results/silent_agent_rewards.npy")

    results = {}
    baseline = {"Silent Agent": baseline.flatten()}

    for lengths, name in [(ldpc, "LDPC"), (toric, "Toric")]:
        results[name] = [lengths.flatten()]

        #
        # print(f"{name} - Mean: {lengths.mean():.2f}, Std: {lengths.std():.2f}, Max: {lengths.max()}, Min: {lengths.min()}")
        # plt.plot(lengths, label=name)

    plot_results(results, baseline, model_config, 0, f"results/simplified_train_env.png")

        # plot_results({"Reward": lengths}, {}, model_config, f"results/{name}_lengths.png")

    # rewards = train_dqn(code_config, model_config, device=device)
    # benchmark_env(code_config, model_config, device=device)

    # rewards = adversarial_training_loop(model_config=model_config, code_config=code_config)
    #rewards = train_dqn(code_config=code_config, model_config=model_config, device=device)
    #baselines = run_baselines(model_config=model_config, code_config=code_config)

    #rewards.pop("Reward")
    #plot_results(rewards, baselines, model_config, f"results/termination.png")
    # render_evaluation_episode(code_config, "checkpoints/dqn_defender_single.zip")/
