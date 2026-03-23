from environments import QLDPCCode
from utils import *
from agents import DQNAgent
import sys
import json
import time
import numpy as np
from tqdm import tqdm
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_configs(model_config_file, code_config_file, agent_name, code_name):
    model_config = json.load(open(model_config_file))
    code_config = json.load(open(code_config_file))
    return model_config[agent_name], code_config[code_name]


def benchmark_env(code_config):

    sample_config = code_config.copy()

    start = time.time()
    env = QLDPCCode(**sample_config, device=device)
    agent = DQNAgent(env, device=device)
    end = time.time()
    print(f"Initialization took {end - start:.5f} seconds")

    obs, info = env.reset()

    step_times = []
    agent_times = []
    for _ in range(1000):

        start = time.time()
        action = agent.select_action(obs)
        end = time.time()
        agent_times.append(end - start)

        start = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        end = time.time()
        step_times.append(end - start)

    print(f"Environment step takes on average {np.mean(step_times[1:]):.5f} seconds")
    print(f"Agent action selection takes on average {np.mean(agent_times[1:]):.5f} seconds")


if __name__ == '__main__':

    agent_name = sys.argv[1] if len(sys.argv) > 1 else "dqn"
    code_name = sys.argv[2] if len(sys.argv) > 2 else "18_4_4"

    model_config, code_config = get_configs("model_configs.json", "code_configs.json", agent_name, code_name)

    rewards = train_dqn(code_config, model_config, device=device)

    # benchmark_env(code_config)

    # rewards = adversarial_training_loop(model_config=model_config, code_config=code_config)
    # rewards = single_agent_training_loop(code_config=code_config, model_config=model_config, model_checkpoint="checkpoints/dqn_defender_single.zip")
    baselines = run_baselines(model_config=model_config, code_config=code_config)
    #
    plot_results(rewards, baselines, model_config, f"results/termination.png")
    # render_evaluation_episode(code_config, "checkpoints/dqn_defender_single.zip")
