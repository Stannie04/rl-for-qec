from environments import MultivariateBicycleCode
from agents import RandomAgent, AdversarialAgent
from utils.plot_utils import plot_results

import json
import sys

from tqdm import tqdm
import numpy as np
import time
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback


class RewardTrackerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.mean_rewards = []

    def _on_step(self):
        # SB3 logs episode reward in info dict under 'episode' (if env is wrapped)
        info = self.locals["infos"][0]

        if "episode" in info:
            ep_reward = info["episode"]["r"]
            self.episode_rewards.append(ep_reward)

            # Compute rolling mean over last N episodes
            window = 20  # choose any window size
            mean_r = np.mean(self.episode_rewards[-window:])
            self.mean_rewards.append(mean_r)

        return True


def get_agent(agent_name):
    if agent_name == "dqn":
        return DQN(env=env, policy="MlpPolicy")
    elif agent_name == "ppo":
        return PPO("MlpPolicy", env)
    elif agent_name == "adversarial":
        return AdversarialAgent(env)

    raise ValueError(f"Unknown agent: {agent_name}")


def get_config(config_file, agent_name):
    full_config = json.load(open(config_file))
    return full_config[agent_name]


if __name__ == '__main__':
    # As per "Tour de gross: A modular quantum computer based on bivariate bicycle codes, Figure 3b"
    # env = MultivariateBicycleCode(l=12, m=6, interaction_vectors=[(3, 6), (6, -3)])
    # agent = Agent(env)

    agent_name = sys.argv[1] if len(sys.argv) > 1 else "dqn"
    config = get_config("model_configs.json", agent_name)

    env = MultivariateBicycleCode(l=5, m=3, num_errors=1, interaction_vectors=[(3, 4)])
    env.render()

    results = {}

    for error_rate in [0.001, 0.005, 0.01, 0.05]:
        rewards = []
        for i in range(config["n_repetitions"]):
            env = MultivariateBicycleCode(l=5, m=3, num_errors=1, interaction_vectors=[(3,4)], error_rate=error_rate)

            agent = get_agent(agent_name)
            callback = RewardTrackerCallback()
            agent.learn(total_timesteps=config["num_timesteps"], progress_bar=True, callback=callback)
            rewards.append(callback.mean_rewards)

        # Pad rewards so all are of equal length
        max_len = max(len(row) for row in rewards)
        results[error_rate] = np.array([row + [row[-1]] * (max_len - len(row)) for row in rewards])

    plot_results(results)
