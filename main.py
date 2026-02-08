from environments import MultivariateBicycleCode
from agents import RandomAgent
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

    raise ValueError(f"Unknown agent: {agent_name}")


def get_config(config_file, agent_name):
    full_config = json.load(open(config_file))
    return full_config[agent_name]


def adversarial_training_loop(
    env,
    defender,
    adversary,
    total_steps,
    config,
    train_freq=4,
):

    rewards = {"Adversary": [], "Defender": []}


    for _ in range(config["n_repetitions"]):

        all_rewards_adversary = []
        all_rewards_defender = []

        current_reward_adversary = 0
        current_reward_defender = 0
        obs, _ = env.reset()

        for step in tqdm(range(total_steps)):

            actor = defender if env.current_player == 0 else adversary

            action, _ = actor.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            info = [info]

            actor.replay_buffer.add(
                obs, next_obs, action, reward, done, info
            )

            if step % train_freq == 0:
                defender.train(batch_size=32, gradient_steps=1)
                adversary.train(batch_size=32, gradient_steps=1)

            obs = next_obs

            if env.current_player == 0:
                current_reward_adversary += reward
            else:
                current_reward_defender += reward

            if done:
                obs, _ = env.reset()
                # print(current_reward_adversary)
                all_rewards_adversary.append(current_reward_adversary)
                all_rewards_defender.append(current_reward_defender)
                current_reward_adversary = 0
                current_reward_defender = 0

        rewards["Adversary"].append(all_rewards_adversary)
        rewards["Defender"].append(all_rewards_defender)

    return rewards


if __name__ == '__main__':
    # As per "Tour de gross: A modular quantum computer based on bivariate bicycle codes, Figure 3b"
    # env = MultivariateBicycleCode(l=12, m=6, interaction_vectors=[(3, 6), (6, -3)])
    # agent = Agent(env)

    l=5
    m=5
    interaction_vectors = [(3, 4)]

    agent_name = sys.argv[1] if len(sys.argv) > 1 else "dqn"
    config = get_config("model_configs.json", agent_name)

    env = MultivariateBicycleCode(
        l=l,
        m=m,
        interaction_vectors=interaction_vectors
    )

    defender = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
    )

    adversary = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
    )

    defender.learn(total_timesteps=0)
    adversary.learn(total_timesteps=0)

    rewards = adversarial_training_loop(
        env=env,
        defender=defender,
        adversary=adversary,
        config=config,
        total_steps=200_000
    )


    #
    # env = MultivariateBicycleCode(l=l, m=m, interaction_vectors=interaction_vectors)
    # env.render()
    #
    # results = {}
    #
    # for error_rate in [0.001, 0.005, 0.01, 0.05]:
    #     rewards = []
    #     for i in range(config["n_repetitions"]):
    #         env = MultivariateBicycleCode(l=l, m=m, interaction_vectors=interaction_vectors, error_rate=error_rate)
    #
    #         agent = get_agent(agent_name)
    #         callback = RewardTrackerCallback()
    #         agent.learn(total_timesteps=config["num_timesteps"], progress_bar=True, callback=callback)
    #         rewards.append(callback.mean_rewards)
    #
    #     # Pad rewards so all are of equal length
    #     max_len = max(len(row) for row in rewards)
    #     results[error_rate] = np.array([row + [row[-1]] * (max_len - len(row)) for row in rewards])
    #

    plot_results(rewards, "results/adversarial.png")
