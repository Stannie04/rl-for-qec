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


def get_agent(agent_name, env):
    if agent_name == "dqn":
        return DQN(env=env, policy="MlpPolicy")
    elif agent_name == "ppo":
        return PPO("MlpPolicy", env)

    raise ValueError(f"Unknown agent: {agent_name}")


def get_config(config_file, agent_name):
    full_config = json.load(open(config_file))
    return full_config[agent_name]


def initialize_agents(config, env):

    defender = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=500_000,
        learning_starts=1_000,
    )

    adversary = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=500_000,
        learning_starts=1_000,
    )

    return defender, adversary


def adversarial_training_loop(
    config,
):

    rewards = {"Adversary": [], "Defender": []}

    for _ in range(config["n_repetitions"]):

        env = MultivariateBicycleCode(l=l, m=m, interaction_vectors=interaction_vectors)

        defender, adversary = initialize_agents(config, env)

        callback = RewardTrackerCallback()
        defender.learn(total_timesteps=config["pretrain_timesteps"], progress_bar=True, callback=callback)
        adversary.learn(total_timesteps=0)

        all_rewards_adversary = []
        all_rewards_defender = []

        current_reward_adversary = 0
        current_reward_defender = 0
        obs, _ = env.reset()

        for step in tqdm(range(config["num_timesteps"])):

            actor = defender if env.current_player == 0 else adversary

            action, _ = actor.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action, single_player=False)
            done = terminated or truncated
            info = [info]

            actor.replay_buffer.add(
                obs, next_obs, action, reward, done, info
            )

            if step % config["train_frequency"] == 0:
                if actor.replay_buffer.size() >= actor.learning_starts:
                    actor.train(batch_size=32, gradient_steps=1)

            obs = next_obs

            if env.current_player == 0:
                current_reward_adversary += reward
            else:
                current_reward_defender += reward

            if done:
                obs, _ = env.reset()
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
    m=3
    interaction_vectors = [(3, 4)]

    agent_name = sys.argv[1] if len(sys.argv) > 1 else "dqn"
    config = get_config("model_configs.json", agent_name)

    rewards = adversarial_training_loop(config=config)

    plot_results(rewards, "results/adversarial_pretrain.png")
