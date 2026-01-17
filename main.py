from environments import MultivariateBicycleCode
from agents import Agent
from utils.plot_utils import plot_results

from tqdm import tqdm
import numpy as np
import time
from stable_baselines3 import DQN, PPO
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

if __name__ == '__main__':
    # As per "Tour de gross: A modular quantum computer based on bivariate bicycle codes, Figure 3b"
    # env = MultivariateBicycleCode(l=12, m=6, interaction_vectors=[(3, 6), (6, -3)])
    # agent = Agent(env)

    env = MultivariateBicycleCode(l=3, m=3, num_errors=1)
    env.render()

    results = {}
    for name in ["dqn"]:
        rewards = []
        for i in range(5):

            agent = DQN(env=env, policy="MlpPolicy") if name == "dqn" else PPO("MlpPolicy", env)
            callback = RewardTrackerCallback()
            agent.learn(total_timesteps=25000, progress_bar=True, callback=callback)
            rewards.append(callback.mean_rewards)

        # Pad rewards so all are of equal length
        max_len = max(len(row) for row in rewards)
        results[name] = np.array([row + [row[-1]] * (max_len - len(row)) for row in rewards])


    plot_results(results)

    # times = []
    # for i in tqdm(range(100)):
    #     observation, info = env.reset()
    #     # env.render()
    #     done = False
    #     start = time.time()
    #     episode_reward = 0
    #     while not done:
    #         observation, reward, terminated, truncated, info = env.step(agent.select_action(observation))
    #         done = terminated or truncated
    #         episode_reward += reward
    #     end = time.time()
    #     times.append(end - start)
    # env.render()
    # print(f"Episode took {np.mean(times)} seconds.")