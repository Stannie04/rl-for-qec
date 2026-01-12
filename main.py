from environments import MultivariateBicycleCode
from agents import Agent
from tqdm import tqdm
import numpy as np
import time

if __name__ == '__main__':
    # As per "Tour de gross: A modular quantum computer based on bivariate bicycle codes, Figure 3b"
    env = MultivariateBicycleCode(l=12, m=6, interaction_vectors=[(3, 6), (6, -3)])
    agent = Agent(env)

    times = []
    for i in tqdm(range(100)):
        observation, info = env.reset()
        # env.render()
        done = False
        start = time.time()
        episode_reward = 0
        while not done:
            observation, reward, terminated, truncated, info = env.step(agent.select_action(observation))
            done = terminated or truncated
            episode_reward += reward
        end = time.time()
        times.append(end - start)
    env.render()
    print(f"Episode took {np.mean(times)} seconds.")