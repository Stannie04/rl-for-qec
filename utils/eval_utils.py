from stable_baselines3 import SAC
from environments import MultivariateBicycleCode
from agents import SilentAgent
from tqdm import tqdm
import numpy as np


def render_evaluation_episode(code_config, model_checkpoint, max_episode_steps=100):

    model = SAC.load(model_checkpoint, device="cuda")

    env = MultivariateBicycleCode(**code_config, evaluation_mode=True)

    obs, info = env.reset()
    env.render()

    for step in range(max_episode_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step+1}: Reward={reward}, Terminated={terminated}, Truncated={truncated}, Info={info}")
        print(f"Action taken: {action}")
        env.render()

        if terminated or truncated:
            print(f"Episode finished after {step+1} steps with reward {reward} and info {info}")
            return

    print("Episode finished without termination or truncation.")


def run_baselines(
    model_config,
    code_config):

    silent_agent = SilentAgent(**model_config)

    for agent, name in [(silent_agent, "Silent Agent")]:
        env = MultivariateBicycleCode(**code_config)

        total_rewards = []
        for i in tqdm(range(100_000), desc=f"Evaluating {name}"):
            obs, info = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.select_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += 1 # Ignore reward structure, just measure how long it takes for the agent to fail.
                done = terminated or truncated

            total_rewards.append(total_reward)

        return {name: total_rewards}