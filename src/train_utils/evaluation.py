import time
import numpy as np
from tqdm import tqdm
from stable_baselines3 import SAC
from prettytable import PrettyTable

from src.agents import RandomAgent, SilentAgent, DQNAgent
from src.environments import QLDPCTrainEnv, QLDPCEvalEnv
from src.read_config import ConfigParser


def render_evaluation_episode(config, model_checkpoint, max_episode_steps=100):

    model = SAC.load(model_checkpoint, device="cuda")

    env = QLDPCTrainEnv(config)

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


def run_baselines(config):

    silent_agent = SilentAgent(config)
    random_agent = RandomAgent(config)

    results = {}
    for agent, name in [(silent_agent, "Silent Agent"), (random_agent, "Random Agent")]:
        env = QLDPCEvalEnv(config)

        total_rewards = []
        for i in tqdm(range(10_000), desc=f"Evaluating {name}"):
            obs, info = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.select_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

            total_rewards.append(total_reward)

        results[name] = total_rewards

    return results


def benchmark_env(config):

    start = time.time()
    env = QLDPCEvalEnv(config, assert_env=True)
    env.render(mode="edge_info")

    agent = DQNAgent(env, config)
    end = time.time()
    print(f"Initialization took {end - start:.5f} seconds")

    obs, info = env.reset()

    step_times = []
    agent_times = []
    buffer_times = []
    train_times = []
    loop_times = []
    for _ in tqdm(range(10_000), desc="Benchmarking environment and agent"):

        loop_start = time.time()
        action = agent.select_action(obs)
        end = time.time()
        agent_times.append(end - loop_start)

        start = time.time()
        next_obs, reward, terminated, truncated, info = env.step(action)
        end = time.time()
        step_times.append(end - start)

        start = time.time()
        agent.replay_buffer.push(obs, action, reward, next_obs, terminated or truncated)
        end = time.time()
        buffer_times.append(end - start)

        start = time.time()
        agent.train_step()
        loop_end = time.time()
        train_times.append(loop_end - start)

        obs = next_obs
        loop_times.append(loop_end - loop_start)


    t = PrettyTable(["Component", "Avg Time (s)", "it/s"])

    t.add_row(["Environment Step", f"{np.mean(step_times[1:]):.5f}", f"{1/np.mean(step_times[1:]):.2f}"])
    t.add_row(["Agent Action Selection", f"{np.mean(agent_times[1:]):.5f}", f"{1/np.mean(agent_times[1:]):.2f}"])
    t.add_row(["Buffer Push", f"{np.mean(buffer_times[1:]):.5f}", f"{1/np.mean(buffer_times[1:]):.2f}"])
    t.add_row(["Agent Training Step", f"{np.mean(train_times[1:]):.5f}", f"{1/np.mean(train_times[1:]):.2f}"])
    t.add_row(["Total Loop", f"{np.mean(loop_times[1:]):.5f}", f"{1/np.mean(loop_times[1:]):.2f}"])

    print(t)


def evaluate_agent(config: ConfigParser, state_dict: dict):
    env = QLDPCEvalEnv(config)
    agent = DQNAgent(env, config, evaluation_mode=True)
    agent.model.load_state_dict(state_dict)

    lengths = []
    for _ in tqdm(range(config.num_eval_episodes), desc="Evaluating Agent", leave=False):
        obs, info = env.reset()
        done = False

        episode_length = 0
        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_length += 1

        lengths.append(episode_length)

    return np.mean(lengths)