from stable_baselines3 import SAC
from environments import QLDPCTrainEnv, QLDPCEvalEnv
from agents import SilentAgent, DQNAgent
from tqdm import tqdm
import numpy as np
import time

import jax
from prettytable import PrettyTable


def render_evaluation_episode(code_config, model_checkpoint, max_episode_steps=100):

    model = SAC.load(model_checkpoint, device="cuda")

    env = QLDPCTrainEnv(**code_config, evaluation_mode=True)

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
        env = QLDPCEvalEnv(**code_config, evaluation_mode=True)

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

        return {name: total_rewards}


def benchmark_jax_env(code_config, device="cpu"):

    sample_config = code_config.copy()

    start = time.time()
    env = JaxQLDPCCode(**sample_config, device=device)
    end = time.time()
    print(f"Initialization took {end - start:.5f} seconds")

    key = jax.random.PRNGKey(0)
    obs, key = env.reset(key)

    step_times = []
    for _ in tqdm(range(1000), desc="Benchmarking JAX environment"):

        start = time.time()
        obs, key = env.step(obs, key)
        end = time.time()
        step_times.append(end - start)

    t = PrettyTable(["Component", "Avg Time (s)", "it/s"])
    t.add_row(["Environment Step", f"{np.mean(step_times[1:]):.5f}", f"{1/np.mean(step_times[1:]):.2f}"])
    print(t)



def benchmark_env(code_config, model_config, device="cpu"):

    sample_config = code_config.copy()

    start = time.time()
    env = QLDPCEvalEnv(**sample_config, device=device, assert_env=True)
    env.render(mode="edge_info")

    agent = DQNAgent(env, **model_config["params"], device=device)
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


def evaluate_agent(code_config, model_params, device, state_dict):
    env = QLDPCEvalEnv(**code_config, device=device)
    agent = DQNAgent(env, device=device, evaluation_mode=True, **model_params)
    agent.model.load_state_dict(state_dict)

    obs, info = env.reset()
    done = False

    episode_length = 0
    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_length += 1


    return episode_length
