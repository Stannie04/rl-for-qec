from time import perf_counter
import numpy as np
from prettytable import PrettyTable
import torch

from tqdm import tqdm


from src.environment import QLDPCEnv
from src.agents import SACAgent

def benchmark_env(config):
    start = perf_counter()
    env = QLDPCEnv(config)

    if config.verbose:
        env.render(mode="edge_info")

    agent = SACAgent(env, config)
    end = perf_counter()
    print(f"Initialization took {end - start:.5f} seconds")

    obs, info = env.reset()

    step_times = []
    reset_times = []
    agent_times = []
    buffer_times = []
    train_times = []
    loop_times = []

    for _ in tqdm(range(10_000), desc="Benchmarking environment and agent"):
        # loop_start = perf_counter()
        # action, _ = agent.select_action(obs)
        # end = perf_counter()
        # agent_times.append(end - loop_start)

        action = torch.tensor(env.action_space.sample())
        start = perf_counter()
        next_obs, reward, terminated, truncated, info = env.step(action)
        end = perf_counter()
        step_times.append(end - start)

        # start = perf_counter()
        # obs, info = env.reset()
        # end = perf_counter()
        # reset_times.append(end - start)

        # start = perf_counter()
        # agent.replay_buffer.push(obs, action, reward, next_obs, terminated or truncated)
        # end = perf_counter()
        # buffer_times.append(end - start)
        #
        # start = perf_counter()
        # agent.train_step()
        # loop_end = perf_counter()
        # train_times.append(loop_end - start)
        #
        # obs = next_obs
        # loop_times.append(loop_end - loop_start)


    t = PrettyTable(["Component", "Avg Time (s)", "it/s"])

    t.add_row(["Environment Step", f"{np.mean(step_times[1:]):.5f}", f"{1/np.mean(step_times[1:]):.2f}"])
    # t.add_row(["Environment Reset", f"{np.mean(reset_times[1:]):.5f}", f"{1/np.mean(reset_times[1:]):.2f}"])
    # t.add_row(["Agent Action Selection", f"{np.mean(agent_times[1:]):.5f}", f"{1/np.mean(agent_times[1:]):.2f}"])
    # t.add_row(["Buffer Push", f"{np.mean(buffer_times[1:]):.5f}", f"{1/np.mean(buffer_times[1:]):.2f}"])
    # t.add_row(["Agent Training Step", f"{np.mean(train_times[1:]):.5f}", f"{1/np.mean(train_times[1:]):.2f}"])
    # t.add_row(["Total Loop", f"{np.mean(loop_times[1:]):.5f}", f"{1/np.mean(loop_times[1:]):.2f}"])

    print(t)
