from time import perf_counter
import numpy as np
from prettytable import PrettyTable
import torch
from torch.profiler import profile, record_function, ProfilerActivity

from tqdm import tqdm


from src.environment import QECEnv
from src.agents import SACAgent

def benchmark_env(config):
    start = perf_counter()
    env = QECEnv(config)

    if config.verbose:
        env.render(mode="edge_info")

    agent = SACAgent(env, config)
    end = perf_counter()
    print(f"Initialization took {end - start:.5f} seconds")

    obs, info = env.reset()

    activities = [ProfilerActivity.CPU]

    if agent.device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    def trace_handler(prof):
        sort_key = "self_cuda_time_total" if agent.device.type == "cuda" else "self_cpu_time_total"
        print(prof.key_averages(group_by_input_shape=True).table(sort_by=sort_key,row_limit=30,))
        prof.export_chrome_trace("rl_profile.json")

    num_steps = 100 + config.batch_size

    with profile(
            activities=activities,
            schedule=torch.profiler.schedule(skip_first=config.batch_size + 2,wait=1,warmup=2,active=10,repeat=1,),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
    ) as prof:
        for _ in tqdm(range(num_steps),desc="Profiling RL agent",):
            with record_function("Agent/select_action"):
                action, _ = agent.select_action(obs)

            with record_function("Environment/step"):
                next_obs, reward, terminated, truncated, info = (env.step(action))

            state = obs

            with record_function("ReplayBuffer/push"):
                agent.replay_buffer.push(state,action,reward,next_obs, terminated or truncated)

            with record_function("Environment/reset"):
                obs, info = env.reset()

            with record_function("Agent/train_step"):
                agent.train_step()
            prof.step()