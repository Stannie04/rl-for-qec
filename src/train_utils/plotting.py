import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
import torch
from src.environment import QLDPCEnv
from src.train_utils.datasets import load_shots
from src.train_utils.evaluation import get_agent

def plot_results(results, baselines, config, window=100, title="results/results.png"):

    plt.figure(figsize=(12, 6))
    for name, runs in results.items():

        mean, ci95 = get_confidence_bounds(runs, window)

        x_len = len(mean)
        progress = np.linspace(0, config["num_timesteps"], x_len)

        plt.plot(progress, mean, label=f"{name} Mean", linewidth=2)

        if len(runs) != 1:
            plt.fill_between(progress, mean - ci95, mean + ci95, alpha=0.3, label=f"{name} 95% CI")

    for name, run in baselines.items():
        line = [np.mean(run)]*config["num_timesteps"]
        plt.plot(line, label=f"{name} Mean", linewidth=2, linestyle='--')


    plt.title("RL Agent Performance on Multivariate Bicycle Code Environment")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()
    plt.savefig(title)


def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed
    window: size of the smoothing window '''

    return savgol_filter(y,window,poly) if window > 0 else y


def get_confidence_bounds(results, window=100):
    """
    Calculates smoothed mean and 95% confidence intervals for the results.
    """

    # Convert to list of arrays
    results = [np.array(r) for r in results]

    if len(results) == 1:
        return smooth(results[0], window=window), np.zeros_like(results[0])

    # Find minimum length
    min_len = min(len(r) for r in results)

    # Truncate all runs
    results = np.array([r[:min_len] for r in results])

    mean = smooth(np.mean(results, axis=0), window=window)
    ci95 = 1.96 * np.std(results, axis=0) / np.sqrt(len(results))

    return mean, smooth(ci95, window=window)


def render_example_environment(config):
    env = QLDPCEnv(config)

    for l in env.code.logical_x:
        initial_x, initial_z = env.code.get_logical_state()
        num = len(torch.argwhere(l == 1).flatten())
        for i, j in enumerate(torch.argwhere(l == 1).flatten()):
            print(f"{i+1} / {num}")
            env.code.flip(j)
            env.code.update_graph(env.curriculum_error_rate)

            current_x, current_z = env.code.get_logical_state()
            x_changed = not torch.equal(initial_x, current_x)
            z_changed = not torch.equal(initial_z, current_z)
            print(f"Flipped qubit {j.item()}: Logical X changed: {x_changed}, Logical Z changed: {z_changed}")


            print(f"Error free: {env.code.is_error_free()}\n")

        env.render()

        env.reset()


def render_mistakes(config):
    agent_name = "bp"

    mistakes = load_shots(config, dataset_type="mistakes", noise_model="bit_flip", agent_name=agent_name)
    env = QLDPCEnv(config, mistakes)
    agent = get_agent(config, env, agent_name)

    for idx, shot in enumerate(mistakes):
        print(f"Rendering mistake {idx + 1}/{len(mistakes)}")
        obs, info = env.reset_with_error_pattern(shot[0], shot[1])

        # Evaluate the agent on this mistake
        error_pred = agent.select_action(obs, evaluate=True)[0]
        print(f"Action: {error_pred}")
        env.code.render_subgraph()
        env.step(error_pred)
        env.code.render_subgraph()

