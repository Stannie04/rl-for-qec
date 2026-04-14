import numpy as np
from tqdm import tqdm
import optuna

import wandb

from src.environments import QLDPCTrainEnv
from src.agents import DQNAgent
from .evaluation import evaluate_agent


def single_agent_training_loop(env, agent, config):

    rewards = []
    lengths = []
    obs, info = env.reset()

    pbar = tqdm(range(config.num_timesteps), desc="Training DQN Agent")
    for step in pbar:

        action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.replay_buffer.push(obs, action, reward, next_obs, terminated or truncated)
        obs = next_obs

        agent.train_step()

        rewards.append(reward)

        if terminated or truncated:
            obs, info = env.reset()

        if (step + 1) % agent.target_update_freq == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        if step % agent.eval_freq == 0:
            state_dict = agent.model.state_dict()
            episode_length = evaluate_agent(config, state_dict)
            pbar.set_description(f"Step {step} - Eval Episode Length: {episode_length}")
            lengths.append([episode_length])

    return lengths, rewards


def train_dqn(config):
    all_rewards = []
    all_lengths = []

    env = QLDPCTrainEnv(config)
    agent = DQNAgent(env, config)

    single_agent_training_loop(env, agent, config)

    return {"Length": all_lengths, "Reward": all_rewards}



def sample_sac_params(trial: optuna.Trial) -> dict:
    """
    Define the search space for SAC hyperparameters.
    """
    return {
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size": trial.suggest_categorical("buffer_size", [100000, 200000, 500000]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "tau": trial.suggest_float("tau", 0.005, 0.02),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 4, 8]),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 1e-1, log=True),
    }


def sample_dqn_params(trial: optuna.Trial) -> dict:
    """
    Define the search space for DQN hyperparameters.
    """
    return {
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size": trial.suggest_categorical("buffer_size", [100000, 200000, 500000]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 4, 8]),
        "target_update_freq": trial.suggest_categorical("target_update_freq", [1000, 5000, 10000]),
        "epsilon_decay": trial.suggest_categorical("epsilon_decay", [1000, 5000, 10000, 50000, 100000]),
    }


def objective(trial: optuna.Trial, code_config) -> float:
    """
    Objective function for Optuna.
    """
    env = QLDPCTrainEnv(**code_config, device='cuda')

    # Sample hyperparameters
    hyperparams = sample_dqn_params(trial)
    agent = DQNAgent(env=env, **hyperparams, device='cuda')

    # Train
    rewards = []
    lengths = []
    obs, info = env.reset()
    for step in range(100_000):

        action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.replay_buffer.push(obs, action, reward, next_obs, terminated or truncated)
        obs = next_obs

        agent.train_step()

        rewards.append(reward)

        if terminated or truncated:
            obs, info = env.reset()

        if step % agent.eval_freq == 0:
            lengths.append(evaluate_agent(code_config, hyperparams, 'cuda', agent.model.state_dict()))

        if (step + 1) % agent.target_update_freq == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

    # Evaluate mean reward over last 100 episodes
    eval_len = min(100, len(lengths))
    mean_reward = np.mean(lengths[:eval_len])

    return mean_reward


def optimize_hyperparameters(code_config, n_trials=100):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, code_config), n_trials=n_trials)

    print("Best hyperparameters:", study.best_params)
    return study.best_params