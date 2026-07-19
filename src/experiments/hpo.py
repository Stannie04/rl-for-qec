import optuna
import numpy as np

from src.environment import QLDPCEnv
from src.agents import SACAgent
from src.train_utils import evaluate_agent


def sample_sac_params(trial: optuna.Trial) -> dict:
    """
    Define the search space for SAC hyperparameters.
    """
    return {
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "actor_learning_rate": trial.suggest_float("actor_learning_rate", 5e-6, 1e-3, log=True),
        "critic_learning_rate": trial.suggest_float("critic_learning_rate", 5e-6, 1e-3, log=True),
        "alpha_learning_rate": trial.suggest_float("alpha_learning_rate", 5e-6, 1e-3, log=True),
        "replay_buffer_capacity": trial.suggest_int("replay_buffer_capacity", 10000, 1_000_000, log=True),
        "gamma": trial.suggest_float("gamma", 0.90, 0.999),
        "tau": trial.suggest_float("tau", 0.001, 0.02),
        "train_frequency": trial.suggest_categorical("train_frequency", [1, 4, 8]),
        "initial_alpha": trial.suggest_float("ent_coef", 1e-2, 3e-1, log=True),
        "max_episode_length": trial.suggest_categorical("max_episode_length", [10, 50, 100]),
        "curriculum_start_error_rate": trial.suggest_float("curriculum_start_error_rate", 0.01, 0.1),
        "curriculum_end_error_rate": trial.suggest_float("curriculum_end_error_rate", 0.001, 0.01),
        "curriculum_warmup_steps": trial.suggest_categorical("curriculum_warmup_steps", [1000, 5000, 10000]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512])
    }



def objective(trial: optuna.Trial, config) -> float:
    """
    Objective function for Optuna.
    """
    env = QLDPCEnv(config)

    # Sample hyperparameters
    hyperparams = sample_sac_params(trial)
    config.update(hyperparams)
    agent = SACAgent(env, config)

    # Train
    rewards = []
    scores = []
    obs, info = env.reset()
    for step in range(config.num_timesteps):

        action, probs = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        agent.replay_buffer.push(obs, action, reward, next_obs, terminated or truncated)
        obs = next_obs

        agent.train_step()

        if terminated or truncated:
            obs, info = env.reset()

        if step % config.steps_between_evaluation == 0:
            scores.append(evaluate_agent(config, agent)["Evaluation/Score"])

    # Evaluate mean reward over last 100 episodes
    eval_scores = min(100, len(scores))
    mean_score = np.mean(scores[:eval_scores])

    return mean_score


def optimize_hyperparameters(config, n_trials=1000):
    study = optuna.create_study(direction="maximize", study_name="hpo", storage=f"sqlite:///hpo_study_{config.n}_{config.k}_{config.d}.db", load_if_exists=True)
    study.optimize(lambda trial: objective(trial, config), n_trials=n_trials)

    print("Best hyperparameters:", study.best_params)
    return study.best_params