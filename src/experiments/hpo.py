import optuna
import numpy as np
import tempfile
import os

import copy
import gc
import torch

from src.environment import QECEnv
from src.agents import SACAgent
from src.train_utils import get_weighted_ler_on_eval_set, create_dataset_from_random_shots_labelled, create_dataset_from_curriculum
from .train_rl import rl_training_loop


def sample_sac_params(trial: optuna.Trial) -> dict:
    """
    Define the search space for SAC hyperparameters.
    """
    return {
        "curriculum_start_error_rate_144": trial.suggest_float("curriculum_start_error_rate_144", 5e-4, 3e-3, log=True),
        "curriculum_end_error_rate_144": trial.suggest_float("curriculum_end_error_rate_144", 1e-2, 4e-2, log=True),
        "curriculum_warmup_steps": trial.suggest_int("curriculum_warmup_steps", 2000, 10000, log=True),
        "entropy_factor": trial.suggest_float("entropy_factor", 0.2, 0.7),
        "initial_alpha": trial.suggest_float("initial_alpha", 0.02, 0.3, log=True),
        "max_episode_length": trial.suggest_int("max_episode_length", 5, 25),
    }



def objective(trial, base_config, eval_set, last_k=3):
    config = copy.deepcopy(base_config)
    config.update(sample_sac_params(trial))
    config.evaluate_during_training = True # required for pruning and the last-k average
    config.wandb_logging = False

    shots = create_dataset_from_curriculum(config, num_samples=config.num_timesteps)

    env = QECEnv(config, shots)
    agent = SACAgent(env, config)
    history = []

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = os.path.join(tmpdir, "eval_agent.pt")

        def _eval_fn(step, agent):
            return get_weighted_ler_on_eval_set(config, agent, ckpt, eval_set)

        def _on_eval(step, ler):
            history.append(ler)
            trial.report(ler, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            rl_training_loop(env, agent, config, eval_fn=_eval_fn, on_eval=_on_eval)
            return float(np.mean(history[-last_k:]))
        finally:
            del agent, env
            gc.collect()
            torch.cuda.empty_cache()


def optimize_hyperparameters(config):
    eval_set = create_dataset_from_random_shots_labelled(config, num_samples=config.hpo_num_eval_samples, error_rates=config.hpo_error_rates)

    study = optuna.create_study(
        direction="minimize",
        study_name="hpo",
        storage=f"sqlite:///hpo_study_{config.code_name}.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3),
    )

    study.optimize(lambda trial: objective(trial, config, eval_set), n_trials=config.hpo_num_trials)

    print("Best hyperparameters:", study.best_params)
    return study.best_params