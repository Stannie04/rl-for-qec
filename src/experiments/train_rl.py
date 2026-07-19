import wandb
import numpy as np
import time
import os
from wandb import Histogram
from tqdm import tqdm
from src.environment import QLDPCEnv
from src.agents import SACAgent, NeuralBPEncoder, SLAgent
from src.train_utils import evaluate_agent, CurriculumScheduler, load_shots, create_dataset_from_nonzero_shots, create_dataset_from_curriculum, create_dataset_from_pretrained_encoder_mistakes
import torch

def get_reset_logs(episode_reward, info, start_errors):
    return {
        "Train/Episode Reward": episode_reward,
        "Train/Episode Steps": info["episode_steps"],
        "Decoding Ability/Errors at End of Episode": info["num_errors"],
        "Decoding Ability/Errors at Start of Episode": start_errors,
        "Decoding Ability/Errors decoded" : start_errors - info["num_errors"],
        "Decoding Ability/Percentage of Errors Decoded": (start_errors - info["num_errors"]) / max(1, start_errors),
    }


def log_wandb_data(step, env, start_time, probs, **kwargs):
    monitoring_logs = {"Monitoring/Elapsed Time": time.time() - start_time,
                       "Monitoring/Error Rate": env.curriculum_error_rate}

    qubit_probability_logs = {}
    for i in range(env.code.n_data):
        qubit_probability_logs[f"Probabilities/Qubit {i}"] = probs.flatten().cpu().numpy()[i]

    all_logs = {**monitoring_logs, **qubit_probability_logs, **kwargs}
    wandb.log(all_logs, step=step)


def single_agent_training_loop(env, agent, config, checkpoint_dir=None):

    start_time = time.time()
    curriculum = CurriculumScheduler(config)
    episode_reward = 0
    obs, info = env.reset()
    start_errors = info["num_errors"]
    best_model_ler = float("inf")

    for step, _ in enumerate(tqdm(env.shots)):
        train_step_logs, done_logs, eval_logs = {}, {}, {}

        action, probs = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.replay_buffer.push(obs, action, reward, next_obs, (terminated or truncated))
        obs = next_obs

        if step % config.train_frequency == 0:
            train_step_logs = agent.train_step()

        curriculum.step(env, step)
        episode_reward += reward

        if truncated or (terminated and (not config.use_noop_head or action == env.code.no_op_index)):
            done_logs = get_reset_logs(episode_reward, info, start_errors)

            episode_reward = 0
            obs, info = env.reset()
            start_errors = info["num_errors"]

        if config.evaluate_during_training and step % config.steps_between_evaluation == 0:
            ler, best_model_ler = evaluate_agent(config, step, best_model_ler, agent, checkpoint_dir)
            eval_logs = {"Evaluation/Logical Error Rate": ler, "Evaluation/Best Logical Error Rate": best_model_ler}

        if config.wandb_logging:
            log_wandb_data(step, env, start_time, probs, **train_step_logs, **eval_logs, **done_logs)

    evaluate_agent(config, len(env.shots), best_model_ler, agent, checkpoint_dir)


def train_rl(config):

    start_time = time.time()

    if config.wandb_logging:
        wandb.init(project=config.wandb_project, name=f"{config.wandb_run_name}_{int(start_time)}", tags=[f"{config.agent_name}", f"{config.code_name}"], config=config.__dict__, dir="/tmp/wandb")

    encoder = "nbp" if config.use_neural_bp else "cgnn"
    checkpoint_dir = f"checkpoints/{config.agent_name}_{encoder}_{config.code_name}_{config.wandb_run_name}_{int(start_time)}"
    os.mkdir(checkpoint_dir)

    try:
        shots = create_dataset_from_curriculum(config, num_samples=config.num_timesteps, noise_model="bit_flip", with_mistakes=False, save=False)
        env = QLDPCEnv(config, shots)
        agent = SACAgent(env, config)

        single_agent_training_loop(env, agent, config, checkpoint_dir)

        if config.wandb_logging:
            wandb.finish()

    except KeyboardInterrupt:
        if not os.listdir(checkpoint_dir):
            os.rmdir(checkpoint_dir)

        else:
            agent.save(f"{checkpoint_dir}/interrupted.pt")

        if config.wandb_logging:
            wandb.finish()
        raise