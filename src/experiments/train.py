import wandb
import numpy as np
import time

from wandb import Histogram

from src.environment import QLDPCEnv
from src.agents import DQNAgent, SACAgent, BPAgent, BPOSDAgent
from src.train_utils import evaluate_agent, CurriculumScheduler


def get_reset_logs(episode_reward, info, start_errors):
    return {
        "Train/Episode Reward": episode_reward.item(),
        "Train/Episode Steps": info["episode_steps"],
        "Decoding Ability/Errors at End of Episode": info["num_errors"],
        "Decoding Ability/Errors at Start of Episode": start_errors,
        "Decoding Ability/Errors decoded" : start_errors - info["num_errors"]
    }


def log_wandb_data(step, env, start_time, probs, **kwargs):
    monitoring_logs = {"Monitoring/Elapsed Time": time.time() - start_time,
                       "Monitoring/Error Rate": env.curriculum_error_rate}

    qubit_probability_logs = {}
    for i in range(env.code.n_data):
        qubit_probability_logs[f"Probabilities/Qubit {i}"] = probs.flatten().cpu().numpy()[i]

    all_logs = {**monitoring_logs, **qubit_probability_logs, **kwargs}
    wandb.log(all_logs, step=step)


def single_agent_training_loop(env, agent, config):

    start_time = time.time()

    curriculum = CurriculumScheduler(config)

    episode_reward = 0
    obs, info = env.reset()
    start_errors = info["num_errors"]

    for step in range(config.num_timesteps):
        train_step_logs, done_logs, eval_logs = {}, {}, {}

        action, probs = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.replay_buffer.push(obs, action, reward, next_obs, (terminated or truncated))
        obs = next_obs

        if step % config.train_frequency == 0:
            train_step_logs = agent.train_step()

        curriculum.step(env, step)
        episode_reward += reward

        if terminated or truncated:
            done_logs = get_reset_logs(episode_reward, info, start_errors)

            episode_reward = 0
            obs, info = env.reset()
            start_errors = info["num_errors"]

        if config.evaluate_during_training and step % config.steps_between_evaluation == 0:
            eval_logs = evaluate_agent(config, agent)

        if config.wandb_logging:
            log_wandb_data(step, env, start_time, probs, **train_step_logs, **eval_logs, **done_logs)


def train(config):

    if config.wandb_logging:
        wandb.init(project=config.wandb_project, tags=[f"{config.agent_name}", f"{config.code_name}"], config=config.__dict__, dir="/tmp/wandb")

    env = QLDPCEnv(config)
    agent = SACAgent(env, config)

    for i in range(config.n_repetitions):
        print(f"Starting training run {i+1}/{config.n_repetitions}")
        single_agent_training_loop(env, agent, config)

    if config.wandb_logging:
        wandb.finish()

    agent.save(f"checkpoints/{config.agent_name}_{config.code_name}.pt")