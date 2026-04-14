import numpy as np
from tqdm import tqdm
import optuna

import wandb

from src.environments import QLDPCTrainEnv
from src.agents import DQNAgent
from src.train_utils import evaluate_agent


def single_agent_training_loop(env, agent, config):

    rewards = []
    lengths = []

    episode_reward = 0
    obs, info = env.reset()

    for step in range(config.num_timesteps):

        action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.replay_buffer.push(obs, action, reward, next_obs, terminated or truncated)
        obs = next_obs

        agent.train_step()

        episode_reward += reward

        wandb.log({"timestep": step, "reward": reward}, step=step)
        wandb.log({k: v for k, v in info.items() if k != "episode_steps"}, step=step)

        if terminated or truncated:
            rewards.append(episode_reward)
            wandb.log({"train_episode_reward": episode_reward, "episode_steps": info["episode_steps"]}, step=step)

            episode_reward = 0
            obs, info = env.reset()

        if (step + 1) % agent.target_update_freq == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        if step % config.steps_between_evaluation == 0:
            state_dict = agent.model.state_dict()
            episode_length = evaluate_agent(config, state_dict)
            lengths.append([episode_length])

            wandb.log({"eval_episode_length": episode_length}, step=step)

    return lengths, rewards


def train_dqn(config) -> dict:
    all_rewards = []
    all_lengths = []

    wandb.init(project=config.wandb_project, tags=[f"{config.agent_name}_{config.code_name}"], config=config.__dict__, dir="/tmp/wandb")

    env = QLDPCTrainEnv(config)
    agent = DQNAgent(env, config)

    for _ in range(config.n_repetitions):
        print(f"Starting DQN training run {_+1}/{config.n_repetitions}")

        lengths, rewards = single_agent_training_loop(env, agent, config)
        all_rewards.append(rewards)
        all_lengths.append(lengths)

    wandb.finish()

    return {"Length": all_lengths, "Reward": all_rewards}