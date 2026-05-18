import wandb
import numpy as np
import time

from wandb import Histogram

from src.environment import QLDPCEnv
from src.agents import DQNAgent, SACAgent, BPAgent, BPOSDAgent
from src.train_utils import evaluate_agent, CurriculumScheduler


def single_agent_training_loop(env, agent, config):

    start_time = time.time()

    curriculum = CurriculumScheduler(config)

    rewards = []
    lengths = []

    episode_reward = 0
    obs, info = env.reset()
    start_errors = info["num_errors"]

    for step in range(config.num_timesteps):

        action, probs = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.replay_buffer.push(obs, action, reward, next_obs, (terminated or truncated))
        obs = next_obs
        if step % config.train_frequency == 0:
            actor_loss, critic_loss, alpha = agent.train_step()

            wandb.log({"Loss/Actor Loss": actor_loss, "Loss/Critic Loss": critic_loss, "Loss/Alpha": alpha}, step=step)
            wandb.log({
                "Actions/Action taken": action.item(),
                "Actions/Accuracy": np.mean(info["correct_actions"][-100:]),
                "Actions/Repeated Actions": np.mean(info["repeated_actions"][-100:])}, step=step)

            for i in range(env.code.n_data):
                wandb.log({f"Probabilities/Qubit {i}": probs.flatten().cpu().numpy()[i]}, step=step)

        curriculum.step(env, step)
        episode_reward += reward

        wandb.log({"Monitoring/Elapsed Time": time.time() - start_time, "Monitoring/Error Rate": env.curriculum_error_rate, "Monitoring/Number of Errors": info["num_errors"]}, step=step)

        if terminated or truncated:
            rewards.append(episode_reward.item())
            wandb.log({"Train/Episode Reward": episode_reward.item(), "Train/Episode Steps": info["episode_steps"]}, step=step)
            wandb.log({"Decoding Ability/Errors at End of Episode": info["num_errors"],
                            "Decoding Ability/Errors at Start of Episode": start_errors,
                            "Decoding Ability/Errors decoded" : start_errors - info["num_errors"]}, step=step)

            episode_reward = 0
            obs, info = env.reset()
            start_errors = info["num_errors"]

        if step % config.steps_between_evaluation == 0:
            log = evaluate_agent(config, agent)
            wandb.log(log, step=step)

    return lengths, rewards


def train(config) -> dict:
    all_rewards = []
    all_lengths = []

    wandb.init(project=config.wandb_project, tags=[f"{config.agent_name}", f"{config.code_name}"], config=config.__dict__, dir="/tmp/wandb")

    env = QLDPCEnv(config)
    agent = SACAgent(env, config)

    for i in range(config.n_repetitions):
        print(f"Starting training run {i+1}/{config.n_repetitions}")

        lengths, rewards = single_agent_training_loop(env, agent, config)
        all_rewards.append(rewards)
        all_lengths.append(lengths)

    wandb.finish()

    agent.save(f"checkpoints/{config.agent_name}_{config.code_name}.pt")

    return {"Length": all_lengths, "Reward": all_rewards}