
from tqdm import tqdm
import numpy as np
import time
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
import optuna

from environments import MultivariateBicycleCode

import json

class RewardTrackerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.mean_rewards = []

    def _on_step(self):
        # SB3 logs episode reward in info dict under 'episode' (if env is wrapped)
        info = self.locals["infos"][0]

        if "episode" in info:
            ep_reward = info["episode"]["r"]
            self.episode_rewards.append(ep_reward)

            # Compute rolling mean over last N episodes
            window = 20  # choose any window size
            mean_r = np.mean(self.episode_rewards[-window:])
            self.mean_rewards.append(mean_r)

        return True


def get_agent(agent_name, env):
    if agent_name == "dqn":
        return DQN(env=env, policy="MlpPolicy")
    elif agent_name == "ppo":
        return PPO("MlpPolicy", env)

    raise ValueError(f"Unknown agent: {agent_name}")


def initialize_agents(config, env):

    defender = SAC(
        policy="MlpPolicy",
        env=env,
        device="cuda",
        **config
    )

    adversary = SAC(
        policy="MlpPolicy",
        env=env,
        device="cuda",
        **config
    )

    return defender, adversary



def single_agent_training_loop(
    model_config,
    code_config,
    model_checkpoint=None
):

    rewards = {"Defender": []}
    for _ in range(model_config["n_repetitions"]):

        env = MultivariateBicycleCode(**code_config)

        defender, adversary = initialize_agents(model_config["params"], env)

        callback = RewardTrackerCallback()
        defender.learn(total_timesteps=model_config["pretrain_timesteps"], progress_bar=True, callback=callback)

        rewards["Defender"].append(callback.episode_rewards)

    # Save agent
    if model_checkpoint is not None:
        defender.save(model_checkpoint)

    return rewards


def adversarial_training_loop(
    model_config,
    code_config,
    pretrain_timesteps=None
):

    pretrain_timesteps = pretrain_timesteps if pretrain_timesteps is not None else model_config["pretrain_timesteps"]

    rewards = {"Adversary": [], "Defender": [], "episode_steps": []}

    for _ in range(model_config["n_repetitions"]):

        env = MultivariateBicycleCode(**code_config)

        defender, adversary = initialize_agents(model_config, env)

        callback = RewardTrackerCallback()
        defender.learn(total_timesteps=pretrain_timesteps, progress_bar=True, callback=callback)
        adversary.learn(total_timesteps=0)

        all_rewards_adversary = []
        all_rewards_defender = []
        episode_steps = []

        current_reward_adversary = 0
        current_reward_defender = 0
        obs, _ = env.reset()

        for step in tqdm(range(model_config["num_timesteps"])):

            actor = defender if env.current_player == 0 else adversary

            action, _ = actor.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action, single_player=False)
            done = terminated or truncated

            # Required for SB3 replay buffer which expects info to be a list of dicts
            info = [info]

            actor.replay_buffer.add(obs, next_obs, action, reward, done, info)

            if step % model_config["train_frequency"] == 0:
                if actor.replay_buffer.size() >= actor.learning_starts:
                    actor.train(batch_size=32, gradient_steps=1)

            obs = next_obs

            if env.current_player == 0:
                current_reward_adversary += reward
            else:
                current_reward_defender += reward

            if done:
                episode_steps.append(info[0]["episode_steps"])
                obs, _ = env.reset()
                all_rewards_adversary.append(current_reward_adversary)
                all_rewards_defender.append(current_reward_defender)
                current_reward_adversary = 0
                current_reward_defender = 0

        rewards["Adversary"].append(all_rewards_adversary)
        rewards["Defender"].append(all_rewards_defender)
        rewards["episode_steps"].append(episode_steps)
    return rewards


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


def objective(trial: optuna.Trial, model_config, code_config) -> float:
    """
    Objective function for Optuna.
    """
    env = MultivariateBicycleCode(**code_config)

    # Sample hyperparameters
    hyperparams = sample_sac_params(trial)
    model = SAC("MlpPolicy", env, verbose=0, **hyperparams)

    # Train
    callback = RewardTrackerCallback()
    model.learn(total_timesteps=50_000, callback=callback)

    # Evaluate mean reward over last 100 episodes
    mean_reward = np.mean(callback.episode_rewards[-100:])

    return mean_reward