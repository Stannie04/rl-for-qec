import wandb
import numpy as np
import time
import os
from wandb import Histogram
from tqdm import tqdm
from src.environment import QLDPCEnv
from src.agents import DQNAgent, SACAgent, BPAgent, BPOSDAgent, NeuralBPPretrainer, NeuralBPEncoder
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


def train(config):

    start_time = time.time()

    if config.wandb_logging:
        wandb.init(project=config.wandb_project, name=f"{config.wandb_run_name}_{int(start_time)}", tags=[f"{config.agent_name}", f"{config.code_name}"], config=config.__dict__, dir="/tmp/wandb")

    checkpoint_dir = f"checkpoints/{config.agent_name}_{config.code_name}_{config.wandb_run_name}_{int(start_time)}"
    os.mkdir(checkpoint_dir)

    try:
        shots = create_dataset_from_curriculum(config, num_samples=config.num_timesteps, noise_model="bit_flip", with_mistakes=False, save=False)
        env = QLDPCEnv(config, shots)
        agent = SACAgent(env, config)

        for i in range(config.n_repetitions):
            print(f"Starting training run {i+1}/{config.n_repetitions}")
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


def finetune(config):

    shots = load_shots(config, dataset_type="mistakes", noise_model="bit_flip", agent_name="bp", num_epochs=100)
    env = QLDPCEnv(config, shots)

    encoder = NeuralBPEncoder(config, env)
    agent = NeuralBPPretrainer(encoder, config, env.code.n_data).to(config.device)
    checkpoint = torch.load(f"checkpoints/encoder_early.pt", map_location=config.device)
    agent.encoder.load_state_dict(checkpoint["encoder"])
    agent.output_layer.load_state_dict(checkpoint["output_layer"])

    opt = torch.optim.Adam(agent.parameters(), lr=config.encoder_learning_rate)
    pbar = tqdm(env.shots)

    for _ in pbar:
        obs, info = env.reset()
        error_true = env.code.x_errors.float()
        error_pred = agent(obs)

        loss = torch.nn.functional.binary_cross_entropy(error_pred, error_true)
        opt.zero_grad()
        loss.backward()
        opt.step()

        pbar.set_description(f"loss: {loss.item():.4f}")

    agent.save(f"checkpoints/encoder_early_finetuned.pt")

    create_dataset_from_pretrained_encoder_mistakes(config, agent, save=True)

    # if config.wandb_logging:
    #     wandb.init(project=config.wandb_project, name=config.wandb_run_name, tags=[f"{config.agent_name}", f"{config.code_name}"], config=config.__dict__, dir="/tmp/wandb")
    #
    # shots = load_shots(config, dataset_type="mistakes")
    # env = QLDPCEnv(config, shots)
    # agent = SACAgent(env, config)
    # checkpoint = torch.load(f"checkpoints/{config.agent_name}_{config.code_name}.pt", map_location=config.device)
    # agent.actor.load_state_dict(checkpoint["actor"])
    # agent.critic1.load_state_dict(checkpoint["critic1"])
    # agent.critic2.load_state_dict(checkpoint["critic2"])
    #
    # for i in range(config.n_repetitions):
    #     print(f"Starting training run {i+1}/{config.n_repetitions}")
    #     single_agent_training_loop(env, agent, config)
    #
    # if config.wandb_logging:
    #     wandb.finish()
    #
    #
    # agent.save(f"checkpoints/{config.agent_name}_{config.code_name}_finetuned.pt")