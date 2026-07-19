from src.agents import NeuralBPEncoder, SLAgent
from src.agents.encoders import CGNNEncoder
from src.environment import QLDPCEnv
from src.train_utils import create_dataset_from_curriculum, create_dataset_from_uniform_shots, create_dataset_from_pretrained_encoder_mistakes
import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import os


def train_sl(config):

    encoder_type = "nbp" if config.use_neural_bp else "cgnn"

    checkpoint_dir = f"sl_{encoder_type}{config.neural_bp_iterations}_{config.code_name}_{config.wandb_run_name}_{int(time.time())}"
    os.makedirs(f"checkpoints/{checkpoint_dir}", exist_ok=True)

    print(f"Pretraining {encoder_type} encoder for code {config.code_name} for {config.num_pretrain_timesteps} steps with learning rate {config.encoder_learning_rate}.")

    shots = create_dataset_from_curriculum(config, config.num_pretrain_timesteps)

    env = QLDPCEnv(config, shots)

    model = SLAgent(config, env).to(config.device)
    opt = torch.optim.Adam(model.parameters(), lr=config.encoder_learning_rate)

    for i in tqdm(range(config.num_pretrain_timesteps)):
        obs, _ = env.reset()
        error_true = env.code.x_errors.float()

        error_pred = model(obs)
        loss = F.binary_cross_entropy(error_pred, error_true)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (i+1) % config.steps_between_pretrain_evaluation == 0:
            evaluate_pretrained_encoder(config, model, checkpoint_dir, threshold=0.5, step=i+1)


def evaluate_pretrained_encoder(config, model, checkpoint_dir, threshold=None, step=None):
    num_samples_per_error = 1000
    max_error = 4
    shots = create_dataset_from_uniform_shots(config, num_samples_per_error, max_error=max_error)
    env = QLDPCEnv(config, shots)

    topk_full_counts = [0] * max_error

    for i in range(max_error):
        error_weight = i + 1
        all_losses = []

        for _ in range(num_samples_per_error):
            obs, _ = env.reset()
            error_true = env.code.x_errors.float()

            with torch.no_grad():
                error_pred = model(obs)

            all_losses.append(F.binary_cross_entropy(error_pred, error_true))
            true_indices = torch.nonzero(error_true).flatten()

            if threshold is not None:
                predicted_indices = torch.nonzero(error_pred > threshold).flatten()

                pred_sorted = torch.sort(predicted_indices).values
                true_sorted = torch.sort(true_indices).values

                if torch.equal(pred_sorted, true_sorted):
                    topk_full_counts[error_weight - 1] += 1

    full_rate_1 = topk_full_counts[0] / num_samples_per_error
    full_rate_2 = topk_full_counts[1] / num_samples_per_error
    full_rate_3 = topk_full_counts[2] / num_samples_per_error
    full_rate_4 = topk_full_counts[3] / num_samples_per_error
    model.save(f"checkpoints/{checkpoint_dir}/{model.encoder.name}_{step}_{full_rate_1:.2f}_{full_rate_2:.2f}_{full_rate_3:.2f}_{full_rate_4:.2f}.pt")


def post_pretrain_evaluation(config, model):
    create_dataset_from_pretrained_encoder_mistakes(config, model, save=True)