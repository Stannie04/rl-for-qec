import time
from tqdm import tqdm
import numpy as np
import os
import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from src.environment import QECEnv
from src.agents import RouterAgent, RouterNetwork
from src.train_utils import load_shots, get_agent_and_inference, inference, create_dataset_from_moe_shots_labelled

class GraphDataset(Dataset):
    def __init__(self, dataset):
        self.x = dataset["x"]
        self.edge_index = dataset["edge_index"]
        self.labels = dataset["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return Data(x=self.x[idx], edge_index=self.edge_index, y=self.labels[idx].unsqueeze(0))



def moe_rl_train_loop(config, env, agent, expert_list, inference_list, train=True):
    start_time = time.time()

    checkpoint_dir = f"checkpoints/moe_{config.code_name}_{config.wandb_run_name}_{int(time.time())}"
    if train:
        os.mkdir(checkpoint_dir)

    moe_correct = 0
    losses = 0
    bp_correct = 0
    action_dist = 0
    results = {}

    for step in tqdm(range(config.moe_num_rl_steps)):
    # for step, _ in tqdm(enumerate(env.shots)):
            # error_rate = get_physical_error_rate(config, step)
            # env.curriculum_error_rate = error_rate
            obs, info = env.reset()
            pattern = env.code.number_of_overlapping_stabilizers()
            results.setdefault(pattern, {"hits": 0, "successes": 0, "action_dist": 0})

            action, log_prob = agent.select_action(obs, evaluate=not train)

            if train:
                decoder = expert_list[0] # For now, just use BP always
                inference_fn = inference_list[0] # For now, just use BP always
            else:
                decoder = expert_list[action]
                inference_fn = inference_list[action]

            success = inference_fn(decoder, env, (obs, info))

            if train:
                reward = 1 if action == success else 0
                loss = agent.update(log_prob, reward)
                losses += loss
            else:
                reward = 1 if success else 0

            moe_correct += 1 if reward > 0 else 0
            bp_correct += 1 if success else 0
            action_dist += 1 if action == 0 else 0

            results[pattern]["hits"] += 1
            results[pattern]["successes"] += 1 if reward > 0 else 0
            # results[pattern]["bp_successes"] += 1 if (reward > 0 and action == 0) or (reward < 0 and action != 0) else 0
            results[pattern]["action_dist"] += 1 if action == 0 else 0

            if train and step % 100 == 0:
                print(f"Step {step+1}/{config.moe_num_rl_steps}, MOE success rate: {moe_correct / 100}, BP success rate: {bp_correct / 100}%, BP chosen: {action_dist / 100}%")
                save_path = os.path.join(checkpoint_dir, f"moe_{step+1}_{moe_correct}.pt")
                agent.save(save_path)
                losses = 0
                moe_correct = 0
                action_dist = 0
                bp_correct = 0

    print(f"MoE Training completed in {time.time() - start_time:.2f} seconds. Success rate: {moe_correct / config.moe_num_rl_steps:.4f}")

    for pattern, pattern_results in results.items():
        total = pattern_results["hits"]
        moe_success_rate = pattern_results["successes"] / total if total > 0 else 0
        action_dist_rate = pattern_results["action_dist"] / total if total > 0 else 0
        print(f"Pattern: {pattern}, Hits: {total}, MOE Success Rate: {moe_success_rate:.4f}, Action distribution (BP chosen): {action_dist_rate:.4f}")


def sl_evaluate(config, router, test, num_experts, epoch, train_loss):
    router.eval()

    correct = torch.zeros((), device=config.device)
    tp = torch.zeros((), device=config.device)
    fp = torch.zeros((), device=config.device)
    fn = torch.zeros((), device=config.device)

    test_loss = 0.0
    total_samples = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in tqdm(test, desc="SL Evaluation", leave=False):
            batch = batch.to(config.device)

            # One binary label per sample
            targets_raw = batch.y.float().reshape(-1)
            logits, latency = router(batch)
            logits = logits.reshape(-1)

            assert logits.shape == targets_raw.shape, (
                f"Output shape {logits.shape} does not match "
                f"label shape {targets_raw.shape}"
            )

            loss = criterion(logits, targets_raw)

            predictions = logits >= 0
            targets = targets_raw >= 0.5

            batch_size = targets.numel()
            test_loss += loss.item() * batch_size
            total_samples += batch_size

            correct += (predictions == targets).sum()
            tp += (predictions & targets).sum()
            fp += (predictions & ~targets).sum()
            fn += (~predictions & targets).sum()

    test_loss /= total_samples
    test_accuracy = correct / total_samples
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    print(
        f"Epoch {epoch} | Train Loss: {train_loss:.4f} "
        f"| Test Loss: {test_loss:.4f} "
        f"| Accuracy: {test_accuracy.item():.4f} "
        f"| Precision: {precision.item():.4f} "
        f"| Recall: {recall.item():.4f}"
    )

    return test_loss


def sl_train_loop(config, router, train, test):
    num_experts = len(config.moe_experts)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(router.parameters(), lr=config.router_sl_learning_rate)
    best_loss = 1e10
    patience = config.router_sl_patience

    for epoch in range(config.router_sl_num_epochs):
        losses = []
        router.train()
        for batch in tqdm(train, desc=f"SL Training - Epoch {epoch}", leave=False):
            batch = batch.to(config.device)
            batch_labels = batch.y
            optimizer.zero_grad()

            logits, latency = router(batch)
            loss = criterion(logits, batch_labels)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        train_loss = sum(losses) / len(losses)
        test_loss = sl_evaluate(config, router, test, num_experts, epoch, train_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            patience = config.router_sl_patience
        else:
            patience -= 1
            if patience <= 0:
                print(f"Early stopping at epoch {epoch} due to no improvement in test loss.")
                break

    router.save(f"checkpoints/evaluate_cps/router_{config.code_name}.pt")


def get_decoders(config, env):
    expert_list, inference_list = [], []
    for decoder_name in config.moe_experts:
        decoder, inference = get_agent_and_inference(config, env, decoder_name)
        expert_list.append(decoder)
        inference_list.append(inference)

    return expert_list, inference_list


def evaluate_moe(config):
    shots = load_shots(config, dataset_type="mistakes", agent_name="bp")
    env = QECEnv(config, shots)
    agent = RouterAgent(config, env, router_checkpoint="checkpoints/evaluate_cps/router.pt")
    expert_list, inference_list = get_decoders(config, env)

    moe_rl_train_loop(config, env, agent, expert_list, inference_list, train=False)


def train_moe_rl(config):
    shots = load_shots(config, dataset_type="moe")
    env = QECEnv(config, shots)
    agent = RouterAgent(config, env, encoder_checkpoint="checkpoints/evaluate_cps/sl_nbp_big.pt")
    expert_list, inference_list = get_decoders(config, env)
    moe_rl_train_loop(config, env, agent, expert_list, inference_list)


def train_moe_sl(config):
    shots = load_shots(config, dataset_type="moe_labelled")
    dataset = GraphDataset(shots)

    generator = torch.Generator().manual_seed(42)
    n_test = int(len(dataset) * config.router_sl_test_fraction)
    n_train = len(dataset) - n_test
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=generator)

    train_loader = DataLoader(train_set, batch_size=config.router_sl_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.router_sl_batch_size, shuffle=False)
    env = QECEnv(config)

    agent = RouterNetwork(config, env)
    sl_train_loop(config, agent, train_loader, test_loader)


def train_router(config):
    train_moe_sl(config)
    # evaluate_moe(config)