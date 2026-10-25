import time
from tqdm import tqdm
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

    num_test_samples = len(test.dataset)
    correct = torch.zeros(num_experts, device=config.device)
    tp = torch.zeros(num_experts, device=config.device)
    fp = torch.zeros(num_experts, device=config.device)
    fn = torch.zeros(num_experts, device=config.device)
    test_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test, desc="SL Evaluation", leave=False):
            batch = batch.to(config.device)
            batch_labels = batch.y

            logits = router(batch)
            loss = torch.nn.BCEWithLogitsLoss()(logits, batch_labels)

            predictions = logits >= 0
            targets = batch_labels >= 0.5

            test_loss += loss.item() * targets.size(0)
            correct += (predictions == targets).sum(dim=0)
            tp += ((predictions == 1) & (targets == 1)).sum(dim=0)
            fp += ((predictions == 1) & (targets == 0)).sum(dim=0)
            fn += ((predictions == 0) & (targets == 1)).sum(dim=0)

    test_loss /= num_test_samples
    test_accuracy = correct / num_test_samples
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Accuracy: {test_accuracy.mean().item():.4f} | Precision: {precision.mean().item():.4f} | Recall: {recall.mean().item():.4f}")
    print(f"Per Expert Accuracy: {test_accuracy.cpu().numpy()}, Precision: {precision.cpu().numpy()}, Recall: {recall.cpu().numpy()}\n")


def sl_train_loop(config, router, train, test):
    num_experts = len(config.moe_experts)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(router.parameters(), lr=config.router_sl_learning_rate)

    for epoch in range(config.router_sl_num_epochs):
        losses = []
        router.train()
        for batch in tqdm(train, desc=f"SL Training - Epoch {epoch}", leave=False):
            batch = batch.to(config.device)
            batch_labels = batch.y
            optimizer.zero_grad()

            logits = router(batch)
            loss = criterion(logits, batch_labels)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        train_loss = sum(losses) / len(losses)
        sl_evaluate(config, router, test, num_experts, epoch, train_loss)

    router.save(f"checkpoints/evaluate_cps/moe_{config.code_name}_router.pt")


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