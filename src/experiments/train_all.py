from src.experiments.train_rl import train_rl
from src.experiments.train_sl import train_sl
from src.experiments.train_router import train_router


def train_all(config):
    """
    Train all agents (RL, SL, and MoE) sequentially.
    """
    print("Starting training for all agents...")

    for i in range(config.n_repetitions):

        # Train SL agent
        print(f"Training SL agent, repetition {i + 1}/{config.n_repetitions}...")
        config.hidden_layers_gnn = [32, 128, 256, 512, 512, 256, 256]
        config.hidden_layers_mlp = [256, 128, 64]
        config.neural_bp_iterations = 8
        config.wandb_run_name = f"sl_rep_{i}"
        config.use_neural_bp = False
        train_sl(config)

        print(f"Training RL agent, repetition {i + 1}/{config.n_repetitions}...")
        config.hidden_layers_gnn = [32,128,256,512,512,256,256]
        config.hidden_layers_mlp = [256, 128, 64]
        config.neural_bp_iterations =  8
        config.wandb_run_name = f"rl_rep_{i}"
        config.use_neural_bp = False
        train_rl(config)

    print("All agents trained successfully.")