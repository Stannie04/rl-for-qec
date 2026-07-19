from src.experiments import train_rl, optimize_hyperparameters, train_router, train_sl, train_all
from src.train_utils import benchmark_env, post_train_evaluation, full_analysis, create_all_datasets, render_mistakes
from src.read_config import ConfigParser
import argparse

import torch
if not torch.cuda.is_available():
    print("\n", "="*80)
    print("WARNING: CUDA is not available. Using CPU for training.")
    print("="*80, "\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agents for quantum error correction.")
    parser.add_argument("-a", "--agent", type=str, default="sac", help="Agent type (e.g., dqn, sac)")
    parser.add_argument("-c", "--code", type=str, default="144_12_12_ldpc", help="Code configuration (e.g., 18_2_3_toric)")
    parser.add_argument("-e", "--experiment", type=str, default="rl", help="Experiment type (e.g., train, eval, benchmark)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-r", "--run_name", type=str, default=None, help="Name of run")
    return parser.parse_args()


def select_experiment(experiment_name):
    match experiment_name:
        case "rl": return train_rl
        case "sl": return train_sl
        case "router": return train_router
        case "all": return train_all
        case "benchmark": return benchmark_env
        case "hpo": return optimize_hyperparameters
        case "render": return render_mistakes
        case "evaluate": return post_train_evaluation
        case "analysis": return full_analysis
        case "dataset": return create_all_datasets
        case _: raise ValueError(f"Unknown experiment: {experiment_name}")


if __name__ == '__main__':

    args = parse_args()

    config = ConfigParser("configs", args.agent, args.code, run_name=args.run_name, verbose=args.verbose)
    experiment = select_experiment(args.experiment)
    experiment(config)
