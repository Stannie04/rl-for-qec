from src.experiments import train, optimize_hyperparameters, train_moe
from src.train_utils import run_baselines, benchmark_env, render_example_environment, post_train_evaluation, full_code_analysis
from src.read_config import ConfigParser
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agents for quantum error correction.")
    parser.add_argument("-a", "--agent", type=str, default="sac", help="Agent type (e.g., dqn, sac)")
    parser.add_argument("-c", "--code", type=str, default="18_4_4_ldpc", help="Code configuration (e.g., 18_2_3_toric)")
    parser.add_argument("-e", "--experiment", type=str, default="train", help="Experiment type (e.g., train, eval, benchmark)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def select_experiment(experiment_name):
    match experiment_name:
        case "train": return train
        case "moe": return train_moe
        case "baselines": return run_baselines
        case "benchmark": return benchmark_env
        case "hpo": return optimize_hyperparameters
        case "render": return render_example_environment
        case "evaluate": return post_train_evaluation
        case "analysis": return full_code_analysis
        case _: raise ValueError(f"Unknown experiment: {experiment_name}")


if __name__ == '__main__':

    args = parse_args()

    config = ConfigParser("configs", args.agent, args.code, verbose=args.verbose)
    experiment = select_experiment(args.experiment)
    experiment(config)
