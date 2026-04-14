from src.experiments import train_dqn, optimize_hyperparameters
from src.train_utils import run_baselines, benchmark_env
from src.read_config import ConfigParser
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agents for quantum error correction.")
    parser.add_argument("--agent", type=str, default="dqn", help="Agent type (e.g., dqn, ppo)")
    parser.add_argument("--code", type=str, default="18_2_3_toric", help="Code configuration (e.g., 18_2_3_toric)")
    parser.add_argument("--experiment", type=str, default="benchmark", help="Experiment type (e.g., train, eval, benchmark)")
    return parser.parse_args()


def select_experiment(experiment_name):
    match experiment_name:
        case "train": return train_dqn
        case "baselines": return run_baselines
        case "benchmark": return benchmark_env
        case "hpo": return optimize_hyperparameters
        case _: raise ValueError(f"Unknown experiment: {experiment_name}")


if __name__ == '__main__':

    args = parse_args()

    config = ConfigParser("configs", args.agent, args.code)
    experiment = select_experiment(args.experiment)

    experiment(config)
