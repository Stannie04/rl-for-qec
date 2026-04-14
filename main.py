from src.experiments import *
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
        case _: raise ValueError(f"Unknown experiment: {experiment_name}")



if __name__ == '__main__':

    args = parse_args()

    config = ConfigParser("configs", args.agent, args.code)
    experiment = select_experiment(args.experiment)

    experiment(config)

    # optimize_hyperparameters(code_config)

    # ldpc = np.load("results/new_results/dqn_lengths_ldpc.npy")
    # toric = np.load("results/new_results/dqn_lengths_toric.npy")
    # baseline = np.load("results/new_results/silent_agent_rewards.npy")
    # random = np.load("results/new_results/random_agent_rewards.npy")

    # results = {}
    # baseline = {"Silent Agent": baseline.flatten(), "Random Agent": random.flatten()}


    # for lengths, name in [(ldpc, "LDPC"), (toric, "Toric")]:
    #     results[name] = [lengths.flatten()]


    #     print(f"{name} - Mean: {lengths.mean():.2f}, Std: {lengths.std():.2f}, Max: {lengths.max()}, Min: {lengths.min()}")
    #     plt.plot(lengths, label=name)

    # plot_results(results, baseline, model_config, 0, f"results/simplified_train_env2.png")

        # plot_results({"Reward": lengths}, {}, model_config, f"results/{name}_lengths.png")

    # rewards = train_dqn(code_config, model_config, device=device)
    # benchmark_env(code_config, model_config, device=device)

    # rewards = adversarial_training_loop(model_config=model_config, code_config=code_config)
    #rewards = train_dqn(code_config=code_config, model_config=model_config, device=device)
    # baselines = run_baselines(model_config=model_config, code_config=code_config)

    #rewards.pop("Reward")
    #plot_results(rewards, baselines, model_config, f"results/termination.png")
    # render_evaluation_episode(code_config, "checkpoints/dqn_defender_single.zip")/
