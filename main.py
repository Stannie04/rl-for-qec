from environments import MultivariateBicycleCode
from utils import plot_results, adversarial_training_loop
import sys
import json


def get_configs(model_config_file, code_config_file, agent_name, code_name):
    model_config = json.load(open(model_config_file))
    code_config = json.load(open(code_config_file))
    return model_config[agent_name], code_config[code_name]


def print_example_env(code_config):
    env = MultivariateBicycleCode(**code_config)
    env.render()


if __name__ == '__main__':

    agent_name = sys.argv[1] if len(sys.argv) > 1 else "dqn"
    code_name = sys.argv[2] if len(sys.argv) > 2 else "multivariate_bicycle_code"

    model_config, code_config = get_configs("model_configs.json", "code_configs.json", agent_name, code_name)

    print_example_env(code_config)

    pretraining_comparison = {}

    for pretrain_timesteps in [0, 10_000, 50_000, 100_000]:
        rewards = adversarial_training_loop(model_config=model_config, code_config=code_config, pretrain_timesteps=pretrain_timesteps)
        pretraining_comparison[pretrain_timesteps] = rewards


    plot_results(pretraining_comparison, "results/pretrain.png")
