from utils import *
import sys
import json
from tqdm import tqdm

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_configs(model_config_file, code_config_file, agent_name, code_name):
    model_config = json.load(open(model_config_file))
    code_config = json.load(open(code_config_file))
    return model_config[agent_name], code_config[code_name]

if __name__ == '__main__':

    agent_name = sys.argv[1] if len(sys.argv) > 1 else "dqn"
    code_name = sys.argv[2] if len(sys.argv) > 2 else "18_4_4"

    model_config, code_config = get_configs("model_configs.json", "code_configs.json", agent_name, code_name)

    # rewards = train_dqn(code_config, model_config, device=device)
    benchmark_env(code_config, device=device)

    # rewards = adversarial_training_loop(model_config=model_config, code_config=code_config)
    # rewards = single_agent_training_loop(code_config=code_config, model_config=model_config, model_checkpoint="checkpoints/dqn_defender_single.zip")
    # baselines = run_baselines(model_config=model_config, code_config=code_config)
    #
    # plot_results(rewards, baselines, model_config, f"results/termination.png")
    # render_evaluation_episode(code_config, "checkpoints/dqn_defender_single.zip")
