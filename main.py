from environments import MultivariateBicycleCode, QLDPCCode
from utils import *
import sys
import json

def get_configs(model_config_file, code_config_file, agent_name, code_name):
    model_config = json.load(open(model_config_file))
    code_config = json.load(open(code_config_file))
    return model_config[agent_name], code_config[code_name]


def print_example_env(code_config):

    sample_config = code_config.copy()
    sample_config["error_rate"] = 0

    # env = QLDPCCode(**sample_config)
    env = MultivariateBicycleCode(**sample_config)

    # print(env.H_x)
    # print()
    # print(env.H_z)

    # env.plot_tanner()

    # Validate logical error
    # logical_operation = np.zeros(env.n_data, dtype=np.int8)
    # logical_operation[sample_config["logical_operators"]] = 1
    # env.step(logical_operation)
    #
    env.render()


if __name__ == '__main__':

    agent_name = sys.argv[1] if len(sys.argv) > 1 else "sac"
    code_name = sys.argv[2] if len(sys.argv) > 2 else "18_4_4"

    model_config, code_config = get_configs("model_configs.json", "code_configs.json", agent_name, code_name)

    print_example_env(code_config)

    # rewards = adversarial_training_loop(model_config=model_config, code_config=code_config)
    rewards = single_agent_training_loop(code_config=code_config, model_config=model_config, model_checkpoint="checkpoints/sac_defender_single.zip")
    baselines = run_baselines(model_config=model_config, code_config=code_config)

    plot_results(rewards, baselines, model_config, f"results/termination.png")
    render_evaluation_episode(code_config, "checkpoints/sac_defender_single.zip")
