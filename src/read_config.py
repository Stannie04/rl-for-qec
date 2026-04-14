import yaml
import torch
import prettytable

class ConfigParser:
    def __init__(self, config_dir, agent_name, code_name):

        with open(f"{config_dir}/train_config.yml", "r") as f:
            train_config = yaml.safe_load(f)

        with open(f"{config_dir}/code_config.yml", "r") as f:
            all_code_configs = yaml.safe_load(f)
            code_config = all_code_configs[code_name]

        with open(f"{config_dir}/model_config.yml", "r") as f:
            all_model_configs = yaml.safe_load(f)
            model_config = all_model_configs[agent_name]

        # Get all the parameters in the config files, and make them attributes of this class
        for c in [train_config, code_config, model_config]:
            for key, value in c.items():
                setattr(self, key, value)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agent_name = agent_name
        self.code_name = code_name

        self._print_configuration()


    def _print_configuration(self):
        table = prettytable.PrettyTable()
        table.field_names = ["Parameter", "Value"]

        for key, value in self.__dict__.items():
            table.add_row([key, value])

        print(table)