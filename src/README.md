# The `/src` Directory

The `/src` directory contains all the code used for creating, training, and evaluating the RL-driven QEC framework. The main components of this directory are as follows:

- `agents/`: Contains the implementation of the agents used for experiments. Additionally, the network architectures are defined.
- `environment/`: Defines the environment for the RL agent, and provides code for generating a QLDPC code based on the configuration provided in `../configs/`.
- `experiments/`: Contains the code for training the RL and supervised agents, as well as the router. Additionally, HPO code is provided.
- `train_utils/`: Miscellaneous utility functions for training and evaluation, including code for generating datasets of shots.
- `read_config.py`: A class for reading the configuration files in `../configs/`.