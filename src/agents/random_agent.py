import numpy as np
import gymnasium as gym

from src.read_config import ConfigParser

class RandomAgent:

    def __init__(self, config: ConfigParser):
        pass

    def select_action(self, observation):
        return np.random.choice(len(observation))