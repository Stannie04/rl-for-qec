import numpy as np
import gymnasium as gym

class RandomAgent:

    def __init__(self, **kwargs):
        pass

    def select_action(self, observation):
        return np.random.choice(len(observation))