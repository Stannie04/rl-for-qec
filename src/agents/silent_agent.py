"""A simple agent that selects actions randomly from a given set of possible actions.

Acts as a parent class for future agents.
"""

import numpy as np
import gymnasium as gym

class SilentAgent:

    def __init__(self, **kwargs):
        pass

    def select_action(self, observation):
        return np.zeros_like(observation, dtype=np.float32)