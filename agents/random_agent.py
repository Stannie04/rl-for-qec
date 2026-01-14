"""A simple agent that selects actions randomly from a given set of possible actions.

Acts as a parent class for future agents.
"""

import gymnasium as gym

class Agent:
    def __init__(self, env: gym.Env):
        self.env = env

    def select_action(self, observation):
        return self.env.action_space.sample()