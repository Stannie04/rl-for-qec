"""A simple agent that selects actions randomly from a given set of possible actions.

This is currently a placeholder, until the environment has been properly defined.
"""

import random

class RandomAgent:
    def __init__(self, actions):
        self.actions = actions

    def select_action(self, state):
        return random.choice(self.actions)