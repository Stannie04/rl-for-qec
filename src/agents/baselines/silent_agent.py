"""A simple agent that always does nothing."""

import numpy as np
import gymnasium as gym
import torch

from src.read_config import ConfigParser

class SilentAgent:

    def __init__(self, env: gym.Env, config: ConfigParser):
        self.env = env
        self.action = env.code.no_op_index
        self.device = env.device

    def select_action(self, observation, evaluate=False):
        return torch.tensor([self.action], device=self.device), None