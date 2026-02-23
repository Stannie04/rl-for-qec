from dataclasses import dataclass
from typing import Tuple
import jax
import jax.numpy as jnp

@dataclass
class EnvState:
    pass

@dataclass
class LDPCParams:
    l: int
    m: int
    max_episode_length: int = 100
    error_threshold: int = 5

def step_env():
    pass