"""Module implementing a multivariate bicycle code environment for quantum error correction."""

import gymnasium as gym
import numpy as np
from typing import Optional
# from .qubits import DataQubit, Stabilizer
# import random

class MultivariateBicycleCode(gym.Env):

    def __init__(self, l: int, m: int, interaction_vectors=None,
                 logical_operators=None, error_rate=None, # No defaults given so that they must be explicitly set in the config files,
                 evaluation_mode=False, action_threshold=None,  # reducing bugs from forgetting to set them.
                 termination_threshold=None, max_episode_length=None):

        super().__init__()

        self.error_rate = error_rate

        self._init_qubits(l, m, interaction_vectors)

        self.observation_space = gym.spaces.Box(0, 1, shape=(self.n_data,), dtype=np.uint8)
        self.action_space = gym.spaces.Box(0, 1, shape=(self.n_data,), dtype=np.float32)

        self.episode_steps = 0
        self.current_player = 0

        self.num_errors = 0
        self.info = None
        self.previous_info = None
        self.logical_operators = logical_operators

        self.max_episode_length = max_episode_length
        self.evaluation_mode = evaluation_mode
        self.action_threshold = action_threshold
        self.termination_threshold = termination_threshold


    def step(self, action, single_player=True):

        self.previous_info = self._get_info()
        # Intermediate visualization of the action
        if self.evaluation_mode:
            print("\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n")
            self.render()

        error_locations = np.where(self.data_errors == 1)[0]

        mask = action > self.action_threshold

        self.data_errors[mask] = 1 - self.data_errors[mask]
        for i in range(self.n_data):
            if mask[i]:
                self.stabilizer_states[self.qubits_to_stabilizers[i]] ^= 1

        # Flip the assigned qubit.
        # self.data_errors[action] = 1 - self.data_errors[action]
        # self.stabilizer_states[self.qubits_to_stabilizers[action]] ^= 1

        # Check whether the episode is finished.
        self.episode_steps += 1
        self.info = self._get_info()
        terminated = self._get_terminated()
        truncated = self.episode_steps > self.max_episode_length


        previous_errors = self.previous_info["num_errors"]
        current_errors = self.info["num_errors"]

        delta_errors = previous_errors - current_errors

        # Calculate reward.
        if single_player or self.current_player == 0:  # Defender
            # The defender gets a positive reward for reducing the number of errors, and a negative reward for increasing it.
            # Additionally, they get a large negative reward if the episode ends due to a logical error.
            reward = 1 + delta_errors #- self.max_episode_length * terminated
        else:  # Adversary
            reward = -(delta_errors - self.max_episode_length * terminated)


        # Update state.
        if single_player:
            self._flip_randomly()
        else:
            self.current_player = 1 - self.current_player

        observation = self._get_obs()
        return observation, reward, terminated, truncated, self.info


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = {}):
        super().reset(seed=seed)

        self.data_errors.fill(0)
        self.stabilizer_states.fill(0)

        # Flip qubits based on an error rate
        self._flip_randomly()

        self.episode_steps = 0
        self.current_player = 0
        return self._get_obs(), self._get_info()


    def render(self, mode='human'):
        rows = []
        for i in range(2 * self.m):
            row_cells = []
            for j in range(2 * self.l):
                if (i + j) % 2 == 0:
                    # Data qubit
                    idx = self._grid_to_data_idx(i, j)
                    val = self.data_errors[idx]
                    color = "\033[0m" if val == 0 else "\033[33m"  # yellow if error
                    row_cells.append(f"{color}{val}\033[0m")
                else:
                    # Stabilizer
                    idx = self._grid_to_stabilizer_idx(i, j)
                    val = self.stabilizer_states[idx]
                    color = "\033[32m" if val == 0 else "\033[31m"  # green or red
                    row_cells.append(f"{color}{'X' if i % 2 == 0 else 'Z'}\033[0m")
            rows.append(" ".join(row_cells))

        print("\n".join(rows) + "\n")


    def _flip_randomly(self):

        # Flip a single bit
        # idx = np.random.randint(self.n_data)
        # self.data_errors[idx] = 1 - self.data_errors[idx]
        # self.stabilizer_states[self.qubits_to_stabilizers[idx]] ^= 1
        mask = np.random.rand(self.n_data) < self.error_rate
        self.data_errors[mask] = 1

        for i in range(self.n_data):
            if mask[i]:
                self.stabilizer_states[self.qubits_to_stabilizers[i]] ^= 1

    def _init_qubits(self, l, m, interaction_vectors=None):

        self.l, self.m = l, m
        self.n_data, self.n_stabilizers = 2*l*m, 2*l*m

        self.data_errors = np.zeros(self.n_data, dtype=np.int8)
        self.stabilizer_states = np.zeros(self.n_stabilizers, dtype=np.int8)

        # TODO: non-uniform error rates
        self.error_rates = np.full(self.n_data, self.error_rate, dtype=np.float32)

        # Get index mapping for data qubits and stabilizers.
        self.qubits_to_stabilizers = [list() for _ in range(self.n_data)]

        for i in range(2 * m):
            for j in range(2 * l):
                if (i + j) % 2 == 0: # Data qubits
                    data_idx = self._grid_to_data_idx(i, j)
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)] + (interaction_vectors or []):
                        di, dj = (di, dj) if i % 2 == 0 else (-di, -dj) # Flip interaction vectors for L sublattice

                        # We are currently only looking at bit-flip errors, so only consider the Z stabilizers (i+j odd) that act on this data qubit.
                        ni = (i + di) % (2*m)
                        nj = (j + dj) % (2*l)
                        if ni % 2 == 1:
                            self.qubits_to_stabilizers[data_idx].append(self._grid_to_stabilizer_idx(ni, nj))


    def _grid_to_data_idx(self, i, j):
        return i*self.l + j//2 if (i + j) % 2 == 0 else i*self.l + (j-1)//2

    def _grid_to_stabilizer_idx(self, i, j):
        return i*self.l + j//2 if (i + j) % 2 == 1 else i*self.l + (j-1)//2


    def _get_obs(self):
        return self.stabilizer_states


    def _get_info(self):
        return {
            "num_errors": np.sum(self.data_errors),
            "num_syndromes": np.sum(self.stabilizer_states),
            "episode_steps": self.episode_steps,
        }


    def _get_terminated(self):
        """Check whether a logical error has occurred."""

        return self.info["num_errors"] > self.termination_threshold



if __name__ == "__main__":
    env = MultivariateBicycleCode(l=5, m=3, interaction_vectors=[(3, 4)], error_rate=0.05)
    obs, info = env.reset()
    env.render()

    # random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    env.render()