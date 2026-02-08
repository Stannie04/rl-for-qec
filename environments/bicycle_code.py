"""Module implementing a multivariate bicycle code environment for quantum error correction."""

import gymnasium as gym
import numpy as np
from typing import Optional
from .qubits import DataQubit, Stabilizer
import random

class MultivariateBicycleCode(gym.Env):

    def __init__(self, l: int, m: int, interaction_vectors=None, error_rate=0.01):
        super().__init__()

        self.error_rate = error_rate

        self.init_qubits(l, m, interaction_vectors)

        num_actions = 2*self.l*self.m

        self.observation_space = gym.spaces.Box(0, 1, shape=(2*self.m*self.l,), dtype=int)
        self.action_space = gym.spaces.Discrete(num_actions)

        self.episode_steps = 0
        self.max_episode_length = 100
        self.current_player = 0

        self.info = None
        self.previous_info = None


    def step(self, action):

        self.previous_info = self._get_info()

        # Flip the assigned qubit.
        qubit_acted_on = self.data_qubits[action]
        qubit_acted_on.flip(operation=1, force=True)

        # Check whether the episode is finished.
        self.episode_steps += 1
        self.info = self._get_info()
        terminated = self._get_terminated()
        truncated = self.episode_steps > self.max_episode_length

        # Calculate reward.
        num_errors = self.info["num_errors"]

        if self.current_player == 0:  # Defender
            reward = -num_errors
        else:  # Adversary
            reward = num_errors
        self.current_player = 1 - self.current_player

        # for qubit in self.data_qubits:
        #     qubit.flip(operation=1)

        observation = self._get_obs()
        return observation, reward, terminated, truncated, self.info


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = {}):
        super().reset(seed=seed)

        self.init_errors(options)

        self.episode_steps = 0
        return self._get_obs(), self._get_info()


    def render(self, mode='human'):
        rows = [" ".join(str(cell) for cell in row) for row in self.full_grid]
        print("\n".join(rows) + "\n")


    def init_qubits(self, l, m, interaction_vectors=None):
        if interaction_vectors is None:
            interaction_vectors = []

        self.l, self.m = l, m
        self.full_grid = np.empty((2*self.m, 2*self.l), dtype=object)

        data_qubits = []
        stabilizers = []

        for i in range(2 * m):
            for j in range(2 * l):
                if (i + j) % 2 == 0:
                    qubit = DataQubit(self.error_rate)
                    self.full_grid[i][j] = qubit
                    data_qubits.append(qubit)
                else:
                    stabilizer = Stabilizer(check_type=i%2, i=i, j=j)
                    self.full_grid[i][j] = stabilizer
                    stabilizers.append(stabilizer)

        self.data_qubits = np.array(data_qubits, dtype=object)
        self.stabilizers = np.array(stabilizers, dtype=object)

        # Initialize connected qubits for each stabilizer
        for stabilizer in self.stabilizers:
            stabilizer.initialize_connected_qubits(self.full_grid, interaction_vectors=interaction_vectors)


    def init_errors(self, options: Optional[dict] = {}):
        """ Set all qubits to 0. Then, randomly flip `num_errors` data qubits in the grid."""

        for qubit in self.data_qubits:
            qubit.reset()

        for stabilizer in self.stabilizers:
            stabilizer.reset()

        # Flip all qubits based on an error rate
        for qubit in self.data_qubits:
            qubit.flip(operation=1)


    def _get_obs(self):
        return np.array([stabilizer.state for stabilizer in self.stabilizers], dtype=np.float32)


    def _get_info(self):
        return {
            "num_errors": sum(1 for qubit in self.data_qubits if qubit.error != 0),
            "num_syndromes": sum(1 for stabilizer in self.stabilizers if stabilizer.state != 0),
        }


    def _get_terminated(self):
        """Check whether a logical error has occurred."""

        return self.info["num_errors"] > 5


if __name__ == "__main__":
    # No interaction vectors: toric code
    # env = MultivariateBicycleCode(l=6, m=12)

    # As per "Tour de gross: A modular quantum computer based on bivariate bicycle codes, Figure 3b"
    env = MultivariateBicycleCode(l=12, m=6, interaction_vectors=[(3, 6), (6, -3)])

    env.init_errors({"num_errors": 3})
    env.render()