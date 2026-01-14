"""Module implementing a multivariate bicycle code environment for quantum error correction."""

import gymnasium as gym
import numpy as np
from typing import Optional
from .qubits import DataQubit, Stabilizer

class MultivariateBicycleCode(gym.Env):

    def __init__(self, l: int, m: int, num_errors: int, interaction_vectors=None):
        super().__init__()

        self.init_qubits(l, m, interaction_vectors)

        num_actions = 2*self.l*self.m

        self.observation_space = gym.spaces.Box(0, 1, shape=(2*self.m*self.l,), dtype=int)
        self.action_space = gym.spaces.Discrete(num_actions)

        self.episode_steps = 0
        self.max_episode_length = 100

        self.num_errors = num_errors

        self.previous_info = None

    def step(self, action):
        qubit_acted_on = self.data_qubits[action]
        qubit_acted_on.flip(operation=1)

        self.episode_steps += 1

        info = self._get_info()

        terminated = info["num_errors"] == 0
        truncated = self.episode_steps > self.max_episode_length

        reward = 1 if terminated else -0.01

        if self.previous_info is not None:
            reward += (self.previous_info["num_syndromes"] - info["num_syndromes"]) * 0.1

        observation = self._get_obs()

        self.previous_info = info

        return observation, reward, terminated, truncated, info


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
                    qubit = DataQubit()
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

        num_errors = options.get("num_errors", self.num_errors)

        for qubit in self.data_qubits:
            qubit.reset()

        for stabilizer in self.stabilizers:
            stabilizer.reset()

        # Example to see cycle pattern
        # for (i, j) in [(8,8), (5, 17)]:
        #     self.full_grid[i][j].flip(operation=3)

        # Logical qubit X error, as per "Tour de gross: A modular quantum computer based on bivariate bicycle codes, Figure 3b"
        # for (i, j) in [(0, 6), (1, 13), (3, 11), (6, 6), (6, 8), (7, 9), (9, 13), (8, 6), (10, 6), (10, 8), (11, 9),
        #                (11, 11)]:
        #     self.full_grid[i][j].flip(operation=1)

        for qubit in np.random.choice(self.data_qubits, size=num_errors, replace=False):
            qubit.flip(operation=1)
            # qubit.flip(operation=np.random.choice([1, 2, 3]))


    def _get_obs(self):
        return [stabilizer.state for stabilizer in self.stabilizers]


    def _get_info(self):
        return {
            "num_errors": sum(1 for qubit in self.data_qubits if qubit.error != 0),
            "num_syndromes": sum(1 for stabilizer in self.stabilizers if stabilizer.state != 0)
        }


if __name__ == "__main__":
    # No interaction vectors: toric code
    # env = MultivariateBicycleCode(l=6, m=12)

    # As per "Tour de gross: A modular quantum computer based on bivariate bicycle codes, Figure 3b"
    env = MultivariateBicycleCode(l=12, m=6, interaction_vectors=[(3, 6), (6, -3)])

    env.init_errors({"num_errors": 3})
    env.render()