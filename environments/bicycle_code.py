"""Module implementing a multivariate bicycle code environment for quantum error correction."""

import gymnasium as gym
import numpy as np

from .qubits import DataQubit, Stabilizer

class MultivariateBicycleCode(gym.Env):

    def __init__(self, l: int, m: int, interaction_vectors=None):
        super().__init__()

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

    def __str__(self):
        rows = []
        for row in self.full_grid:
            rows.append(" ".join(str(cell) for cell in row))
        return "\n".join(rows)


    def init_errors(self, num_errors: int, seed: int = 42):
        """ Randomly flip `num_errors` data qubits in the grid."""

        # Example to see cycle pattern
        # for (i, j) in [(8,8), (5, 17)]:
        #     self.full_grid[i][j].flip(operation=3)

        # Logical qubit X error, as per "Tour de gross: A modular quantum computer based on bivariate bicycle codes, Figure 3b"
        # for (i, j) in [(0,6), (1,13), (3,11), (6,6), (6,8), (7,9), (9,13), (8,6), (10,6), (10,8), (11,9), (11,11)]:
        #     self.full_grid[i][j].flip(operation=1)

        for qubit in np.random.choice(self.data_qubits, size=num_errors, replace=False):
            qubit.flip(operation=np.random.choice([1, 2, 3]))


    def update_stabilizers(self):
        """ Update the state of all stabilizers based on the current errors on data qubits."""
        for stabilizer in self.stabilizers:
            stabilizer.parity_check(stabilizer.connected_qubits)


    def num_errors(self):
        return sum(1 for qubit in self.data_qubits if qubit.error != 0)


    def num_syndromes(self):
        return sum(1 for stabilizer in self.stabilizers if stabilizer.state != 0)

if __name__ == "__main__":
    # No interaction vectors: toric code
    # env = MultivariateBicycleCode(l=6, m=12)

    # As per "Tour de gross: A modular quantum computer based on bivariate bicycle codes, Figure 3b"
    env = MultivariateBicycleCode(l=12, m=6, interaction_vectors=[(3, 6), (6, -3)])

    env.init_errors(num_errors=1)
    env.update_stabilizers()
    print(env)
    print(f"Number of errors: {env.num_errors()}")
    print(f"Number of syndromes: {env.num_syndromes()}")