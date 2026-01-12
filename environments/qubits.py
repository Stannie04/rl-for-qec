"""Module defining data qubits and stabilizers for quantum error correction codes."""

import numpy as np

class DataQubit:
    def __init__(self):
        self.error = 0 # 0: no error, 1: X error, 2: Z error, 3: Y error

    def __str__(self):
        color = "\033[0m" if self.error == 0 else "\033[33m"
        reset = "\033[0m"
        return f"{color}{str(self.error)}{reset}"

    def reset(self):
        self.error = 0

    def flip(self, operation: int):
        self.error ^= operation


class Stabilizer:
    def __init__(self, check_type: int, i: int, j: int):
        self.state = 0
        self.check_type = check_type # 0: X stabilizer, 1: Z stabilizer
        self.connected_qubits = None

        self.i = i
        self.j = j


    def __str__(self):
        color = "\033[32m" if self.state == 0 else "\033[31m"
        reset = "\033[0m"
        letter = 'X' if self.check_type == 0 else 'Z'
        return f"{color}{letter}{reset}"


    def initialize_connected_qubits(self, full_grid, interaction_vectors):
        """ Get the data qubits connected to this stabilizer at position (i, j) in the grid.

        Assume the code is on a torus, so wrap around edges.
        """

        if self.check_type == 1:
            interaction_vectors = [(-di, -dj) for (di, dj) in interaction_vectors]

        qubits = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)] + interaction_vectors:
            ni = (self.i + di) % full_grid.shape[0]
            nj = (self.j + dj) % full_grid.shape[1]

            # We do not do any checks for whether the neighbor is a data qubit; we assume the grid is a checkerboard.
            qubits.append(full_grid[ni][nj])

        self.connected_qubits = np.array(qubits)

        del self.i, self.j


    def parity_check(self, data_qubits):
        """ Update the stabilizer state based on the errors on connected data qubits.

        Note that X stabilizers check for Z errors (error == 2 or 3),
        and Z stabilizers check for X errors (error == 1 or 3).
        """

        parity = 0
        for qubit in data_qubits:
            if self.check_type == 0:  # X stabilizer checks for Z errors
                if qubit.error in [2, 3]:
                    parity ^= 1
            else:  # Z stabilizer checks for X errors
                if qubit.error in [1, 3]:
                    parity ^= 1
        self.state = parity