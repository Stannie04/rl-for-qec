"""Module defining data qubits and stabilizers for quantum error correction codes."""

import numpy as np

class DataQubit:
    def __init__(self):
        self.error = 0 # 0: no error, 1: X error, 2: Z error, 3: Y error
        self.connected_stabilizers = []

    def __str__(self):
        color = "\033[0m" if self.error == 0 else "\033[33m"
        reset = "\033[0m"
        return f"{color}{str(self.error)}{reset}"

    def reset(self):
        self.error = 0

    def flip(self, operation: int):
        self.error ^= operation

        # Update stabilizers
        for stabilizer in self.connected_stabilizers:
            if stabilizer.check_type == operation or operation == 3:
                stabilizer.flip_parity()


    def add_stabilizer_to_connections(self, stabilizer):
        self.connected_stabilizers.append(stabilizer)


class Stabilizer:
    def __init__(self, check_type: int, i: int, j: int):
        self.state = 0
        self.check_type = check_type # 0: X stabilizer, 1: Z stabilizer

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

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)] + interaction_vectors:
            ni = (self.i + di) % full_grid.shape[0]
            nj = (self.j + dj) % full_grid.shape[1]

            # We currently do not do any checks for whether the neighbor is a data qubit; we assume the grid is a checkerboard.
            full_grid[ni][nj].add_stabilizer_to_connections(self)

        # We currently do not need the coordinates of the stabilizer for anything else.
        # Nevertheless, we store them for now and delete them here, just in case we need them later.
        del self.i, self.j

    def flip_parity(self):
        self.state = 1 - self.state

    def reset(self):
        self.state = 0