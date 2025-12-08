import gymnasium as gym
from gymnasium import spaces
import jax
import jax.numpy as jnp
from dataclasses import dataclass
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np


@dataclass
class State:
    pass



class MultivariateBicycleCode(gym.Env):

    def __init__(self, l: int, m: int):
        super().__init__()

        I_m = jnp.eye(m)
        I_l = jnp.eye(l)
        S_m = jnp.roll(I_m, shift=1, axis=1)
        S_l = jnp.roll(I_l, shift=1, axis=1)

        x = jnp.kron(S_l, I_m)
        y = jnp.kron(I_l, S_m)
        z = jnp.kron(S_l, S_m)

        # Using x, y and z, create matrices A and B as powers of x, y, z
        # TODO: Find a better way to define the powers
        A = jnp.linalg.matrix_power(x, 1) + jnp.linalg.matrix_power(y, 2)
        B = jnp.linalg.matrix_power(x, 3) + jnp.linalg.matrix_power(y, 4)

        self.H_X = jnp.concatenate([A, B], axis=1)
        self.H_Z = jnp.concatenate([jnp.transpose(B), jnp.transpose(A)], axis=1)

        self.n, self.k, self.d = self.get_code_params(l, m, A, B)


    def get_code_params(self, l, m, A, B):
        """ Return the parameters [n, k, d] of the CSS code defined by self.H_X and self.H_Z."""

        # n = 2lm
        n = 2 * l * m

        # k = 2·dim(ker(A) ∩ ker(B)). Note that ker(A) = n - rank(A).
        k = 2 * (n - jnp.linalg.matrix_rank(A) - jnp.linalg.matrix_rank(B))

        # Distance calculation is non-trivial; placeholder value
        d = -1
        return n, k, d


    def render(self, spacing=1.5):
        """ Plot the Tanner graph for the CSS code defined by self.H_X and self.H_Z."""

        Hx = self.H_X
        Hz = self.H_Z

        n_qubits = Hx.shape[1]
        n_x = Hx.shape[0]
        n_z = Hz.shape[0]

        G = nx.Graph()

        # Split qubits left/right
        mid = n_qubits // 2
        left_qubits = [f"q{i}" for i in range(mid)]
        right_qubits = [f"q{i}" for i in range(mid, n_qubits)]
        qubits = left_qubits + right_qubits

        xchecks = [f"Xc{i}" for i in range(n_x)]
        zchecks = [f"Zc{i}" for i in range(n_z)]

        G.add_nodes_from(qubits)
        G.add_nodes_from(xchecks)
        G.add_nodes_from(zchecks)

        # Add edges from Hx
        for r in range(n_x):
            for q in range(n_qubits):
                if Hx[r, q] == 1:
                    G.add_edge(f"q{q}", f"Xc{r}")

        # Add edges from Hz
        for r in range(n_z):
            for q in range(n_qubits):
                if Hz[r, q] == 1:
                    G.add_edge(f"q{q}", f"Zc{r}")

        pos = {}

        for i, xc in enumerate(xchecks):
            col = i * 2 + 1  # X-checks in odd columns
            pos[xc] = (col * spacing, 0)

        for i, rq in enumerate(right_qubits):
            col = i * 2  # R-data qubits in even columns
            pos[rq] = (col * spacing, 0)

        # Other way round for Z-checks and L-data qubits, so checks are not below each other
        for i, zc in enumerate(zchecks):
            col = i * 2
            pos[zc] = (col * spacing, -spacing)

        for i, lq in enumerate(left_qubits):
            col = i * 2 + 1
            pos[lq] = (col * spacing, -spacing)

        plt.figure(figsize=(12, 6))
        nx.draw_networkx_edges(G, pos, alpha=0.6, width=1.3)


        # Left qubits = blue, Right qubits = orange
        nx.draw_networkx_nodes(G, pos,
                               nodelist=left_qubits,
                               node_color="#00AEEF",
                               node_size=400,
                               node_shape="o")
        nx.draw_networkx_nodes(G, pos,
                               nodelist=right_qubits,
                               node_color="#FFA500",
                               node_size=400,
                               node_shape="o")

        # X checks = pink squares
        nx.draw_networkx_nodes(G, pos,
                               nodelist=xchecks,
                               node_color="#E889F1",
                               node_size=600,
                               node_shape="s")

        # Z checks = cyan squares
        nx.draw_networkx_nodes(G, pos,
                               nodelist=zchecks,
                               node_color="#39D2AF",
                               node_size=600,
                               node_shape="s")

        plt.axis("off")
        plt.show()


    def __str__(self):
        return f"[[n, k, d]] = [[{self.n}, {self.k}, {self.d}]]\n\nH_X and H_Z:\n{str(self.H_X)}\n\n{str(self.H_Z)}"


if __name__ == "__main__":
    env = MultivariateBicycleCode(l=6, m=1)
    print(env)
    env.render()