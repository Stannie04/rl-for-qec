import gymnasium as gym
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class QLDPCCode(gym.Env):

    def __init__(self, l: int, m: int, **kwargs):

        super().__init__()

        self.l = l
        self.m = m

        self.H_x, self.H_z = self._init_parity_check_matrices(kwargs.get("params"))


    def _init_parity_check_matrices(self, params):

        def __polynomial_to_matrix(terms, x, y):
            matrix = np.zeros_like(x @ y)

            for x_exp, y_exp in terms:
                term_matrix = np.linalg.matrix_power(x, x_exp) @ np.linalg.matrix_power(y, y_exp)
                matrix += term_matrix

            return matrix

        I_l = np.eye(self.l, dtype=np.int8)
        I_m = np.eye(self.m, dtype=np.int8)

        S_l = np.roll(I_l, 1, axis=1)
        S_m = np.roll(I_m, 1, axis=1)

        x = np.kron(S_l, I_m)
        y = np.kron(I_l, S_m)

        A = __polynomial_to_matrix(params["A"], x, y)
        B = __polynomial_to_matrix(params["B"], x, y)

        return np.hstack([A, B]), np.hstack([B.T, A.T])


    def plot_tanner(self):

        G = nx.Graph()

        n_x, n_qubits = self.H_x.shape
        n_z, _ = self.H_z.shape

        for q in range(n_qubits):
            G.add_node(f"q{q}", node_type="qubit")

        for i in range(n_x):
            G.add_node(f"x{i}", node_type="xcheck")

        for i in range(n_z):
            G.add_node(f"z{i}", node_type="zcheck")

        for i in range(n_x):
            for j in range(n_qubits):
                if self.H_x[i, j] == 1:
                    G.add_edge(f"x{i}", f"q{j}")

        for i in range(n_z):
            for j in range(n_qubits):
                if self.H_z[i, j] == 1:
                    G.add_edge(f"z{i}", f"q{j}")

        # --- Layout ---
        pos = nx.spring_layout(G, seed=42)

        # Separate node lists
        qubits = [n for n in G.nodes if G.nodes[n]["node_type"] == "qubit"]
        xchecks = [n for n in G.nodes if G.nodes[n]["node_type"] == "xcheck"]
        zchecks = [n for n in G.nodes if G.nodes[n]["node_type"] == "zcheck"]

        plt.figure(figsize=(10, 8))

        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3)

        # Draw nodes by type (different shapes & colors)
        nx.draw_networkx_nodes(G, pos,
                               nodelist=qubits,
                               node_color="black",
                               node_shape="o",
                               node_size=200,
                               label="Data qubits")

        nx.draw_networkx_nodes(G, pos,
                               nodelist=xchecks,
                               node_color="red",
                               node_shape="s",
                               node_size=300,
                               label="X checks")

        nx.draw_networkx_nodes(G, pos,
                               nodelist=zchecks,
                               node_color="blue",
                               node_shape="^",
                               node_size=300,
                               label="Z checks")

        plt.legend(scatterpoints=1)
        plt.axis("off")
        plt.title("CSS Tanner Graph (Bicycle Code)")
        plt.show()


if __name__ == "__main__":
    code = QLDPCCode(l=3, m=3)

    print(code.H_x)
    print()
    print(code.H_z)