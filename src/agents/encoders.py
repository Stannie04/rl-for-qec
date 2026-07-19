import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv, GATConv


def scatter_add(x, index, dim_size):
    # Alternative to torch_scatter.scatter_add for SM_120 / torch 12.8 / Blackwell compatability
    out_shape = (dim_size,) + x.shape[1:]
    out = torch.zeros(out_shape, device=x.device, dtype=x.dtype)
    out.index_add_(0, index, x)
    return out


class CGNNEncoder(nn.Module):
    def __init__(self, config, env):
        super().__init__()

        hidden_dims = config.hidden_layers_gnn
        feature_dim = env.code.feature_dim

        assert len(hidden_dims) > 0, "hidden_dims must contain at least one layer size."

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = feature_dim
        for out_dim in hidden_dims:
            self.convs.append(GraphConv(in_dim, out_dim))
            self.norms.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.name = "tanner"

    def forward(self, data):
        x = data.x

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            prev_x = x
            x = conv(x, data.edge_index)
            x = norm(x)
            x = F.relu(x)

            # Residual connection only if dimensions match
            if prev_x.shape[-1] == x.shape[-1]:
                x = x + prev_x

        return x


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims):
        super().__init__()

        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class NeuralBPEncoder(nn.Module):
    def __init__(self, config, env):
        """NOTE: we require hidden_dim to be the same for all layers, due to the GRUCell updates.
        Otherwise, we need to enforce hidden_dims[0] == hidden_dims[-1] for the GRUCell updates to work correctly.
        """

        super().__init__()

        hidden_dim = config.neural_bp_hidden_dim
        mlp_dims = config.hidden_layers_gnn
        feature_dim = env.code.feature_dim
        num_iters = config.neural_bp_iterations

        mlp_hidden_dims = mlp_dims+[hidden_dim]

        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        self.q_to_c_mlp = MLP(3*hidden_dim, mlp_hidden_dims)
        self.c_to_q_mlp = MLP(3*hidden_dim+2, mlp_hidden_dims)
        self.qubit_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.check_gru = nn.GRUCell(hidden_dim, hidden_dim)

        self.qubit_norm = nn.LayerNorm(hidden_dim)
        self.check_norm = nn.LayerNorm(hidden_dim)

        self.num_iters = num_iters

        self.name = "nbp"

    def forward(self, data):
        x = F.relu(self.input_proj(data.x))
        syndrome = data.x[:, 3:5] # Raw syndrome info for action masking


        qubit_mask = data.x[:,0] > 0.5
        check_mask = ~qubit_mask

        q_idx = qubit_mask.nonzero(as_tuple=False).squeeze(-1)
        c_idx = check_mask.nonzero(as_tuple=False).squeeze(-1)

        src, dst = data.edge_index  # Edges go between qubits and checks
        q_src = src[qubit_mask[src]]
        q_dst = dst[qubit_mask[src]]
        c_src = src[check_mask[src]]
        c_dst = dst[check_mask[src]]

        for _ in range(self.num_iters):
            x_prev = x
            q_in = torch.cat([x_prev[q_src], x_prev[q_dst], x_prev[q_src]-x_prev[q_dst]], dim=-1)
            q_msg = self.q_to_c_mlp(q_in)
            check_aggregate = scatter_add(q_msg, q_dst, x_prev.size(0))

            c_old = x_prev[c_idx]
            c_new = self.check_gru(check_aggregate[c_idx], c_old)
            c_new = self.check_norm(c_old + c_new)

            x = x_prev.clone()
            x[c_idx] = c_new

            c_in = torch.cat([x[c_src], x[c_dst], x[c_src]-x[c_dst], syndrome[c_src]], dim=-1)
            c_msg = self.c_to_q_mlp(c_in)
            qubit_aggregate = scatter_add(c_msg, c_dst, x.size(0))

            q_old = x[q_idx]
            q_new = self.qubit_gru(qubit_aggregate[q_idx], q_old)
            q_new = self.qubit_norm(q_old + q_new)

            x[q_idx] = q_new

        return x