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


def get_action_mask(state, env):
    # Based on encoding:
    # [is_qubit, is_x_syndrome, is_z_syndrome, x_syndrome, z_syndrome]
    # mask = H_x_T @ x_syndrome + H_z_T @ z_syndrome
    # Gets only qubits that are connected to active syndromes

    x_mask = env.code.H_x_T @ state.x[env.code.x_idx, 3]
    z_mask = env.code.H_z_T @ state.x[env.code.z_idx, 4]

    return (x_mask + z_mask) > 0


def get_qubit_mask(state):
    return torch.where(state.x[:,0] > 0.5)[0]


class TannerEncoder(nn.Module):
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

        self.q_to_c_mlp = MLP(hidden_dim, mlp_hidden_dims)
        self.c_to_q_mlp = MLP(hidden_dim + hidden_dim + 2, mlp_hidden_dims) # +2 for syndrome info

        self.qubit_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.check_gru = nn.GRUCell(hidden_dim, hidden_dim)

        self.qubit_norm = nn.LayerNorm(hidden_dim)
        self.check_norm = nn.LayerNorm(hidden_dim)

        self.num_iters = num_iters

    def forward(self, data):
        x = F.relu(self.input_proj(data.x))
        syndrome = data.x[:, 3:5] # Raw syndrome info for action masking

        src, dst = data.edge_index # Edges go between qubits and checks
        qubit_mask = data.x[:,0] > 0.5
        q_src = src[qubit_mask[src]]
        q_dst = dst[qubit_mask[src]]

        check_mask = ~qubit_mask
        c_src = src[check_mask[src]]
        c_dst = dst[check_mask[src]]

        for _ in range(self.num_iters):
            # Qubits to checks
            q_msg = self.q_to_c_mlp(x[q_src])
            check_aggregate = scatter_add(q_msg, q_dst, x.size(0))

            # Checks to qubits
            c_input = torch.cat([x[c_src], check_aggregate[c_src], syndrome[c_src]], dim=-1)
            c_msg = self.c_to_q_mlp(c_input)
            qubit_aggregate = scatter_add(c_msg, c_dst, x.size(0))

            # Recurrent updates
            gru_out = torch.empty_like(x)
            gru_out[check_mask] = self.check_norm(self.check_gru(check_aggregate[check_mask], x[check_mask]))
            gru_out[qubit_mask] = self.qubit_norm(self.qubit_gru(qubit_aggregate[qubit_mask], x[qubit_mask]))
            x = gru_out

        return x

class NeuralBPPretrainer(nn.Module):
    def __init__(self, encoder, config, num_qubits):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.output_layer = nn.Linear(config.encoder_hidden_dim, 1)
        self.num_qubits = num_qubits

    def forward(self, data):
        x = self.encoder(data)
        qubit_mask = get_qubit_mask(data)
        return torch.sigmoid(self.output_layer(x[qubit_mask])).squeeze(-1)  # Output shape: (num_qubits,)

    def save(self, path):
        torch.save({
        'encoder': self.encoder.state_dict(),
        'output_layer': self.output_layer.state_dict(),
        }, path)