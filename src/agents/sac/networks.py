from email.utils import encode_rfc2231

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv, GATConv


def _action_mask(state, env):
    # Based on encoding:
    # [is_qubit, is_x_syndrome, is_z_syndrome, x_syndrome, z_syndrome]
    # mask = H_x_T @ x_syndrome + H_z_T @ z_syndrome
    # Gets only qubits that are connected to active syndromes

    x_mask = env.code.H_x_T @ state.x[env.code.x_idx, 3]
    z_mask = env.code.H_z_T @ state.x[env.code.z_idx, 4]

    return (x_mask + z_mask) > 0


def _qubit_mask(state, env):
    return torch.where(state.x[:,0] > 0.5)[0]
    # mask = torch.zeros(state.x.size(0), dtype=torch.bool, device=state.x.device)
    # mask[env.code.q_idx] = True
    # return mask


def _get_batch(data, num_nodes, device):
    if hasattr(data, "batch") and data.batch is not None:
        return data.batch
    return torch.zeros(num_nodes, dtype=torch.long, device=device)


class TannerEncoder(nn.Module):
    def __init__(self, hidden_dims, feature_dim):
        super().__init__()

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

class GNNActor(nn.Module):
    def __init__(self, config, env):
        super().__init__()

        self.env = env
        self.use_action_mask = bool(config.use_action_mask)
        self.encoder = TannerEncoder(config.hidden_layers_gnn, env.code.feature_dim)

        encoder_output_dim = config.hidden_layers_gnn[-1]

        action_head_layers = []
        prev_dim = encoder_output_dim*2
        for hidden_dim in config.hidden_layers_mlp:
            action_head_layers.append(nn.Linear(prev_dim, hidden_dim))
            action_head_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        action_head_layers.append(nn.Linear(prev_dim, 1))
        self.action_head = nn.Sequential(*action_head_layers)

        noop_head_layers = []
        prev_dim = encoder_output_dim
        for hidden_dim in config.hidden_layers_mlp:
            noop_head_layers.append(nn.Linear(prev_dim, hidden_dim))
            noop_head_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        noop_head_layers.append(nn.Linear(prev_dim, 1))
        self.no_op_head = nn.Sequential(*noop_head_layers)


    def forward(self, state):
        h = self.encoder(state)
        batch = _get_batch(state, h.size(0), h.device)

        graph_feat = global_mean_pool(h, batch)

        qubit_mask = _qubit_mask(state, self.env)

        q_h = h[qubit_mask]
        q_graph_feat = graph_feat[batch[qubit_mask]]

        q_input = torch.cat([q_h, q_graph_feat], dim=-1)
        q_logits = self.action_head(q_input).squeeze(-1)
        q_logits = q_logits.view(graph_feat.size(0), -1)

        if self.use_action_mask:
            action_mask = _action_mask(state, self.env)
            q_logits.masked_fill(~action_mask.unsqueeze(0), -1e9)

        noop_logit = self.no_op_head(graph_feat)

        logits = torch.cat([q_logits, noop_logit], dim=-1)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        return logits, log_probs, probs


class GNNCritic(nn.Module):
    def __init__(self, config, env):
        super().__init__()

        self.env = env
        self.encoder = TannerEncoder(config.hidden_layers_gnn, env.code.feature_dim)

        encoder_output_dim = config.hidden_layers_gnn[-1]

        action_head_layers = []
        prev_dim = encoder_output_dim*2
        for hidden_dim in config.hidden_layers_mlp:
            action_head_layers.append(nn.Linear(prev_dim, hidden_dim))
            action_head_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        action_head_layers.append(nn.Linear(prev_dim, 1))
        self.action_head = nn.Sequential(*action_head_layers)

        noop_head_layers = []
        prev_dim = encoder_output_dim
        for hidden_dim in config.hidden_layers_mlp:
            noop_head_layers.append(nn.Linear(prev_dim, hidden_dim))
            noop_head_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        noop_head_layers.append(nn.Linear(prev_dim, 1))
        self.no_op_head = nn.Sequential(*noop_head_layers)



    def forward(self, data, action=None):
        qubit_mask = _qubit_mask(data, self.env)

        h = self.encoder(data)
        batch = _get_batch(data, h.size(0), h.device)
        graph_feat = global_mean_pool(h, batch)

        q_h = h[qubit_mask]
        q_graph_feat = graph_feat[batch[qubit_mask]]

        q_input = torch.cat([q_h, q_graph_feat], dim=-1)
        q_values = self.action_head(q_input).squeeze(-1)
        q_values = q_values.view(graph_feat.size(0), -1)

        noop_q = self.no_op_head(graph_feat)

        q_values = torch.cat([q_values, noop_q], dim=-1)
        return q_values