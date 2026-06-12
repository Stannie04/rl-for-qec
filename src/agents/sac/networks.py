from email.utils import encode_rfc2231

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv, GATConv
from src.agents.common import TannerEncoder


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


def _get_batch(data, num_nodes, device):
    if hasattr(data, "batch") and data.batch is not None:
        return data.batch
    return torch.zeros(num_nodes, dtype=torch.long, device=device)


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

        if config.use_noop_head:
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
        logits = q_logits.view(graph_feat.size(0), -1)

        if self.use_action_mask:
            action_mask = _action_mask(state, self.env)
            logits.masked_fill(~action_mask.unsqueeze(0), -1e9)

        if hasattr(self, "no_op_head"):
            noop_logit = self.no_op_head(graph_feat)
            logits = torch.cat([logits, noop_logit], dim=-1)

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

        if config.use_noop_head:
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

        if hasattr(self, "no_op_head"):
            noop_q = self.no_op_head(graph_feat)
            q_values = torch.cat([q_values, noop_q], dim=-1)

        return q_values