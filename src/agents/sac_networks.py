import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv, GATConv
from src.agents.common import  get_action_mask, get_qubit_mask
from src.agents.encoders import NeuralBPEncoder, CGNNEncoder


def _get_batch(data, num_nodes, device):
    if hasattr(data, "batch") and data.batch is not None:
        return data.batch
    return torch.zeros(num_nodes, dtype=torch.long, device=device)


class GNNActor(nn.Module):
    def __init__(self, config, env):
        super().__init__()

        self.env = env
        self.use_action_mask = bool(config.use_action_mask)

        if config.use_neural_bp:
            self.encoder = NeuralBPEncoder(config, env)
            encoder_output_dim = config.neural_bp_hidden_dim
        else:
            self.encoder = CGNNEncoder(config, env)
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
        data_qubit_mask = get_qubit_mask(state)

        h = self.encoder(state)
        batch = _get_batch(state, h.size(0), h.device)

        graph_feat = global_mean_pool(h, batch)

        q_h = h[data_qubit_mask]
        q_graph_feat = graph_feat[batch[data_qubit_mask]]
        q_input = torch.cat([q_h, q_graph_feat], dim=-1)
        q_logits = self.action_head(q_input).squeeze(-1)
        logits = q_logits.view(graph_feat.size(0), -1)

        if self.use_action_mask:
            action_mask = get_action_mask(state, self.env)
            logits = logits.masked_fill(~action_mask.unsqueeze(0), -1e9)

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

        if config.use_neural_bp:
            self.encoder = NeuralBPEncoder(config, env)
            encoder_output_dim = config.neural_bp_hidden_dim
        else:
            self.encoder = CGNNEncoder(config, env)
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


    def forward(self, state, action=None):
        data_qubit_mask = get_qubit_mask(state)

        h = self.encoder(state)
        batch = _get_batch(state, h.size(0), h.device)
        graph_feat = global_mean_pool(h, batch)

        q_h = h[data_qubit_mask]
        q_graph_feat = graph_feat[batch[data_qubit_mask]]

        q_input = torch.cat([q_h, q_graph_feat], dim=-1)
        q_values = self.action_head(q_input).squeeze(-1)
        q_values = q_values.view(graph_feat.size(0), -1)

        if hasattr(self, "no_op_head"):
            noop_q = self.no_op_head(graph_feat)
            q_values = torch.cat([q_values, noop_q], dim=-1)

        if action is not None:
            q_values = q_values.gather(1, action.long().unsqueeze(-1))

        return q_values