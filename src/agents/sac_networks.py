import torch
import torch.nn as nn
import torch.nn.functional as F
from src.agents.common import  get_action_mask, get_qubit_mask, _get_batch, graph_mean_pool
from src.agents.encoders import NeuralBPEncoder, CGNNEncoder

class GNNActor(nn.Module):
    def __init__(self, config, env):
        super().__init__()

        self.env = env

        if config.use_neural_bp:
            self.encoder = NeuralBPEncoder(config, env)
            encoder_output_dim = config.neural_bp_hidden_dim
        else:
            self.encoder = CGNNEncoder(config, env)
            encoder_output_dim = config.hidden_layers_cgnn[-1]

        action_head_layers = []
        prev_dim = encoder_output_dim * 2
        for hidden_dim in config.hidden_layers_mlp:
            action_head_layers.append(nn.Linear(prev_dim, hidden_dim))
            action_head_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        action_head_layers.append(nn.Linear(prev_dim, 2))

        self.action_head = nn.Sequential(*action_head_layers)


    def forward(self, state):
        data_qubit_mask = get_qubit_mask(state)

        h = self.encoder(state)
        batch = _get_batch(state, h.size(0), h.device)

        graph_feat = graph_mean_pool(h, state, batch)
        num_graphs = graph_feat.size(0)

        q_h = h[data_qubit_mask]
        q_graph_feat = graph_feat[batch[data_qubit_mask]]
        q_input = torch.cat([q_h, q_graph_feat], dim=-1)
        q_values = self.action_head(q_input).squeeze(-1)

        x_logits = q_values[:, 0].view(num_graphs, -1)
        z_logits = q_values[:, 1].view(num_graphs, -1)

        logits = torch.cat([x_logits, z_logits], dim=-1)


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
            encoder_output_dim = config.hidden_layers_cgnn[-1]

        action_head_layers = []
        prev_dim = encoder_output_dim*2
        for hidden_dim in config.hidden_layers_mlp:
            action_head_layers.append(nn.Linear(prev_dim, hidden_dim))
            action_head_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        action_head_layers.append(nn.Linear(prev_dim, 2))
        self.action_head = nn.Sequential(*action_head_layers)


    def forward(self, state, action=None):
        data_qubit_mask = get_qubit_mask(state)

        h = self.encoder(state)
        batch = _get_batch(state, h.size(0), h.device)
        graph_feat = graph_mean_pool(h, state, batch)
        num_graphs = graph_feat.size(0)

        q_h = h[data_qubit_mask]
        q_graph_feat = graph_feat[batch[data_qubit_mask]]

        q_input = torch.cat([q_h, q_graph_feat], dim=-1)
        q_values = self.action_head(q_input).squeeze(-1)

        x_q = q_values[:, 0].view(num_graphs, -1)
        z_q = q_values[:, 1].view(num_graphs, -1)

        logits = torch.cat([x_q, z_q], dim=-1)

        if action is not None:
            action = action.long().view(-1, 1)
            logits = logits.gather(1, action)

        return logits