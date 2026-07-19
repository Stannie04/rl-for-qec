import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv, GATConv
from src.agents.encoders import NeuralBPEncoder, CGNNEncoder


class Router(nn.Module):
    def __init__(self, config, env):
        super().__init__()

        self.env = env
        self.device = env.device

        if config.use_neural_bp:
            self.encoder = NeuralBPEncoder(config, env)
            prev_dim = config.encoder_hidden_dim
        else:
            self.encoder = CGNNEncoder(config.moe_hidden_layers_gnn, env.code.feature_dim)
            prev_dim = config.moe_hidden_layers_gnn[-1]

        classification_layers = []
        for hidden_dim in config.moe_hidden_layers_mlp:
            classification_layers.append(nn.Linear(prev_dim, hidden_dim))
            classification_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        classification_layers.append(nn.Linear(prev_dim, len(config.moe_experts)))
        self.action_head = nn.Sequential(*classification_layers)

        self = self.to(self.device)


    def forward(self, x):
        h = self.encoder(x)
        graph_feat = global_mean_pool(h, torch.zeros(h.size(0), dtype=torch.long, device=h.device))
        return self.action_head(graph_feat)


class RouterAgent:
    def __init__(self, config, env, router_checkpoint=None, encoder_checkpoint=None):
        self.env = env
        self.device = env.device
        self.router = Router(config, env).to(self.device)
        self.optimizer = torch.optim.Adam(self.router.parameters(), lr=config.moe_learning_rate)

        self.baseline = 0.0
        self.baseline_alpha = 0.99

        if router_checkpoint is not None:
            self.load_router(router_checkpoint)

        if encoder_checkpoint is not None:
            self.load_encoder(encoder_checkpoint)


    def update(self, log_prob, reward):
        self.baseline = self.baseline_alpha * self.baseline + (1 - self.baseline_alpha) * reward
        advantage = reward - self.baseline
        loss = -log_prob * advantage

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def select_action(self, obs, evaluate=False):
        logits = self.router(obs)

        if evaluate:
            action = torch.argmax(logits, dim=-1)
            return action.item(), None

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        return action.item(), dist.log_prob(action)

    def save(self, path):
        torch.save(self.router.state_dict(), path)

    def load_router(self, checkpoint):
        self.router.load_state_dict(torch.load(checkpoint, map_location=self.device))

    def load_encoder(self, checkpoint):
        full_cp = torch.load(checkpoint, map_location=self.device)
        self.router.encoder.load_state_dict(full_cp["encoder"])