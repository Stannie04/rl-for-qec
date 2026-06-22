import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv, GATConv
from src.agents.common import TannerEncoder, NeuralBPEncoder


class Router(nn.Module):
    def __init__(self, config, env):
        super().__init__()

        self.env = env
        self.device = env.device
        # self.encoder = TannerEncoder(config.moe_hidden_layers_gnn, env.code.feature_dim)
        # prev_dim = config.moe_hidden_layers_gnn[-1]

        self.encoder = NeuralBPEncoder(config, env)
        prev_dim = config.encoder_hidden_dim

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


    def save(self):
        torch.save(self.state_dict(), "checkpoints/router.pt")

class MoEAgent:
    def __init__(self, config, env):
        self.env = env
        self.device = env.device
        self.router = Router(config, env).to(self.device)
        self.optimizer = torch.optim.Adam(self.router.parameters(), lr=config.moe_learning_rate)

        self.baseline = 0.0
        self.baseline_alpha = 0.99


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

    def save(self):
        self.router.save()