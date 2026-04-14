from torch_geometric.nn import GraphConv, GCNConv
from torch import nn
import torch.nn.functional as F


class GNN(nn.Module):
    def __init__(
        self,
        num_node_features=4,
        hidden_channels=64,
        num_actions=1
    ):
        super().__init__()

        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)

        # Small MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )


    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        return self.mlp(x)