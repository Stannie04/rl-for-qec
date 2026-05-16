import torch
from torch import nn
from torch_geometric.nn import global_mean_pool, GraphConv
import torch.nn.functional as F

class GNN(torch.nn.Module):
    """
    GNN with several consecutive GraphConv layers, whose final output is
    converted to a single graph embedding (feature vector) with global_mean_pool.
    This graph embedding is passed to a dense network which perform binary
    classification. The binary classifications represent the 2 logical
    equivalence classes.
    """

    def __init__(
            self,
            edge_index,
            hidden_channels_GCN=[32, 128, 256, 512, 512, 256, 256],
            hidden_channels_MLP=[256, 128, 64],
            num_node_features=3,
            n_actions=1,
    ):
        super().__init__()

        self.edge_index = torch.tensor(edge_index.detach().clone(), dtype=torch.long)

        # GCN layers
        channels = [num_node_features] + hidden_channels_GCN
        self.graph_layers = nn.ModuleList(
            [GraphConv(in_channels, out_channels)
             for (in_channels, out_channels) in zip(channels[:-1], channels[1:])]
        )

        # Dense layers for the MLP classifier
        channels = hidden_channels_GCN[-1:] + hidden_channels_MLP
        self.dense_layers = nn.ModuleList(
            [nn.Linear(in_channels, out_channels)
             for (in_channels, out_channels) in zip(channels[:-1], channels[1:])]
        )

        # Output layers (one for each class)
        self.output_layer = nn.Linear(hidden_channels_MLP[-1], n_actions)

    def forward(self, x):
        for graph_layer in self.graph_layers:
            x = F.relu(graph_layer(x, self.edge_index))

        # x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        for dense_layer in self.dense_layers:
            x = F.relu(dense_layer(x))

        return self.output_layer(x).squeeze(-1)