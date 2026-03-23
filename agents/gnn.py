from torch_geometric.nn import GraphConv, global_mean_pool
from torch import nn
import torch


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
            hidden_channels_GCN=[32, 128, 256, 512, 512, 256, 256],
            hidden_channels_MLP=[256, 128, 64],
            num_node_features=4,
            num_actions=1,
            manual_seed=None,
    ):
        super().__init__()
        if manual_seed is not None:
            torch.manual_seed(manual_seed)

        # GCN layers
        gcn_channels = [num_node_features] + hidden_channels_GCN
        self.graph_layers = nn.ModuleList(
            [GraphConv(in_channels, out_channels)
             for (in_channels, out_channels) in zip(gcn_channels[:-1], gcn_channels[1:])]
        )

        # Dense layers for the MLP classifier
        mlp_channels = hidden_channels_GCN[-1:] + hidden_channels_MLP
        self.dense_layers = nn.ModuleList(
            [nn.Linear(in_channels, out_channels)
             for (in_channels, out_channels) in zip(mlp_channels[:-1], mlp_channels[1:])]
        )

        # Output layers (one for each class)
        self.output_layer = nn.Linear(hidden_channels_MLP[-1], num_actions)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        #  node embeddings
        for layer in self.graph_layers:
            x = layer(x, edge_index, edge_attr)
            x = torch.nn.functional.relu(x, inplace=True)

        # pass through MLPs
        for layer in self.dense_layers:
            x = layer(x)
            x = torch.nn.functional.relu(x, inplace=True)

        return self.output_layer(x)