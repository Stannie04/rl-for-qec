import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv, GATConv

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