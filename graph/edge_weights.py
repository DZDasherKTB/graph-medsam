import torch
import torch.nn as nn
from torch_cluster import knn_graph
from torch_geometric.utils import to_undirected

class EdgeWeightProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.register_buffer('sigma_f', torch.tensor(config['graph']['sigma_f']))
        self.register_buffer('sigma_s', torch.tensor(config['graph']['sigma_s']))
        self.k = 8
        self.feat_dim = config['model']['embedding_dim']

    def forward(self, node_features, node_coords):
        """
        node_features: (B, K, C)
        node_coords: (B, K, 3)
        """
        B, K, C = node_features.shape
        device = node_features.device
        
        # 1. k-NN Graph Construction
        flat_coords = node_coords.view(-1, 3)
        batch = torch.arange(B, device=device).repeat_interleave(K)
        
        # Build directed k-NN
        edge_index = knn_graph(flat_coords, k=self.k, batch=batch, loop=False)

        # Problem 1 Fix: Convert to undirected for symmetric message passing
        # This ensures that if node i influences j, j influences i equally
        edge_index = to_undirected(edge_index)

        # 2. Vectorized Edge Weight Calculation
        u, v = edge_index[0], edge_index[1]
        flat_features = node_features.view(-1, C)
        
        # Normalized Feature Distance: Scale by dimension to prevent saturation
        # ||zi - zj||^2 / C
        diff_f = (flat_features[u] - flat_features[v]).pow(2).sum(dim=-1) / self.feat_dim
        
        # Spatial Distance
        diff_p = (flat_coords[u] - flat_coords[v]).pow(2).sum(dim=-1)

        # Problem 2 Fix: Stable Gaussian Kernel
        # Using 2 * sigma^2 denominator and dimension-normalized features
        denom_f = 2 * self.sigma_f.pow(2) + 1e-6
        denom_p = 2 * self.sigma_s.pow(2) + 1e-6
        
        weights = torch.exp(-diff_f / denom_f) * torch.exp(-diff_p / denom_p)

        return edge_index, weights