import torch
import torch.nn as nn
import math
from torch_scatter import scatter_mean

class SupervoxelClusterer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_nodes = config['model']['num_supervoxels']
        self.feat_dim = config['model']['embedding_dim']
        
        # 1. Spatial Anchors (Fixed Grid)
        grid_size = int(math.ceil(self.num_nodes ** (1/3)))
        steps = torch.linspace(0, 1, grid_size)
        grid = torch.stack(torch.meshgrid(steps, steps, steps, indexing='ij'), dim=-1).view(-1, 3)
        spatial_prototypes = grid[:self.num_nodes].clone().detach()
        self.register_buffer("spatial_prototypes", spatial_prototypes)
        
        # 2. Feature Prototypes (Learnable)
        # This allows nodes to specialize in specific tissue types (vessel vs background)
        self.feature_prototypes = nn.Parameter(torch.randn(self.num_nodes, self.feat_dim))
        
        # 3. Scaling Buffers
        self.register_buffer('lambda_f', torch.tensor(config['graph']['lambda_f']))
        self.register_buffer('lambda_s', torch.tensor(config['graph']['lambda_s']))

    def forward(self, tokens, coords):
        """
        tokens: (B, N, C)
        coords: (B, N, 3)
        """
        B, N, C = tokens.shape
        device = tokens.device

        # Compute Joint Distance: D(i,j) = λf ||zi - μj||^2 + λs ||pi - πj||^2
        # Feature Distance: (B, N, K)
        dist_f = torch.cdist(tokens, self.feature_prototypes.unsqueeze(0).expand(B, -1, -1)).pow(2)
        
        # Spatial Distance: (B, N, K)
        dist_s = torch.cdist(coords, self.spatial_prototypes.unsqueeze(0).expand(B, -1, -1)).pow(2)

        # Combined Weighted Distance
        joint_dist = (self.lambda_f * dist_f) + (self.lambda_s * dist_s)
        cluster_ids = torch.argmin(joint_dist, dim=-1) # (B, N)

        # Vectorized Aggregation
        flat_tokens = tokens.reshape(B * N, C)
        flat_coords = coords.reshape(B * N, 3)
        
        offsets = torch.arange(B, device=device).unsqueeze(1) * self.num_nodes
        batch_cluster_ids = (cluster_ids + offsets).reshape(-1)

        dim_size = B * self.num_nodes
        node_features = scatter_mean(flat_tokens, batch_cluster_ids, dim=0, dim_size=dim_size)
        node_coords = scatter_mean(flat_coords, batch_cluster_ids, dim=0, dim_size=dim_size)

        # Reshape to (B, K, C)
        node_features = node_features.reshape(B, self.num_nodes, C)
        node_coords = node_coords.reshape(B, self.num_nodes, 3)

        # Robustness Fallback: Use spatial prototypes if a cluster is empty
        mask = (node_features.abs().sum(dim=-1, keepdim=True) == 0)
        if mask.any():
            node_coords = torch.where(mask.expand(-1, -1, 3), 
                                     self.spatial_prototypes.unsqueeze(0).expand(B, -1, -1), 
                                     node_coords)
            # For empty features, we assign the learnable feature prototype as a baseline
            node_features = torch.where(mask.expand(-1, -1, C),
                                       self.feature_prototypes.unsqueeze(0).expand(B, -1, -1),
                                       node_features)

        return cluster_ids, node_features, node_coords