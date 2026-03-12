import torch
import torch.nn as nn

class FeatureBroadcaster(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_nodes = config['model']['num_supervoxels']
        self.feat_dim = config['model']['gnn_hidden_dim']

    def forward(self, node_features, cluster_ids):
        """
        node_features: (B*K, C) - Refined features from GNN
        cluster_ids: (B, N) - Token cluster assignments
        """
        B, N = cluster_ids.shape
        K, C = self.num_nodes, self.feat_dim

        # 1. Reshape nodes to (B, K, C)
        node_features = node_features.view(B, K, C)

        # 2. Vectorized Gather (The 'Fix')
        # We create a batch index [0, 0...1, 1...] to align with cluster_ids
        batch_idx = torch.arange(B, device=node_features.device).view(B, 1)
        
        # This maps (B, N) -> (B, N, C) without a loop
        z_graph = node_features[batch_idx, cluster_ids]

        return z_graph