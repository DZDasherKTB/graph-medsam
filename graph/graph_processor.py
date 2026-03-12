import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class DifferentiableGraphModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_nodes = config['model']['num_supervoxels']
        self.feat_dim = config['model']['embedding_dim']
        
        # 1. Soft Clustering: Learnable Prototypes (Nodes)
        self.prototypes = nn.Parameter(torch.randn(self.num_nodes, self.feat_dim))
        
        # 2. GraphSAGE Layers
        self.sage1 = SAGEConv(self.feat_dim, self.feat_dim)
        self.sage2 = SAGEConv(self.feat_dim, self.feat_dim)
        
        # 3. Normalization for Fusion
        self.graph_norm = nn.LayerNorm(self.feat_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, tokens, edge_index, edge_weights):
        """
        tokens: (B, N, C) - from MedSAM
        """
        B, N, C = tokens.shape
        
        # --- STEP 1: Differentiable Soft Assignment ---
        # Compute similarity between tokens and prototypes: (B, N, K)
        # This replaces Hard KMeans
        attn_weights = torch.matmul(tokens, self.prototypes.t()) 
        attn_weights = F.softmax(attn_weights, dim=-1) 
        
        # Aggregate tokens into nodes: (B, K, C)
        node_features = torch.matmul(attn_weights.transpose(1, 2), tokens)
        
        # --- STEP 2: GraphSAGE Refinement ---
        # Flatten batch for PyG
        x = node_features.view(-1, C) 
        
        h = self.sage1(x, edge_index)
        h = F.relu(h)
        h = self.sage2(h, edge_index)
        
        # --- STEP 3: Soft Broadcast ---
        # Map back to token space using the same attention weights
        # (B, K, C) -> (B, N, C)
        h_nodes = h.view(B, self.num_nodes, C)
        z_graph = torch.matmul(attn_weights, h_nodes)
        
        # --- STEP 4: Normalized Residual Fusion ---
        z_graph = self.graph_norm(z_graph)
        z_final = tokens + self.gamma * z_graph
        
        return z_final