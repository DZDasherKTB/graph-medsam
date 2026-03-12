import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class VesselGNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feat_dim = config['model']['embedding_dim']
        self.hidden_dim = config['model']['gnn_hidden_dim']
        self.dropout = config['model']['dropout']
        
        # Projection layer to align MedSAM tokens with GNN hidden space
        self.proj_in = nn.Linear(self.feat_dim, self.hidden_dim) if self.feat_dim != self.hidden_dim else nn.Identity()

        # GraphSAGE Layers
        self.sage1 = SAGEConv(self.hidden_dim, self.hidden_dim)
        self.sage2 = SAGEConv(self.hidden_dim, self.hidden_dim)
        
        # Intermediate and Final Normalization
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm_final = nn.LayerNorm(self.hidden_dim)

    def forward(self, x, edge_index, edge_weight):
        """
        x: (B*K, C) - Flattened supervoxel node features
        edge_index: (2, E)
        edge_weight: (E,)
        """
        # Align dimensions and save identity for residual
        x = self.proj_in(x)
        identity = x

        # Layer 1: SAGE -> ReLU -> Norm -> Dropout
        x = self.sage1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.norm1(x) # Improvement: Intermediate stability
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2: SAGE -> Dropout (Residual logic follows)
        x = self.sage2(x, edge_index, edge_weight=edge_weight)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final Residual addition and normalization
        return self.norm_final(x + identity)