import torch
import torch.nn as nn

class GraphResidualFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feat_dim = config['model']['embedding_dim']
        
        # Learnable scaling factor lambda
        # Initialized to 0 so the model starts with pure MedSAM features
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Layer normalization to stabilize the fused features
        self.norm = nn.LayerNorm(self.feat_dim)

    def forward(self, z_encoder, z_graph):
        """
        Args:
            z_encoder (torch.Tensor): (B, 2048, 768) - Original MedSAM features
            z_graph (torch.Tensor): (B, 2048, 768) - Features refined via GAT
        Returns:
            torch.Tensor: (B, 2048, 768) - Fused features for the mask decoder
        """
        # Z_final = Z_encoder + λ * Z_graph
        # The learnable gamma allows the model to control the influence of the graph
        z_final = z_encoder + self.gamma * z_graph
        
        # Apply normalization before passing to the decoder
        return self.norm(z_final)