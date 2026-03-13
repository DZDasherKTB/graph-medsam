import torch
import torch.nn as nn

class MedSAMEncoder(nn.Module):
    def __init__(self, sam_model, frozen_layers=8):
        super().__init__()
        self.encoder = sam_model.image_encoder
        
        # Proper parameter-level freezing
        for i, block in enumerate(self.encoder.blocks):
            if i < frozen_layers:
                for param in block.parameters():
                    param.requires_grad = False

    def forward(self, x):
        """
        x: (B, 1, D, H, W) or (B, 1, H, W)
        """
        # Issue 2 Fix: Handle 1-channel to 3-channel replication for SAM backbone
        if x.shape[1] == 1:
            # Replicate along the channel dimension (dim 1)
            x = x.repeat(1, 3, 1, 1, 1) if x.dim() == 5 else x.repeat(1, 3, 1, 1)

        tokens = self.encoder(x)
        
        # Issue 1 Fix: Handle both 5D (3D volumes) and 4D (2D slices) encoder outputs
        if tokens.dim() == 5:
            B, C, D, H, W = tokens.shape
            tokens = tokens.permute(0, 2, 3, 4, 1).contiguous()
            tokens = tokens.view(B, -1, C) # (B, 2048, 768)
        elif tokens.dim() == 4:
            B, C, H, W = tokens.shape
            tokens = tokens.permute(0, 2, 3, 1).contiguous()
            tokens = tokens.view(B, -1, C)
            
        return tokens