import torch
import torch.nn as nn

class BoundaryLoss(nn.Module):
    def __init__(self, config):
        super(BoundaryLoss, self).__init__()

    def forward(self, logits, sdf_tensor):
        """
        sdf_tensor: (B, 1, D, H, W) - Pre-computed in Dataset Loader
        """
        probs = torch.sigmoid(logits)
        
        # Standard Boundary Loss formulation:
        # Integrating the probability over the distance field.
        # Inside GT: SDF is negative | Outside GT: SDF is positive
        # This encourages probabilities to stay in the negative zone (inside).
        loss = (probs * sdf_tensor).mean()
        
        return loss