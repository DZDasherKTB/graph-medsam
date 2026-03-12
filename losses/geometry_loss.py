import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometryLoss(nn.Module):
    def __init__(self, config):
        super(GeometryLoss, self).__init__()
        # 3D Laplacian Kernel to compute local neighborhood differences
        # This kernel calculates: 6*phi(x,y,z) - sum(neighbors)
        kernel = torch.tensor([[[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]],
                               [[0, 1, 0],
                                [1, -6, 1],
                                [0, 1, 0]],
                               [[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]]], dtype=torch.float32)
        self.register_buffer('kernel', kernel.view(1, 1, 3, 3, 3))

    def forward(self, logits):
        """
        Args:
            logits (torch.Tensor): (B, 1, D, H, W) - Raw output from decoder
        Returns:
            torch.Tensor: Scalar Geometry (Smoothness) loss
        """
        probs = torch.sigmoid(logits)
        
        # Compute the Laplacian of the probability map
        # High values indicate high-frequency noise or jagged edges
        laplacian = F.conv3d(probs, self.kernel, padding=1)
        
        # L2 norm of the Laplacian encourages local smoothness
        loss = torch.mean(torch.pow(laplacian, 2))
        
        return loss