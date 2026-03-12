import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as edt

class BoundaryLoss(nn.Module):
    def __init__(self, config):
        super(BoundaryLoss, self).__init__()
        # Weight from config is handled in the final trainer.py loss summation

    def compute_sdf(self, mask_np):
        """
        Computes the Signed Distance Function (SDF) for a 3D binary mask.
        """
        posmask = mask_np.astype(bool)
        if posmask.any():
            negmask = ~posmask
            # Distance to boundary from outside and inside
            dist_out = edt(negmask)
            dist_in = edt(posmask)
            # Combine: Positive outside, negative inside
            sdf = dist_out - dist_in
        else:
            sdf = np.zeros(mask_np.shape)
        return sdf

    def forward(self, logits, target):
        """
        Args:
            logits (torch.Tensor): (B, 1, D, H, W)
            target (torch.Tensor): (B, 1, D, H, W)
        Returns:
            torch.Tensor: Scalar Boundary loss
        """
        probs = torch.sigmoid(logits)
        target_np = target.detach().cpu().numpy()
        
        # Calculate Distance Transform for each volume in the batch
        # This is a non-differentiable step performed on CPU/NumPy
        sdf_list = []
        for b in range(target_np.shape[0]):
            sdf = self.compute_sdf(target_np[b, 0])
            sdf_list.append(sdf)
            
        sdf_tensor = torch.from_numpy(np.stack(sdf_list)).float().to(logits.device)
        sdf_tensor = sdf_tensor.unsqueeze(1) # (B, 1, D, H, W)

        # Boundary Loss = Mean(Product of Probability and Distance Map)
        # Note: target * sdf_tensor usually equals negative distances (inside GT)
        # and (1-target) * sdf_tensor equals positive distances (outside GT)
        loss = probs * sdf_tensor
        
        return loss.mean()