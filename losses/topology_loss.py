import torch
import torch.nn as nn
import torch.nn.functional as F

class TopologyLoss(nn.Module):
    def __init__(self, iterations=3):
        super(TopologyLoss, self).__init__()
        self.iterations = iterations

    def soft_skeletonize(self, x):
        """
        Applies differentiable soft-skeletonization using min/max pooling.
        This preserves the thin structures (vessels) while eroding the bulk.
        """
        for _ in range(self.iterations):
            # Soft erosion: Local min
            # Soft dilation: Local max
            erosion = -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)
            dilation = F.max_pool3d(erosion, kernel_size=3, stride=1, padding=1)
            # The skeleton is the difference between the image and its opening
            x = torch.clamp(x - dilation, 0, 1)
        return x

    def forward(self, logits, target):
        """
        Args:
            logits (torch.Tensor): (B, 1, D, H, W)
            target (torch.Tensor): (B, 1, D, H, W)
        Returns:
            torch.Tensor: Scalar Topology loss
        """
        probs = torch.sigmoid(logits)
        
        # Extract soft skeletons for both prediction and ground truth
        skel_pred = self.soft_skeletonize(probs)
        skel_true = self.soft_skeletonize(target.float())

        # Tversky-like penalty to focus on false negatives (broken connections)
        # We penalize cases where the GT skeleton exists but the pred skeleton doesn't
        intersection = (skel_pred * skel_true).sum()
        false_neg = (skel_true * (1 - skel_pred)).sum()
        
        # Focus heavily on recovering missing connections
        loss = false_neg / (false_neg + intersection + 1e-6)
        
        return loss