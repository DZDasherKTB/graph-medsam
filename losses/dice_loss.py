import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        """
        logits/target: (B, 1, D, H, W)
        """
        B = logits.shape[0]
        probs = torch.sigmoid(logits)
        
        # Flatten per sample: (B, N)
        probs = probs.view(B, -1)
        target = target.view(B, -1)

        # Compute per-sample intersection and denominator
        intersection = (probs * target).sum(dim=1)
        denominator = probs.sum(dim=1) + target.sum(dim=1)

        # Batch Dice scores
        dice = (2. * intersection + self.smooth) / (denominator + self.smooth)
        
        # Return average loss across the batch
        return 1. - dice.mean()