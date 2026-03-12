import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        """
        Args:
            logits (torch.Tensor): (B, 1, D, H, W) - Raw output from MedSAM decoder
            target (torch.Tensor): (B, 1, D, H, W) - Ground truth binary mask
        Returns:
            torch.Tensor: Scalar Dice loss
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Flatten tensors for calculation
        probs = probs.view(-1)
        target = target.view(-1)

        intersection = (probs * target).sum()
        denominator = probs.sum() + target.sum()

        # Dice = 2 * |P ∩ G| / (|P| + |G|)
        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
        
        return 1. - dice_score