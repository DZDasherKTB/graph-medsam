import torch
import torch.nn as nn
import torch.nn.functional as F

class TopologyLoss(nn.Module):
    def __init__(self, iterations=3):
        super(TopologyLoss, self).__init__()
        self.iterations = iterations

    def soft_skeletonize(self, x):
        for _ in range(self.iterations):
            erosion = -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)
            dilation = F.max_pool3d(erosion, kernel_size=3, stride=1, padding=1)
            x = torch.clamp(x - dilation, 0, 1)
        return x

    def forward(self, logits, target, current_epoch=0, start_epoch=5):
        if current_epoch < start_epoch:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        probs = torch.sigmoid(logits)
        skel_pred = self.soft_skeletonize(probs)
        skel_true = self.soft_skeletonize(target.float())

        intersection = (skel_pred * skel_true).sum()
        false_neg = (skel_true * (1 - skel_pred)).sum()
        
        return false_neg / (false_neg + intersection + 1e-6)