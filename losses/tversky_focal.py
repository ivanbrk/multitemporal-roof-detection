import torch
import torch.nn as nn


class TverskyFocalLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1.0, class_weights=None):
        super(TverskyFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        if class_weights is None:
            class_weights = (1.0, 1.0)
        self.bg_weight = float(class_weights[0])
        self.fg_weight = float(class_weights[1])

    def forward(self, logits, targets):
        targets = targets.float()
        probabilities = torch.sigmoid(logits)
        dims = (1, 2, 3)

        tp = (self.fg_weight * probabilities * targets).sum(dim=dims)
        fp = (self.bg_weight * probabilities * (1.0 - targets)).sum(dim=dims)
        fn = (self.fg_weight * (1.0 - probabilities) * targets).sum(dim=dims)

        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        focal_tversky = torch.pow(1.0 - tversky_index, self.gamma)
        return focal_tversky.mean()
