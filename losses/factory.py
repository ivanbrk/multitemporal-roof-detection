import torch.nn as nn

from .tversky_focal import TverskyFocalLoss


def build_loss(name, class_weights=None):
    normalized = str(name).lower()
    if normalized in ("tversky_focal", "focal_tversky", "tversky"):
        return TverskyFocalLoss(class_weights=class_weights)
    if normalized in ("bce", "bcewithlogits"):
        pos_weight = None
        if class_weights is not None and len(class_weights) == 2 and class_weights[0] > 0:
            pos_weight = class_weights[1] / class_weights[0]
        if pos_weight is not None:
            import torch

            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
        return nn.BCEWithLogitsLoss()
    raise ValueError("Unsupported loss: %s" % name)
