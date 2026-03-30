import torch

from utils.distributed import reduce_sum_tensor


class SegmentationMeter(object):
    def __init__(self, threshold=0.5):
        self.threshold = float(threshold)
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

    def update_from_logits(self, logits, targets):
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= self.threshold).float()
        self.update(predictions, targets)

    def update(self, predictions, targets):
        predictions = predictions.float()
        targets = targets.float()

        self.tp += float(((predictions == 1) * (targets == 1)).sum().item())
        self.fp += float(((predictions == 1) * (targets == 0)).sum().item())
        self.fn += float(((predictions == 0) * (targets == 1)).sum().item())
        self.tn += float(((predictions == 0) * (targets == 0)).sum().item())

    def synchronize_between_processes(self, device):
        state = torch.tensor([self.tp, self.fp, self.fn, self.tn], dtype=torch.float64, device=device)
        state = reduce_sum_tensor(state)
        self.tp, self.fp, self.fn, self.tn = [float(item) for item in state.tolist()]

    def compute(self):
        return compute_metrics_from_counts(self.tp, self.fp, self.fn, self.tn)


def compute_metrics_from_counts(tp, fp, fn, tn):
    del tn
    epsilon = 1e-8
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = (2.0 * precision * recall) / (precision + recall + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "iou": iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
