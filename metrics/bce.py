import torch
import torch.nn.functional as F
from torchmetrics import Metric


class BinaryCrossEntropy(Metric):
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state('bce_sum', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        assert preds.shape == target.shape
        bce_batch_mean = F.binary_cross_entropy(preds, target.float(), reduction='mean')
        self.bce_sum += bce_batch_mean * preds.shape[0]
        self.total += preds.shape[0]

    def compute(self):
        return self.bce_sum.float() / self.total
