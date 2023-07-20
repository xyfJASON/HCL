import torch
from torchmetrics import Metric


class IntersectionOverUnion(Metric):
    full_state_update: bool = False

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.add_state('iou_sum', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        assert preds.shape == target.shape
        assert torch.eq(target.bool(), target).all()
        target = target.bool()
        preds = torch.where(torch.lt(preds, self.threshold), 0, 1).bool()
        preds = preds.flatten(start_dim=1)
        target = target.flatten(start_dim=1)

        intersection_batch = torch.sum(preds & target, dim=1).float()
        union_batch = torch.sum(preds | target, dim=1).float()
        iou_batch = intersection_batch / union_batch
        self.iou_sum += iou_batch.sum()
        self.total += preds.shape[0]

    def compute(self):
        return self.iou_sum.float() / self.total
