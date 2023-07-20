from typing import Dict

import torch
from torchmetrics import Metric


class AverageMeter(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, val: torch.Tensor, n: int):
        self.sum += val * n
        self.total += n

    def compute(self):
        return self.sum / self.total


class KeyValueAverageMeter(Metric):
    full_state_update = False

    def __init__(self, keys):
        super().__init__()
        self.keys = keys
        for k in keys:
            self.add_state(f'{k}_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, kvs: Dict, n: int):
        assert set(kvs.keys()) == set(self.keys)
        for key, val in kvs.items():
            s = getattr(self, f'{key}_sum')
            s += val * n
        self.total += n

    def compute(self):
        return {k: getattr(self, f'{k}_sum') / self.total for k in self.keys}
