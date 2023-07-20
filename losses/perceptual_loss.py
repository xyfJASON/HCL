from typing import Sequence
import torch
import torch.nn as nn


class PerceptualLoss(nn.Module):
    def __init__(self,
                 feature_extractor: nn.Module,
                 use_features: Sequence[str] = ('relu2_2', 'relu3_2', 'relu4_2'),
                 weights: Sequence[float] = (1., 1., 1.)):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.feature_extractor = feature_extractor
        self.use_features = use_features
        self.weights = weights

    def forward(self, fake_img: torch.Tensor, real_img: torch.Tensor):
        fake_feature = self.feature_extractor(fake_img)
        real_feature = self.feature_extractor(real_img)
        loss = sum(w * self.l1(fake_feature[f], real_feature[f]) for f, w in zip(self.use_features, self.weights))
        return loss
