import torch.nn as nn
from torch import Tensor

from models.modules import ConvNormAct


class PatchDiscriminator(nn.Module):
    def __init__(self, base_n_channels: int = 64):
        super().__init__()
        self.disc = nn.Sequential(
            ConvNormAct(3, base_n_channels, 5, 2, 2, activation='leakyrelu', sn=True),
            ConvNormAct(base_n_channels, 2 * base_n_channels, 5, 2, 2, activation='leakyrelu', sn=True),
            ConvNormAct(2 * base_n_channels, 2 * base_n_channels, 5, 2, 2, activation='leakyrelu', sn=True),
            ConvNormAct(2 * base_n_channels, 4 * base_n_channels, 5, 2, 2, activation='leakyrelu', sn=True),
            ConvNormAct(4 * base_n_channels, 8 * base_n_channels, 5, 2, 2, activation='leakyrelu', sn=True),
        )

    def forward(self, X: Tensor):
        return self.disc(X)
