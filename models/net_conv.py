from typing import List, Mapping, Any

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from models.modules import ConvNormAct, PatchResizing2d, GatedConv2d, ResBlock, Upsample, Downsample
from models.modules import confidence_mask_hierarchy


class StackedResBlock(nn.Module):
    def __init__(self, depth: int, *args, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(ResBlock(*args, **kwargs) for _ in range(depth))

    def forward(self, X: Tensor, mask: Tensor = None):
        for blk in self.blocks:
            X = blk(X, mask)
            if isinstance(X, (tuple, list)):
                X, mask = X
        if mask is None:
            return X
        else:
            return X, mask


class Encoder(nn.Module):
    def __init__(
            self,
            img_size: int = 256,                # size of input image
            dim: int = 64,                      # channels after the first convolution
            n_conv_stages: int = 0,             # number of convolution stages
                                                # the input image will be downsampled by 2 in each stage
            dim_mults: List[int] = (1, 2, 4),   # a list of channel multiplers in transformer stages
                                                # the length is the number of transformer stages
            depths: List[int] = (6, 4, 2),      # number of blocks in each transformer stage
                                                # the length should be the same as dim_mults
            kernel_size: int = 3,               # kernel size of convolution layers
    ):
        super().__init__()
        assert len(dim_mults) == len(depths)
        self.n_stages = len(dim_mults)
        self.dims = [dim * dim_mults[i] for i in range(self.n_stages)]
        res = img_size

        # first conv
        self.first_conv = ConvNormAct(3, dim, kernel_size=5, stride=1, padding=2, activation='gelu')

        # convolution stages
        self.conv_down_blocks = nn.ModuleList([])
        for i in range(n_conv_stages):
            self.conv_down_blocks.append(nn.Sequential(
                Downsample(dim, dim),
                nn.GELU(),
            ))
            res = res // 2

        # convolution stages
        self.down_blocks = nn.ModuleList([])
        for i in range(self.n_stages):
            self.down_blocks.append(nn.ModuleList([
                StackedResBlock(
                    depth=depths[i],
                    in_channels=self.dims[i],
                    out_channels=self.dims[i],
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    activation='gelu',
                ),
                PatchResizing2d(
                    in_channels=self.dims[i],
                    out_channels=self.dims[i+1],
                    down=True,
                ) if i < self.n_stages - 1 else nn.Identity(),
            ]))
            if i < self.n_stages - 1:
                res = res // 2

    def forward(self, X: Tensor):
        X = self.first_conv(X)

        for blk in self.conv_down_blocks:
            X = blk(X)

        projs, skips = [], []
        for blk, down in self.down_blocks:
            X = blk(X)
            skips.append(X)
            X = down(X)

        return X, skips


class ProjectHeads(nn.Module):
    def __init__(
            self,
            dim: int = 64,                      # same as Encoder
            dim_mults: List[int] = (1, 2, 4),   # same as Encoder
            proj_dim: int = 128,                # dimension of projected feature
            fuse: bool = True,                  # whether to concat the output of the next stage to current stage
    ):
        super().__init__()
        self.n_stages = len(dim_mults)
        self.dims = [dim * dim_mults[i] for i in range(self.n_stages)] + [0]

        # ================== Projection Heads ================== #
        self.proj_heads = nn.ModuleList([])
        for i in range(self.n_stages):
            in_dim = (self.dims[i] + self.dims[i+1] // 4) if fuse else self.dims[i]
            self.proj_heads.append(nn.Sequential(
                nn.Conv2d(in_dim, proj_dim * 2, kernel_size=1, stride=1),
                nn.GELU(),
                nn.Conv2d(proj_dim * 2, proj_dim, kernel_size=1, stride=1),
            ))

        # ================== Linear Mappings ================== #
        self.fuse = None
        if fuse is True:
            self.fuse = nn.ModuleList([])
            for i in range(1, self.n_stages):
                self.fuse.append(nn.Sequential(
                    nn.Conv2d(self.dims[i], self.dims[i] // 4, kernel_size=1),
                    nn.UpsamplingNearest2d(scale_factor=2),
                ))

    def forward(self, features: List[Tensor]):
        projs = []
        for i in range(self.n_stages-1, -1, -1):
            if self.fuse is not None and i < self.n_stages - 1:
                concatX = torch.cat((features[i], self.fuse[i](features[i+1])), dim=1)
            else:
                concatX = features[i]
            projX = self.proj_heads[i](concatX)
            projX = F.normalize(projX, dim=1)
            projs.append(projX)
        projs = list(reversed(projs))
        return projs


class Bottleneck(nn.Module):
    def __init__(
            self,
            dim: int = 64,
            dim_mults: List[int] = (1, 2, 4),
    ):
        super().__init__()
        self.bottleneck = ResBlock(
            in_channels=dim * dim_mults[-1],
            out_channels=dim * dim_mults[-1],
            kernel_size=3,
            stride=1,
            padding=1,
            activation='gelu',
        )

    def forward(self, X: Tensor):
        return self.bottleneck(X)


class Decoder(nn.Module):
    def __init__(
            self,
            img_size: int = 256,                # size of input image.
            dim: int = 64,                      # channels after the first convolution.
            n_conv_stages: int = 0,             # number of convolution stages.
                                                # The input will be downsampled by 2 in each stage.
            dim_mults: List[int] = (1, 2, 4),   # a list of channel multiples in transformer stages.
                                                # The length is the number of transformer stages.
            depths: List[int] = (6, 4, 2),      # number of blocks in each transformer stage.
                                                # The length should be the same as dim_mults.
    ):
        super().__init__()
        assert len(dim_mults) == len(depths)
        self.n_stages = len(dim_mults)
        self.dims = [dim * dim_mults[i] for i in range(self.n_stages)]
        self.n_heads = dim_mults
        res = img_size // (2 ** n_conv_stages)

        # transformer stages
        self.up_blocks = nn.ModuleList([])
        for i in range(self.n_stages):
            self.up_blocks.append(nn.ModuleList([
                StackedResBlock(
                    depth=depths[i],
                    in_channels=self.dims[i] * 2,
                    out_channels=self.dims[i] * 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    activation='gelu',
                    partial=True,
                ),
                PatchResizing2d(
                    in_channels=self.dims[i] * 2,
                    out_channels=self.dims[i-1],
                    up=True,
                ) if i > 0 else nn.Identity(),
            ]))
            res = res // 2

        # convolution stages
        self.conv_up_blocks = nn.ModuleList([])
        for i in range(n_conv_stages):
            self.conv_up_blocks.append(nn.Sequential(
                Upsample(dim * 2, dim * 2),
                nn.GELU(),
            ))

        # last convolution
        self.last_conv = ConvNormAct(dim * 2, 3, kernel_size=1, stride=1, padding=0, activation='tanh')

    def forward(self, X: Tensor, skips: List[Tensor], masks: List[Tensor]):
        for (blk, up), skip, mask in zip(reversed(self.up_blocks), reversed(skips), reversed(masks)):
            X, mask = blk(torch.cat((X, skip), dim=1), mask.float())
            X = up(X)
        for blk in self.conv_up_blocks:
            X = blk(X)
        X = self.last_conv(X)
        return X


class Classifier(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.cls = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, X: Tensor):
        return self.cls(X)


class RefineNet(nn.Module):
    def __init__(
            self,
            dim: int = 64,
            dim_mults: List[int] = (1, 2, 4, 8, 8),
    ):
        super().__init__()
        n_stages = len(dim_mults)
        dims = [dim * dim_mults[i] for i in range(n_stages)]

        self.refine_first = GatedConv2d(3 + 1, dims[0], 5, stride=1, padding=2, activation='gelu')
        self.refine_encoder = nn.ModuleList([])
        for i in range(n_stages):
            self.refine_encoder.append(
                nn.ModuleList([
                    ResBlock(dims[i], dims[i], 3, stride=1, padding=1, activation='gelu', gated=True),
                    Downsample(dims[i], dims[i+1]) if i < n_stages - 1 else nn.Identity(),
                ])
            )
        self.refine_bottleneck = ResBlock(dims[-1], dims[-1], 3, stride=1, padding=1, activation='gelu', gated=True)
        self.refine_decoder = nn.ModuleList([])
        for i in range(n_stages-1, -1, -1):
            self.refine_decoder.append(
                nn.ModuleList([
                    ResBlock(dims[i] * 2, dims[i], 3, stride=1, padding=1, activation='gelu', gated=True),
                    Upsample(dims[i], dims[i-1]) if i > 0 else nn.Identity(),
                ])
            )
        self.refine_last = ConvNormAct(dims[0], 3, 1, stride=1, padding=0, activation='tanh')

    def forward(self, X: Tensor, mask: Tensor):
        skips = []
        X = self.refine_first(torch.cat((X, mask.float()), dim=1))
        for blk, down in self.refine_encoder:
            X = blk(X)
            skips.append(X)
            X = down(X)
        X = self.refine_bottleneck(X)
        for blk, up in self.refine_decoder:
            X = blk(torch.cat((X, skips.pop()), dim=1))
            X = up(X)
        X = self.refine_last(X)
        return X


# ==================================================================================================================== #


class InpaintNet(nn.Module):
    """ Wraps Encoder, ProjectHeads, Classifier, Bottleneck and Decoder """
    def __init__(
            self,
            img_size: int = 256,                            # encoder & decoder
            dim: int = 64,                                  # encoder & decoder
            n_conv_stages: int = 0,                         # encoder & decoder
            dim_mults: List[int] = (1, 2, 4),               # encoder & decoder
            proj_dim: int = 128,                            # project heads & classifier
            fuse: bool = True,                              # project heads
            encoder_kernel_size: int = 3,                   # encoder
            encoder_depths: List[int] = (6, 4, 2),          # encoder
            decoder_depths: List[int] = (2, 2, 2),          # decoder

            conf_threshs: List[float] = (1.0, 0.95, 0.95),  # kmeans
            temperature: float = 0.1,                       # kmeans
            kmeans_n_iters: int = 10,                       # kmeans
            kmeans_repeat: int = 3,                         # kmeans

            gt_mask_to_decoder: bool = False,               # control flow
            dual_inpainting: bool = False,                  # control flow
    ):
        super().__init__()
        assert len(dim_mults) == len(encoder_depths) == len(decoder_depths)
        self.img_size = img_size
        self.n_conv_stages = n_conv_stages
        self.n_stages = len(dim_mults)
        # kmeans params
        self.conf_threshs = conf_threshs
        self.temperature = temperature
        self.kmeans_n_iters = kmeans_n_iters
        self.kmeans_repeat = kmeans_repeat
        # control flow
        self.gt_mask_to_decoder = gt_mask_to_decoder
        self.dual_inpainting = dual_inpainting

        self.encoder = Encoder(
            img_size=img_size,
            dim=dim,
            n_conv_stages=n_conv_stages,
            dim_mults=dim_mults,
            depths=encoder_depths,
            kernel_size=encoder_kernel_size,
        )
        self.project_heads = ProjectHeads(
            dim=dim,
            dim_mults=dim_mults,
            proj_dim=proj_dim,
            fuse=fuse,
        )
        self.bottleneck = Bottleneck(
            dim=dim,
            dim_mults=dim_mults,
        )
        self.decoder = Decoder(
            img_size=img_size,
            dim=dim,
            n_conv_stages=n_conv_stages,
            dim_mults=dim_mults,
            depths=decoder_depths,
        )

    def forward(self, X: Tensor, gt_mask: Tensor = None, classifier: nn.Module = None):
        """
        Args:
            X (Tensor): input corrupted images
            gt_mask (Tensor): ground-truth masks, should not be given during inference
            classifier (nn.Module): Classifier
        """
        # encoder
        X, skips = self.encoder(X)

        # project heads
        projs = self.project_heads(skips)

        # kmeans
        conf_mask_hier = confidence_mask_hierarchy(
            projs=[p.detach() for p in projs],
            gt_mask=gt_mask.float() if gt_mask is not None else None,
            conf_threshs=self.conf_threshs,
            temperature=self.temperature,
            kmeans_n_iters=self.kmeans_n_iters,
            kmeans_repeat=self.kmeans_repeat,
            classifier=classifier,
        )
        pred_masks = conf_mask_hier['pred_masks']

        # bottleneck
        X = self.bottleneck(X)

        # decoder
        if self.gt_mask_to_decoder:
            assert gt_mask is not None
            masks_to_decoder = [
                F.interpolate(gt_mask.float(), size=self.img_size // (2 ** i))
                for i in range(self.n_conv_stages, self.n_conv_stages + self.n_stages)
            ]
        else:
            masks_to_decoder = pred_masks
        out = self.decoder(X, skips, masks_to_decoder)

        if not self.dual_inpainting:
            return out, projs, conf_mask_hier

        # dual inpainting (bidirectional inpainting)
        if self.gt_mask_to_decoder:
            assert gt_mask is not None
            masks_to_decoder = [
                (1. - F.interpolate(gt_mask.float(), size=self.img_size // (2 ** i)))  # reverse
                for i in range(self.n_conv_stages, self.n_conv_stages + self.n_stages)
            ]
        else:
            masks_to_decoder = [~m for m in pred_masks]  # reverse
        out2 = self.decoder(X, skips, masks_to_decoder)
        return (out, out2), projs, conf_mask_hier

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        self.encoder.load_state_dict(state_dict['encoder'], strict=strict)
        self.project_heads.load_state_dict(state_dict['project_heads'], strict=strict)
        self.bottleneck.load_state_dict(state_dict['bottleneck'], strict=strict)
        self.decoder.load_state_dict(state_dict['decoder'], strict=strict)

    def my_state_dict(self):
        return dict(
            encoder=self.encoder.state_dict(),
            project_heads=self.project_heads.state_dict(),
            bottleneck=self.bottleneck.state_dict(),
            decoder=self.decoder.state_dict(),
        )


if __name__ == '__main__':
    inpaintnet = InpaintNet(
        img_size=256,
        dim=64,
        n_conv_stages=0,
        dim_mults=[1, 2],
        proj_dim=64,
        fuse=True,
        encoder_kernel_size=3,
        encoder_depths=[10, 5],
        decoder_depths=[2, 2],
    )
    dummy_input = torch.randn(10, 3, 256, 256)
    inpaintnet(dummy_input)
    print(sum(p.numel() for p in inpaintnet.parameters()))
    print(sum(p.numel() for p in inpaintnet.encoder.parameters()))
    print(sum(p.numel() for p in inpaintnet.project_heads.parameters()))
    print(sum(p.numel() for p in inpaintnet.bottleneck.parameters()))
    print(sum(p.numel() for p in inpaintnet.decoder.parameters()))
