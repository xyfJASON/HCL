import math
from typing import List
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from models.kmeans import SoftKMeans


def init_weights(init_type=None, gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__

        if classname.find('BatchNorm') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=gain)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=gain)
            elif init_type is None:
                m.reset_parameters()
            else:
                raise ValueError(f'invalid initialization method: {init_type}.')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_func


def token2image(X):
    B, N, C = X.shape
    img_size = int(math.sqrt(N))
    assert img_size * img_size == N
    return X.permute(0, 2, 1).reshape(B, C, img_size, img_size)


def image2token(X):
    B, C, H, W = X.shape
    return X.reshape(B, C, H*W).permute(0, 2, 1)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                 norm: str = None, activation: str = None, init_type: str = 'normal', sn: bool = False):
        super().__init__()
        if sn:
            self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True))
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm is None:
            self.norm = None
        else:
            raise ValueError(f'norm {norm} is not valid.')
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f'activation {activation} is not valid.')
        self.conv.apply(init_weights(init_type))

    def forward(self, X: Tensor):
        X = self.conv(X)
        if self.norm:
            X = self.norm(X)
        if self.activation:
            X = self.activation(X)
        return X


class TransposedConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                 norm: str = None, activation: str = None, init_type: str = 'normal', sn: bool = False):
        super().__init__()
        self.conv = ConvNormAct(in_channels, out_channels, kernel_size, stride, padding, norm, activation, init_type, sn)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, X: Tensor, X_lateral: Tensor = None):
        X = self.up(X)
        if X_lateral is not None:
            X = self.conv(torch.cat([X, X_lateral], dim=1))
        else:
            X = self.conv(X)
        return X


def Upsample(in_channels: int, out_channels: int, legacy_v: int = 4):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear' if legacy_v in [3, 4] else 'nearest'),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    )


def Downsample(in_channels: int, out_channels: int):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)


class PartialConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                 norm: str = None, activation: str = None, init_type: str = 'normal'):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bias = nn.Parameter(torch.zeros((out_channels, )))
        self.mask_conv_weight = torch.ones(1, 1, kernel_size, kernel_size)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm is None:
            self.norm = None
        else:
            raise ValueError(f'norm {norm} is not valid.')
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f'activation {activation} is not valid.')
        self.conv.apply(init_weights(init_type))

    def forward(self, X: Tensor, mask: Tensor):
        """ Note that 0 in mask denote invalid pixels (holes).

        Args:
            X: Tensor[bs, C, H, W]
            mask: Tensor[bs, 1, H, W]

        """
        if mask is None:
            mask = torch.ones_like(X[:, :1, :, :])
        self.mask_conv_weight = self.mask_conv_weight.to(device=mask.device)
        with torch.no_grad():
            mask_conv = F.conv2d(mask, self.mask_conv_weight, stride=self.stride, padding=self.padding)
        invalid_pos = mask_conv == 0

        scale = self.kernel_size * self.kernel_size / (mask_conv + 1e-8)
        scale.masked_fill_(invalid_pos, 0.)  # type: ignore

        X = self.conv(X * mask)
        X = X * scale + self.bias.view(1, -1, 1, 1)
        X.masked_fill_(invalid_pos, 0.)
        if self.norm:
            X = self.norm(X)
        if self.activation:
            X = self.activation(X)

        new_mask = torch.ones_like(mask_conv)
        new_mask.masked_fill_(invalid_pos, 0.)

        return X, new_mask


class GatedConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                 norm: str = None, activation: str = None, init_type: str = 'normal'):
        super().__init__()
        self.gate = ConvNormAct(in_channels, out_channels, 3, stride=1, padding=1, norm=norm, activation='sigmoid')
        self.conv = ConvNormAct(in_channels, out_channels, kernel_size, stride, padding, norm, activation, init_type)

    def forward(self, X: Tensor):
        return self.conv(X) * self.gate(X)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                 norm: str = None, activation: str = None, init_type: str = 'normal', sn: bool = False,
                 partial: bool = False, gated: bool = False):
        super().__init__()
        assert not (partial and gated)
        self.partial = partial
        if partial:
            self.conv1 = PartialConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, norm=norm, activation=activation, init_type=init_type)
            self.conv2 = PartialConv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, norm=norm, activation=activation, init_type=init_type)
        elif gated:
            self.conv1 = GatedConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, norm=norm, activation=activation, init_type=init_type)
            self.conv2 = GatedConv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, norm=norm, activation=activation, init_type=init_type)
        else:
            self.conv1 = ConvNormAct(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, norm=norm, activation=activation, init_type=init_type, sn=sn)
            self.conv2 = ConvNormAct(out_channels, out_channels, kernel_size=1, stride=1, padding=0, norm=norm, activation=activation, init_type=init_type, sn=sn)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()
        self.apply(init_weights(init_type))

    def forward(self, X: Tensor, mask: Tensor = None):
        shortcut = self.shortcut(X)
        if not self.partial:
            X = self.conv1(X)
            X = self.conv2(X)
            return X + shortcut
        else:
            X, mask = self.conv1(X, mask)
            X, mask = self.conv2(X, mask)
            return X + shortcut, mask


class PatchResizing2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, down: bool = False, up: bool = False, partial: bool = False):
        super().__init__()
        assert not (down and up), f'down and up cannot be both True'
        Conv = PartialConv2d if partial else ConvNormAct
        if down:
            self.conv = Conv(in_channels, out_channels, kernel_size=3, stride=2, padding=1, activation='gelu')
        else:
            self.conv = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation='gelu')
        if up:
            self.up = nn.UpsamplingNearest2d(scale_factor=2)
        else:
            self.up = None

    def forward(self, x, mask=None):
        if self.up is not None:
            x = self.up(x)
            if mask is not None:
                mask = self.up(mask)
        if isinstance(self.conv, PartialConv2d):
            x, mask = self.conv(x, mask)
            return x, mask
        else:
            x = self.conv(x)
            return x


# =============================================== #
# Hierarchical Confidence Mask Detection
# =============================================== #

def upsample2x(X: Tensor):
    """ F.interpolate() can only deal with tensors of dtype float """
    B, C, H, W = X.shape
    X = X.reshape(B, C, H, W, 1, 1).repeat(1, 1, 1, 1, 2, 2)
    X = X.permute(0, 1, 2, 4, 3, 5)
    X = X.reshape(B, C, H*2, W*2)
    return X


def extract_edge(X: Tensor):
    """
    Extract the edge of a 0/1 image using morphological edge detection method (dilate & substract)

    Args:
        X: [bs, 1, H, W] or [1, H, W]
    Returns:
        Tensor[bs, 1, H, W] or [1, H, W]
    """
    X_pad = F.pad(X.float(), pad=(1, 1, 1, 1), mode='replicate')
    X_edge = torch.ne(1. - X.float(), F.max_pool2d(1. - X_pad, kernel_size=3, stride=1))
    return X_edge


def mask_indexing(X: Tensor, mask: Tensor, hard_mining: bool = False, conf: Tensor = None):
    """
    Extract values in the input tensor X indexed by a boolean tensor mask

    Args:
        X: [C, H, W]
        mask: [1, H, W], dtype=bool
        hard_mining: if True, return edge features and non-edge features sorted by conf
        conf: [1, H, W], confidence mask if hard_mining is True
    Returns:
        Tensor[N, C], extracted tensors where mask is True
    """
    if not hard_mining:
        return X[:, mask.squeeze(0)].T
    else:
        ids = torch.argsort(conf.flatten())             # [H*W]
        X = X.flatten(start_dim=1)[:, ids].T            # [H*W, C]
        mask_edge = extract_edge(mask)                  # [1, H, W]
        mask_not_edge = mask & ~mask_edge               # [1, H, W]
        mask_edge = mask_edge.flatten()[ids]            # [H*W]
        mask_not_edge = mask_not_edge.flatten()[ids]    # [H*W]
        return X[mask_edge, :], X[mask_not_edge, :]


@torch.no_grad()
def confidence_mask_single_stage(
        feature: Tensor,
        classifier: nn.Module,
        prev_centers: Tensor = None,
        acc_valid_mask: Tensor = None,
        acc_invalid_mask: Tensor = None,
        acc_conf: Tensor = None,
        gt_mask: Tensor = None,
        conf_thresh: float = 0.95,
        by_percentage: bool = True,
        lowconf_with_edge: bool = True,
        temperature: float = 0.1,
        kmeans_n_iters: int = 10,
        kmeans_repeat: int = 3,
):
    """
    Apply soft kmeans on features whose confidence are low in the previous stage, i.e. those that need to be refined.

    Args:
        feature: [bs, C, H, W]
        classifier: use classifier to determine which cluster center is the masked area
        prev_centers: [bs, 2, C], cluster center of previous stage
        acc_valid_mask: [bs, 1, H//2, W//2], accumulated valid positions from previous stage to the last stage
        acc_invalid_mask: [bs, 1, H//2, W//2], accumulated invalid positions from previous stage to the last stage
        acc_conf: [bs, 1, H//2, W//2], accumulated confidence from previous stage to the last stage
        gt_mask: [bs, 1, H', W']
        conf_thresh: threshold of high confidence
        lowconf_with_edge: whether to view the edge pixels as low confidence
        by_percentage: thresholding by percentage or by value
        temperature: kmeans, temperature param in softmax
        kmeans_n_iters: kmeans, number of iterations
        kmeans_repeat: kmeans, repeat time to avoid local optima

    Returns:
        cur_valid_mask: [bs, 1, H, W], valid positions in current stage
        cur_invalid_mask: [bs, 1, H, W], invalid positions in current stage
        cur_conf: [bs, 1, H, W], confidence score in current stage
        acc_valid_mask: [bs, 1, H, W], accumulated valid positions from current stage to the last stage
        acc_invalid_mask: [bs, 1, H, W], accumulated invalid positions from current stage to the last stage
        pred_mask: [bs, 1, H, W]
    """
    kmeans = SoftKMeans(
        n_iters=kmeans_n_iters,
        repeat=kmeans_repeat,
        temperature=temperature,
    )

    bs, C, H, W = feature.shape
    last_stage = prev_centers is None
    if last_stage:
        lowconf_mask = torch.ones_like(feature[:, :1, :, :]).bool()             # [bs, 1, H, W]
        acc_valid_mask = torch.zeros_like(feature[:, :1, :, :]).bool()          # [bs, 1, H, W]
        acc_invalid_mask = torch.zeros_like(feature[:, :1, :, :]).bool()        # [bs, 1, H, W]
        acc_conf = torch.full_like(feature[:, :1, :, :], fill_value=0.5)        # [bs, 1, H, W]
    else:
        lowconf_mask = ~(acc_valid_mask | acc_invalid_mask)
        lowconf_mask = upsample2x(lowconf_mask)                                 # [bs, 1, H, W]
        acc_valid_mask = upsample2x(acc_valid_mask)                             # [bs, 1, H, W]
        acc_invalid_mask = upsample2x(acc_invalid_mask)                         # [bs, 1, H, W]
        acc_conf = upsample2x(acc_conf)                                         # [bs, 1, H, W]
    cur_valid_mask = torch.zeros_like(feature[:, :1, :, :]).bool()              # [bs, 1, H, W]
    cur_invalid_mask = torch.zeros_like(feature[:, :1, :, :]).bool()            # [bs, 1, H, W]
    cur_conf = torch.full_like(feature[:, :1, :, :], fill_value=0.5)            # [bs, 1, H, W]
    cur_centers = torch.zeros((bs, 2, C), device=feature.device)                # [bs, 2, C]
    if gt_mask is not None:
        gt_mask = F.interpolate(gt_mask, size=(H, W))                           # [bs, 1, H, W]

    for b in range(bs):
        if (~lowconf_mask[b]).all():
            continue
        lowconf_feature = mask_indexing(feature[b], lowconf_mask[b])                                # [N, C]
        kmeans.fit(lowconf_feature)
        cur_conf[b] = kmeans.predict(feature[b].reshape(C, H*W).permute(1, 0)).reshape(1, H, W)     # [1, H, W]
        cur_centers[b] = torch.clone(kmeans.centers)                                                # [2, C]

        # Determine which cluster center is actually the masked area
        reverse = False
        if last_stage:
            # First, use gt_mask if it is provided (e.g. during training);
            # Then, use the classifier.
            if gt_mask is not None:
                if F.l1_loss(cur_conf[b], gt_mask[b]) > F.l1_loss(1. - cur_conf[b], gt_mask[b]):
                    reverse = True
            elif classifier is not None:
                scores = classifier(cur_centers[b])
                if scores[0].item() < scores[1].item():
                    reverse = True
            else:
                if torch.sum(cur_conf[b]) < torch.sum(1. - cur_conf[b]):
                    reverse = True
        else:
            # If current stage is not the last stage, refer to previous stage's cluster centers
            weights = kmeans.predict(prev_centers[b])
            if weights[0] < weights[1]:
                reverse = True
        if reverse:
            cur_conf[b] = 1. - cur_conf[b]
            cur_centers[b] = torch.stack([cur_centers[b][1, :], cur_centers[b][0, :]], dim=0)
        acc_conf[b] = torch.where(lowconf_mask[b], cur_conf[b], acc_conf[b])                        # [1, H, W]

        if by_percentage:
            cur_valid_idx = torch.topk(cur_conf[b].flatten(), k=int(torch.ge(cur_conf[b], 0.5).sum() * conf_thresh))[1]
            cur_valid_mask[b][:, torch.div(cur_valid_idx, W, rounding_mode='floor'), cur_valid_idx % W] = True
            cur_invalid_idx = torch.topk(cur_conf[b].flatten(), k=int(torch.lt(cur_conf[b], 0.5).sum() * conf_thresh), largest=False)[1]
            cur_invalid_mask[b][:, torch.div(cur_invalid_idx, W, rounding_mode='floor'), cur_invalid_idx % W] = True
        else:
            cur_valid_mask[b] = cur_conf[b] >= conf_thresh
            cur_invalid_mask[b] = cur_conf[b] <= 1. - conf_thresh
        if lowconf_with_edge:
            cur_valid_mask[b] = cur_valid_mask[b] & ~extract_edge(cur_valid_mask[b])
            cur_invalid_mask[b] = cur_invalid_mask[b] & ~extract_edge(cur_invalid_mask[b])
        cur_valid_mask[b] = cur_valid_mask[b] & lowconf_mask[b]
        cur_invalid_mask[b] = cur_invalid_mask[b] & lowconf_mask[b]
        acc_valid_mask[b] |= cur_valid_mask[b]
        acc_invalid_mask[b] |= cur_invalid_mask[b]

    pred_mask = acc_valid_mask | ((cur_conf >= 0.5) & lowconf_mask)
    return dict(
        cur_valid_mask=cur_valid_mask,
        cur_invalid_mask=cur_invalid_mask,
        cur_conf=cur_conf,
        cur_centers=cur_centers,
        acc_valid_mask=acc_valid_mask,
        acc_invalid_mask=acc_invalid_mask,
        acc_conf=acc_conf,
        pred_mask=pred_mask,
    )


@torch.no_grad()
def confidence_mask_hierarchy(
        projs: List[Tensor],
        classifier: nn.Module,
        gt_mask: Tensor = None,
        conf_threshs: List[float] = (1.0, 0.95, 0.95),
        by_percentage: bool = True,
        temperature: float = 0.1,
        kmeans_n_iters: int = 10,
        kmeans_repeat: int = 3,
):
    """ Apply soft kmeans stage-by-stage and refine the predicted mask by confidence """
    result = dict(
        valid_masks=[],
        invalid_masks=[],
        confs=[],
        centers=[],
        acc_valid_masks=[],
        acc_invalid_masks=[],
        acc_confs=[],
        pred_masks=[],
    )
    n_stages = len(projs)
    for i in range(n_stages-1, -1, -1):
        prev_center = result['centers'][0] if i < n_stages - 1 else None
        acc_valid_mask = result['acc_valid_masks'][0] if i < n_stages - 1 else None
        acc_invalid_mask = result['acc_invalid_masks'][0] if i < n_stages - 1 else None
        acc_conf = result['acc_confs'][0] if i < n_stages - 1 else None
        res = confidence_mask_single_stage(
            feature=projs[i],
            classifier=classifier,
            prev_centers=prev_center,
            acc_valid_mask=acc_valid_mask,
            acc_invalid_mask=acc_invalid_mask,
            acc_conf=acc_conf,
            gt_mask=gt_mask,
            conf_thresh=conf_threshs[i],
            by_percentage=by_percentage,
            temperature=temperature,
            kmeans_n_iters=kmeans_n_iters,
            kmeans_repeat=kmeans_repeat,
        )
        result['valid_masks'].insert(0, res['cur_valid_mask'])
        result['invalid_masks'].insert(0, res['cur_invalid_mask'])
        result['confs'].insert(0, res['cur_conf'])
        result['centers'].insert(0, res['cur_centers'])
        result['acc_valid_masks'].insert(0, res['acc_valid_mask'])
        result['acc_invalid_masks'].insert(0, res['acc_invalid_mask'])
        result['acc_confs'].insert(0, res['acc_conf'])
        result['pred_masks'].insert(0, res['pred_mask'])
    return result


def mask_prop(mask: Tensor, patch_size: int):
    assert (torch.eq(mask, 0) | torch.eq(mask, 1)).all()
    B, C, H, W = mask.shape
    mask = mask.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    mask = mask.permute(0, 1, 2, 4, 3, 5)
    mask = torch.sum(mask, dim=(-1, -2))
    return mask / (patch_size ** 2)


def _test():
    import torchvision.transforms as T
    mask = torch.ones((1, 1, 256, 256))
    mask[:, :, 32:145, 63:153] = 0
    mask[:, :, 9:15, 163:253] = 0
    mask[:, :, 9:215, 65:70] = 0

    mask2 = mask_prop(mask, patch_size=16)

    mask_pad = F.pad(mask[0], pad=(1, 1, 1, 1), mode='replicate')
    mask_edge = torch.ne(1. - mask[0].float(), F.max_pool2d(1. - mask_pad, kernel_size=3, stride=1))
    assert torch.eq(mask_edge & mask[0].bool(), mask_edge).all()
    mask_not_edge = mask[0].bool() & ~mask_edge

    T.ToPILImage()(mask[0]).show()
    T.ToPILImage()(mask2[0]).show()
    T.ToPILImage()(mask_edge.float()).show()
    T.ToPILImage()(mask_not_edge.float()).show()


if __name__ == '__main__':
    _test()
