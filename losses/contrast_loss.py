from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from models.modules import mask_indexing, mask_prop, upsample2x


class SimplifiedCircleLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        assert reduction in ['mean', 'sum', 'none']
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor):
        """
        Args:
            pred (Tensor): [..., C]
            target (Tensor): [..., C], 1 pos, -1 neg, 0 not care
        """
        assert ((target == 0) | (target == 1) | (target == -1)).all()
        assert pred.shape == target.shape
        logits = -target * pred
        BigNegNum = torch.tensor(-1e12, device=pred.device)
        logits_neg = torch.where(target != -1, BigNegNum, logits)
        logits_pos = torch.where(target != 1, BigNegNum, logits)
        lse_neg = torch.logsumexp(logits_neg, dim=-1, keepdim=True)
        lse_pos = torch.logsumexp(logits_pos, dim=-1, keepdim=True)
        lse = lse_neg + lse_pos
        loss = torch.logsumexp(torch.cat([lse, torch.zeros_like(lse)], dim=-1), dim=-1)
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError


class ContrastLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, sample_num: int = 512, hard_mining: bool = True, hard_num: int = 512):
        super().__init__()
        self.Circle = SimplifiedCircleLoss()
        self.sample_num = sample_num
        self.temperature = temperature
        self.hard_mining = hard_mining
        self.hard_num = hard_num

    def forward(self,
                feature: Tensor,
                label: Tensor,
                prev_valid_feature: Tensor or Tuple[Tensor, Tensor],
                prev_invalid_feature: Tensor or Tuple[Tensor, Tensor]):
        """
        Args:
            feature (Tensor): [N, C]
            label (Tensor): [N], 0/1
            prev_valid_feature (Tensor): [N1, C]
            prev_invalid_feature (Tensor): [N2, C]
        """
        if feature.shape[0] == 0:  # prevent nan
            return torch.tensor(0., device=feature.device)
        assert ((label == 0) | (label == 1)).all()

        # randomly pick 512 feature vectors to prevent OOM
        if feature.shape[0] > self.sample_num:
            ids = torch.randperm(feature.shape[0])[:self.sample_num]
            feature, label = feature[ids, :], label[ids]

        # randomly pick 256 + 256 reference feature vectors to prevent OOM
        if not self.hard_mining:
            if prev_valid_feature is not None and prev_valid_feature.shape[0] > self.hard_num//2:
                ids = torch.randperm(prev_valid_feature.shape[0])[:self.hard_num//2]
                prev_valid_feature = prev_valid_feature[ids, :]
            if prev_invalid_feature is not None and prev_invalid_feature.shape[0] > self.hard_num//2:
                ids = torch.randperm(prev_invalid_feature.shape[0])[:self.hard_num//2]
                prev_invalid_feature = prev_invalid_feature[ids, :]
        else:
            if prev_valid_feature is not None:
                prev_valid_feature_edge, prev_valid_feature_other = prev_valid_feature
                if prev_valid_feature_edge.shape[0] + prev_valid_feature_other.shape[0] > self.hard_num//2:
                    n1 = min(self.hard_num//4, prev_valid_feature_edge.shape[0])
                    n2 = min(self.hard_num//2 - n1, prev_valid_feature_other.shape[0])
                    prev_valid_feature = torch.cat((prev_valid_feature_edge[:n1//2],
                                                    prev_valid_feature_edge[-(n1-n1//2):],
                                                    prev_valid_feature_other[:n2//2],
                                                    prev_valid_feature_other[-(n2-n2//2):]), dim=0)
                else:
                    prev_valid_feature = torch.cat((prev_valid_feature_edge,
                                                    prev_valid_feature_other), dim=0)
            if prev_invalid_feature is not None:
                prev_invalid_feature_edge, prev_invalid_feature_other = prev_invalid_feature
                if prev_invalid_feature_edge.shape[0] + prev_invalid_feature_other.shape[0] > self.hard_num//2:
                    n1 = min(self.hard_num//4, prev_invalid_feature_edge.shape[0])
                    n2 = min(self.hard_num//2 - n1, prev_invalid_feature_other.shape[0])
                    prev_invalid_feature = torch.cat((prev_invalid_feature_edge[:n1//2],
                                                      prev_invalid_feature_edge[-(n1-n1//2):],
                                                      prev_invalid_feature_other[:n2//2],
                                                      prev_invalid_feature_other[-(n2-n2//2):]), dim=0)
                else:
                    prev_invalid_feature = torch.cat((prev_invalid_feature_edge,
                                                      prev_invalid_feature_other), dim=0)

        logits = torch.mm(feature, feature.T)                                                                   # [N, N]
        labels = torch.eq(label[:, None], label[None, :]).float()                                               # [N, N]
        if prev_valid_feature is not None:
            valid_logits = torch.mm(feature, prev_valid_feature.T)                                              # [N, N1]
            valid_labels = torch.eq(label[:, None], torch.ones_like(prev_valid_feature[:, :1].T)).float()       # [N, N1]
            logits = torch.cat((logits, valid_logits), dim=1)
            labels = torch.cat((labels, valid_labels), dim=1)
        if prev_invalid_feature is not None:
            invalid_logits = torch.mm(feature, prev_invalid_feature.T)                                          # [N, N2]
            invalid_labels = torch.eq(label[:, None], torch.zeros_like(prev_invalid_feature[:, :1].T)).float()  # [N, N2]
            logits = torch.cat((logits, invalid_logits), dim=1)
            labels = torch.cat((labels, invalid_labels), dim=1)
        labels[labels == 0] = -1.
        loss = self.Circle(logits / self.temperature, labels)
        return loss


class HierarchicalContrastiveLoss(nn.Module):
    def __init__(self,
                 start_stage: int,
                 total_stages: int,
                 temperature: float = 0.1,
                 sample_num: int = 512,
                 valid_thresh: float = 1,
                 invalid_thresh: float = 0,
                 hard_mining: bool = True,
                 hard_num: int = 512):
        super().__init__()
        self.Contrast = ContrastLoss(temperature, sample_num, hard_mining, hard_num)
        self.start_stage = start_stage
        self.total_stages = total_stages
        self.valid_thresh = valid_thresh
        self.invalid_thresh = invalid_thresh
        self.hard_mining = hard_mining

    def forward(self,
                features: List[Tensor],
                valid_masks: List[Tensor],
                invalid_masks: List[Tensor],
                confs: List[Tensor],
                acc_valid_masks: List[Tensor],
                acc_invalid_masks: List[Tensor],
                gt_mask: Tensor):
        assert len(features) == len(valid_masks) == len(invalid_masks) == self.total_stages
        losses = []
        for st in range(self.start_stage, self.total_stages):
            bs = features[st].shape[0]
            mask_ratio = mask_prop(gt_mask, patch_size=2 ** st)
            if st + 1 == self.total_stages:
                lowconf_mask = torch.ones_like(acc_valid_masks[st]).bool()
            else:
                lowconf_mask = ~(acc_valid_masks[st+1] | acc_invalid_masks[st+1])
                lowconf_mask = upsample2x(lowconf_mask)

            loss_stage = torch.tensor(0., requires_grad=True, device=gt_mask.device)
            for b in range(bs):
                if st + 1 == self.total_stages:
                    prev_valid_feature, prev_invalid_feature = None, None
                else:
                    prev_valid_feature = mask_indexing(features[st+1][b].detach(), valid_masks[st+1][b], self.hard_mining, conf=confs[st+1][b])
                    prev_invalid_feature = mask_indexing(features[st+1][b].detach(), invalid_masks[st+1][b], self.hard_mining, conf=confs[st+1][b])

                feature = mask_indexing(features[st][b], lowconf_mask[b])
                mr = mask_indexing(mask_ratio[b], lowconf_mask[b]).squeeze(-1)
                lowconf_valid_ids = torch.ge(mr, self.valid_thresh)
                lowconf_invalid_ids = torch.le(mr, self.invalid_thresh)
                feature = torch.cat((feature[lowconf_valid_ids], feature[lowconf_invalid_ids]), dim=0)
                label = torch.cat((torch.ones((lowconf_valid_ids.sum(), ), device=feature.device),
                                   torch.zeros((lowconf_invalid_ids.sum(), ), device=feature.device)), dim=0)
                loss_stage = loss_stage + self.Contrast(feature, label, prev_valid_feature, prev_invalid_feature)

            losses.append(loss_stage / bs)
        return losses


class IndependentContrastiveLoss(nn.Module):
    """ For ablation study """
    def __init__(self, total_stages: int, temperature: float = 0.1, valid_thresh: float = 1, invalid_thresh: float = 0):
        super().__init__()
        self.Contrast = ContrastLoss(temperature, False)
        self.total_stages = total_stages
        self.valid_thresh = valid_thresh
        self.invalid_thresh = invalid_thresh

    def forward(self, features: List[Tensor], gt_mask: Tensor):
        assert len(features) == self.total_stages
        losses = []
        for st in range(self.total_stages):
            bs = features[st].shape[0]
            mask_ratio = mask_prop(gt_mask, patch_size=2 ** st)
            lowconf_mask = torch.ones_like(features[st][:, :1, :, :]).bool()

            loss_stage = torch.tensor(0., requires_grad=True, device=gt_mask.device)
            for b in range(bs):
                feature = mask_indexing(features[st][b], lowconf_mask[b])
                mr = mask_indexing(mask_ratio[b], lowconf_mask[b]).squeeze(-1)
                lowconf_valid_ids = torch.ge(mr, self.valid_thresh)
                lowconf_invalid_ids = torch.le(mr, self.invalid_thresh)
                feature = torch.cat((feature[lowconf_valid_ids], feature[lowconf_invalid_ids]), dim=0)
                label = torch.cat((torch.ones((lowconf_valid_ids.sum(), ), device=feature.device),
                                   torch.zeros((lowconf_invalid_ids.sum(), ), device=feature.device)), dim=0)
                loss_stage = loss_stage + self.Contrast(feature, label, None, None)

            losses.append(loss_stage / bs)
        return losses


def _test():
    contrast = ContrastLoss()
    feature = torch.rand(10, 4)
    lowconf_valid_ids = torch.zeros(10).bool()
    lowconf_invalid_ids = torch.zeros(10).bool()
    feature = torch.cat((feature[lowconf_valid_ids], feature[lowconf_invalid_ids]), dim=0)
    label = torch.cat((torch.ones((lowconf_valid_ids.sum(), )), torch.zeros((lowconf_invalid_ids.sum(), ))), dim=0)
    print(contrast(feature, label, None, None))


if __name__ == '__main__':
    _test()
