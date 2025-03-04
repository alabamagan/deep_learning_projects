import torch
import torch.nn as nn
import torch.nn.functional as F


class DualLoss(nn.Module):
    def __init__(self, cls_loss_weight: float = 0.5):
        r"""This loss function combinat"""
        super(DualLoss, self).__init__()
        self.class_loss = nn.BCELoss(reduction='mean')
        self.seg_loss = nn.BCELoss(reduction='mean')
        self.register_buffer('weight', torch.FloatTensor([cls_loss_weight])) # This weight is classification for loss
        assert self.weight <= 1, "Class loss weight must be <= 1"
        self.register_buffer('seg_loss_weight', 1 - self.weight)


    def forward(self,
                pred_cls: torch.FloatTensor,
                pred_seg: torch.FloatTensor,
                target_cls: torch.LongTensor,
                target_seg: torch.LongTensor) -> torch.FloatTensor:
        target_seg = target_seg * target_cls.view([-1, 1, 1, 1, 1]).expand_as(target_seg)
        return (self.weight * self.class_loss(pred_cls, target_cls) \
                + self.seg_loss_weight * self.seg_loss(pred_seg, target_seg))
