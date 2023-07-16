import torch
import torch.nn as nn
from torch.nn.utils.rnn import *
from pytorch_med_imaging.loss import BinaryFocalLoss
import os
import pprint

__all__ = ['ConfidenceBCELoss']

class ConfidenceBCELoss(nn.Module):
    r"""
    .. notes::
        The loss function is defined as follows:

    """
    def __init__(self, *args, conf_factor=0.3, conf_pos_weight=1, **kwargs):
        r"""Assumes input from model have already passed through sigmoid"""
        super(ConfidenceBCELoss, self).__init__()
        self.base_loss = nn.BCEWithLogitsLoss(*args, **kwargs, reduction='none')
        self.conf_loss = nn.BCEWithLogitsLoss(reduction='none',
                                              pos_weight=torch.FloatTensor([conf_pos_weight]))

        self.register_buffer('_epsilon', torch.DoubleTensor([1E-20]))
        # Number of slices before the decision becomes important
        self.register_buffer('_delay', torch.DoubleTensor([5.]))
        self.register_buffer('_sigma', torch.DoubleTensor([2.]))

    def forward(self,
                input: PackedSequence,
                target: torch.DoubleTensor):
        # Typical classification loss for first dimension
        batch_size = len(input)
        if input.dim() == 3:
            # Create a weight that has the same weight as slice-wise prediction
            losses = []
            for pred, tar in zip(input, target):
                # detect trailing zeros
                last = -1
                while pred[last] == 0:
                    last -= 1
                pred = pred[:last]
                loss_classification = self.base_loss.forward(pred, tar.expand_as(pred))
                weight = torch.arange(loss_classification.shape[0]).view_as(loss_classification).\
                    to(loss_classification.device)
                # Morph this weight into a sigmoid that rises to 0.5 at the `self._delay`-th slice
                weight = torch.sigmoid((weight - self._delay))

                if pred.requires_grad:
                    single_loss = (loss_classification.flatten() * weight.flatten()).sum() / weight.sum()
                else:
                    # if input does not require gradient, assume it is in validation mode. Save checkpoint with best
                    # performance and ignore regularizer
                    single_loss = loss_classification[-3]
                losses.append(single_loss)
            loss = torch.stack(losses).mean()
        else:
            loss = self.base_loss.forward(input.flatten(), target.flatten()).mean()
        return loss

    @property
    def weight(self):
        r"""Override to return weights of base_loss"""
        return self.base_loss.weight

    @weight.setter
    def weight(self, x):
        r"""Override to set weights of base_loss"""
        self.base_loss.weight = x
