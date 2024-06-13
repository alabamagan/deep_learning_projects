import torch
import torch.nn as nn
from torch.nn.utils.rnn import *
from pytorch_med_imaging.loss import BinaryFocalLoss
import os
import pprint
from typing import Any

__all__ = ['ConfidenceBCELoss']

class ConfidenceBCELoss(nn.Module):
    r"""

    This assumes the prediction input has shape (B x S x C), where S is the slice dimension. For each
    slice, the loss is evaluated with increasing weights such that initially, the error tolerance is
    big and the prediction is not expected to be correct. The weights grows in a sigmoid like function
    when the prediction screens across slices, with tolerance for error reduce slice after slice.

    For each mini-batch element, the loss is given by:

    .. math::
        \text{L}(X, Y) = \frac{1}{S}\sum_s^N \text{sigmoid}(s;d) \cdot
        \text{BCE}(x_s, Y)

    where:
        :math:`X`: The prediction for 3D image with size :math:`\mathbb{R}^{S\times C}`
        :math:`x_s`: The prediction for :math:`s`-th slice.
        :math:`Y`: The ground-truth for supervised learning
        :math:`N`: The number of slices in the prediction
        :math:`\epsilon`: The sigmoid hyperparameter
        :math:`d`: The delay before the weight starts to rise significantly
        :math:`S`: The sum of all weights :math:`\sum_s \text{sigmoid}(s;d)`.

    .. note::
        The confidence is now embedded as weights, which guide the loss to be small for initial
        slices, and then increases the weight for later slices. The delay of this increase in
        weight can be controlled using the parameter 'delay'

    """
    def __init__(self, *args, conf_factor=0.3, conf_pos_weight=1, delay=5, **kwargs):
        r"""Assumes input from model have already passed through sigmoid"""
        super(ConfidenceBCELoss, self).__init__()
        self.base_loss = nn.BCEWithLogitsLoss(*args, **kwargs, reduction='none')
        # self.conf_loss = nn.BCEWithLogitsLoss(reduction='none',
        #                                       pos_weight=torch.FloatTensor([conf_pos_weight]))

        # self.register_buffer('_epsilon', torch.DoubleTensor([1E-20]))
        # Number of slices before the decision becomes important (in unit number of slices)
        self.register_buffer('_delay', torch.DoubleTensor([delay]))
        # self.register_buffer('_sigma', torch.DoubleTensor([2.]))

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
