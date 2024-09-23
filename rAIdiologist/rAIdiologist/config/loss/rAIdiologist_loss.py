import torch
import torch.nn as nn
from torch.nn.modules.module import T
from torch.nn.utils.rnn import *
from torch.utils.hooks import RemovableHandle

from pytorch_med_imaging.loss import BinaryFocalLoss
import os
import pprint
from typing import Any, Callable, Dict, Optional, Tuple, Union

__all__ = ['ConfidenceBCELoss', 'ConfidenceCELoss']

class ConfidenceBCELoss(nn.Module):
    r"""Binary cross entropy loss with confidence weighting applied.

    This assumes the prediction input has shape (B x S x C), where S is the slice dimension. For each
    slice, the loss is evaluated with increasing weights such that initially, the error tolerance is
    big and the prediction is not expected to be correct. The weights grows in a sigmoid like function
    when the prediction screens across slices, with tolerance for error reduce slice after slice.

    For each mini-batch element, the loss is given by:

    .. math::
        \text{L}(X, Y) = \frac{1}{S}\sum_s^N \text{sigmoid}(s;d) \cdot
        \text{BCE}(x_s, Y)

    where:
        - :math:`X`: The prediction for 3D image with size :math:`\mathbb{R}^{S\times C}`
        - :math:`x_s`: The prediction for :math:`s`-th slice.
        - :math:`Y`: The ground-truth for supervised learning
        - :math:`N`: The number of slices in the prediction
        - :math:`\epsilon`: The sigmoid hyperparameter
        - :math:`d`: The delay before the weight starts to rise significantly
        - :math:`S`: The sum of all weights :math:`\sum_s \text{sigmoid}(s;d)`.

    .. note::
        The confidence is now embedded as weights, which guide the loss to be small for initial
        slices, and then increases the weight for later slices. The delay of this increase in
        weight can be controlled using the parameter 'delay'

    """
    def __init__(self, *args, conf_weight=0.3, over_conf_weight=0.5, delay=5, **kwargs):
        r"""Assumes input from model have already passed through sigmoid"""
        super(ConfidenceBCELoss, self).__init__()
        self.base_loss = nn.BCEWithLogitsLoss(*args, **kwargs, reduction='none')
        # self.conf_loss = nn.BCEWithLogitsLoss(reduction='none',
        #                                       pos_weight=torch.FloatTensor([conf_pos_weight]))

        # self.register_buffer('_epsilon', torch.DoubleTensor([1E-20]))
        # Number of slices before the decision becomes important (in unit number of slices)
        # self.register_buffer('_delay', torch.DoubleTensor([delay]))
        self.register_buffer('alpha', torch.DoubleTensor([conf_weight]))
        self.register_buffer('beta', torch.DoubleTensor([over_conf_weight]))
        # self.register_buffer('_sigma', torch.DoubleTensor([2.]))

    def forward(self,
                input: PackedSequence,
                target: torch.DoubleTensor):
        # Typical classification loss for first dimension
        batch_size = len(input)
        if input.dim() == 3:
            predictions = input[:, 0]
            sig_pred = torch.sigmoid(predictions)
            confidence = input[:, 1]
            sig_conf = torch.sigmoid(confidence)

            # Confidence regularization: Encourage high confidence for correct predictions
            confidence_loss = torch.mean((1.0 - sig_conf) * torch.abs(sig_pred - target))


            loss = self.base_loss.forward(predictions, target) + self.alpha * confidence_loss

            # Penalize overconfidence on wrong predictions
            incorrect_pred_mask = (sig_pred > 0.5) ^ (target > 0.5)
            if torch.sum(incorrect_pred_mask) > 0:
                overconfidence_penalty = torch.sum(confidence[incorrect_pred_mask])
                loss = loss.mean() + self.beta * overconfidence_penalty
            else:
                loss = loss.mean()

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


class ConfidenceCELoss(nn.Module):
    def __init__(self, *args, lambda_1=0.01, lambda_2=0.3,  **kwargs):
        r"""This is essentially the CrossEntropy version of the above binary prediction loss.

        Features:
            - Regularizer term

        .. math::
            \operatorname{Loss}(X, Y) = \frac{1}{N}\sum_s^N \text{sigmoid}(s;d)

        Args:
            lambda_1 (float, Optional): Regularizer for CNN:RNN weight, encourage it to be far from 0.5
            lambda_2 (float, Optional): Regularizer for X, encourage it to be non zero.

            *args (Any): Arguments for base_loss.
            **kwargs (Any): Keyword arguments for base_loss.

        """
        super(ConfidenceCELoss, self).__init__()
        self.register_buffer('lambda_1', torch.DoubleTensor([lambda_1]))
        self.register_buffer('lambda_2', torch.DoubleTensor([lambda_2]))
        self.base_loss = nn.CrossEntropyLoss(*args, **kwargs)


    def forward(self,
                input: Tuple[torch.FloatTensor, torch.FloatTensor],
                target: torch.LongTensor) -> torch.FloatTensor:
        r"""

        .. note::
            * There are no sigmoid/softwmax needed for the prediction, but the confidence value should have be
              sigmoided within range of 0 to 1.

        Args:
            input (tuple):
                Input should be a tuple of two tensors, first on is the prediction, second is the confidence.
                The inputs should have a size :math:`(B \times C)` and :math:`(B \times 1)`, respectively.
            target (torch.LongTensor):
                Target class

        Returns:
            torch.FloatTensor: Loss
        """
        # Unpack
        try:
            # if input contains multiple elements
            model_pred, model_conf = input

            # Get baseloss
            base_loss = self.base_loss(model_pred, target)

            # calculate regularization losses
            rho = 0.2
            lam1 = self.lambda_1 * (torch.exp(-(model_conf-0.5).div(rho).pow(2))).mean() # penalize conf near 0.5
            # lam2 = self.lambda_2 * (- torch.sigmoid((model_pred.abs())) + 1).mean() # penalize pred near zero
            return base_loss + lam1
        except ValueError:
            # otherwise its just an ordinary CE loss
            return self.base_loss(input, target)

    @property
    def weight(self):
        r"""Override to return weights of base_loss"""
        return self.base_loss.weight

    @weight.setter
    def weight(self, x):
        r"""Override to set weights of base_loss"""
        self.base_loss.weight = x
