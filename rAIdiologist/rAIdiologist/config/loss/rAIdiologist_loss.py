import torch
import torch.nn as nn

__all__ = ['ConfidenceBCELoss']

class ConfidenceBCELoss(nn.Module):
    r"""
    .. notes::
        The loss function is defined as follows:


        where :math:`N` is the batch size, :math:`x_i` and :math:`y_i` are the predicted probability and the ground truth label
        for the i-th sample, :math:`c_i` is the predicted confidence score for the i-th sample, and :math:`t_i` is the target
        confidence score for the i-th sample. :math:`\sigma(\cdot)` denotes the sigmoid function, and :math:`\mathrm{clip}(\cdot)`
            is a function that clips its input values to be within a specified range.
    """
    def __init__(self, *args, conf_factor=0.3, conf_pos_weight=0.3, **kwargs):
        r"""Assumes input from model have already passed through sigmoid"""
        super(ConfidenceBCELoss, self).__init__()
        self.base_loss = nn.BCEWithLogitsLoss(*args, **kwargs, reduction='none')
        self.conf_factor = conf_factor
        # pos_weight punish wrong guess heavily while reward right guess lightly
        self.conf_loss = nn.BCEWithLogitsLoss(reduction='none',
                                              pos_weight=torch.FloatTensor([conf_pos_weight]))

        self.register_buffer('_epsilon', torch.DoubleTensor([1E-20]))
        self.register_buffer('_gamma', torch.DoubleTensor([0.05]))
        self.register_buffer('_sigma', torch.DoubleTensor([5.]))

    def forward(self,
                input: torch.DoubleTensor,
                target: torch.DoubleTensor):
        # Typical classification loss for first dimension
        loss_classification = self.base_loss.forward(input[...,0].flatten(), target.flatten()).mean()
        # loss_classification = torch.clamp(loss_classification, 0, 10).mean()

        if input.shape[-1] >= 2:
            # Regularize the weights so that it don't goes land slide to one of the sides
            weights = input[..., 2] # input is not sigmoided.
            weight_loss = self._gamma * weights.div(self._sigma).pow(2)
            return loss_classification + weight_loss.mean()
        else:
            return loss_classification

    @property
    def weight(self):
        r"""Override to return weights of base_loss"""
        return self.base_loss.weight

    @weight.setter
    def weight(self, x):
        r"""Override to set weights of base_loss"""
        self.base_loss.weight = x