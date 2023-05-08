import torch
import torch.nn as nn
import os
import pprint

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
    def __init__(self, *args, conf_factor=0.3, conf_pos_weight=1, **kwargs):
        r"""Assumes input from model have already passed through sigmoid"""
        super(ConfidenceBCELoss, self).__init__()
        self.base_loss = nn.BCEWithLogitsLoss(*args, **kwargs, reduction='none')
        self.conf_loss = nn.BCEWithLogitsLoss(reduction='none',
                                              pos_weight=torch.FloatTensor([conf_pos_weight]))

        self.register_buffer('_epsilon', torch.DoubleTensor([1E-20]))
        self.register_buffer('_gamma', torch.DoubleTensor([float(os.environ.get('loss_gamma', 0.3))]))
        self.register_buffer('_sigma', torch.DoubleTensor([float(os.environ.get('loss_sigma', 5))]))

    def forward(self,
                input: torch.DoubleTensor,
                target: torch.DoubleTensor):
        # Typical classification loss for first dimension
        overall_prediction = input[..., 0]
        loss_classification = self.base_loss.forward(overall_prediction.flatten(), target.flatten())
        # loss_classification = torch.clamp(loss_classification, 0, 10).mean()

        if input.shape[-1] >= 2:
            cnn_pred  = input[..., 1].view(-1, 1)
            weight    = input[..., 2].view(-1, 1) # this is already sigmoided
            lstm_pred = input[..., 3].view(-1, 1)

            regularizer = - self._gamma * torch.log(weight)
            overall_loss = loss_classification.flatten()
            lstm_loss = self.base_loss.forward(lstm_pred.flatten(), target.flatten())
            cnn_loss = self.base_loss.forward(cnn_pred.flatten(), target.flatten())

            torch.set_printoptions(precision=4, sci_mode=False)
            msg = pprint.pformat({
                'Overall loss': overall_loss.view(-1, 1).detach().cpu(),
                'lstm_loss': lstm_loss.view(-1, 1).detach().cpu(),
                'cnn_loss': cnn_loss.view(-1, 1).detach().cpu(),
                'reg': regularizer.view(-1, 1).detach().cpu()
            }, width=120)

            if input.requires_grad:
                loss = (overall_loss.flatten() + lstm_loss.flatten() * weight.flatten()) /2. + regularizer.flatten()
                loss = loss.mean()
            else:
                # if input does not require gradient, assume it is in validation mode. Save checkpoint with best
                # performance and ignore regularizer
                loss = overall_loss
                loss = loss.mean()
        else:
            loss = loss_classification.mean()
        return loss

    @property
    def weight(self):
        r"""Override to return weights of base_loss"""
        return self.base_loss.weight

    @weight.setter
    def weight(self, x):
        r"""Override to set weights of base_loss"""
        self.base_loss.weight = x