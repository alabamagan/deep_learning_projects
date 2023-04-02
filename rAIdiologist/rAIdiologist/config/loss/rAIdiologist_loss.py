import torch
import torch.nn as nn

__all__ = ['ConfidenceBCELoss']

class ConfidenceBCELoss(nn.Module):
    r"""
    .. notes::
        The loss function is defined as follows:

        .. math::
            \mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}\Bigg[\mathrm{clip}\Big(-y_i\log\sigma(x_i) -
            (1-y_i)\log(1-\sigma(x_i)), 0, 10\Big) +
            \gamma\mathrm{clip}\Big(-t_i\log\sigma(c_i) - (1-t_i)\log(1-\sigma(c_i)), 0, 10\Big)\Bigg],

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

    def forward(self,
                input: torch.DoubleTensor,
                target: torch.DoubleTensor):
        # Typical classification loss for first dimension
        loss_classification = self.base_loss.forward(input[...,0].flatten(), target.flatten()).mean()
        # loss_classification = torch.clamp(loss_classification, 0, 10).mean()

        # can the confidence score "predict right/wrong prediction"
        if input.shape[-1] >= 2:
            # If confidence is large and result is correctly predicted, adjust to lower loss, vice versa.
            # predict = torch.sigmoid(input[...,0].view_as(target)) > 0.5
            predict = input[...,1].view_as(target) > 0 # the CNN prediction should be the second element
            gt = target > 0
            correct_prediction = predict == gt

            # The third component would be the confidence, this configuration encourages the LSTM to correct the
            # the prediction of the CNN based on the features it extracted.
            loss_conf = self.conf_loss(input[..., 2].flatten(), correct_prediction.float().flatten()).mean()
            # loss_conf = torch.clamp(loss_conf, 0, 10).mean()

            return (loss_classification + loss_conf * self.conf_factor) / (1 + self.conf_factor)
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