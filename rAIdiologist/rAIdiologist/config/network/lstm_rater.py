from typing import Union

import torch
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, unpack_sequence


class LSTM_rater(nn.Module):
    r"""This LSTM rater receives inputs as a sequence of deep features extracted from each slice. This module has two
    operating mode, `stage_1` and `stage_2`. In `stage_1`, the module inspect the whole stack of features and directly
    return the output. In `stage_2`, the module will first inspect the whole stack, generating a prediction for each
    slice together with a confidence score. Then, starts at the middle slice, it will scan either upwards or downwards
    until the confidence score reaches a certain level for a successive number of times.

    This module also offers to record the predicted score, confidence score and which slice they are deduced. The
    playback are stored in the format:
        [(torch.Tensor([prediction, direction, slice_index]),  # data 1
         (torch.Tensor([prediction, direction, slice_index]),  # data 2
         ...]

    """
    def __init__(self, in_ch, embed_ch=1024, out_ch=2, record=False, iter_limit=5, dropout=0.2, bidirectional=False):
        super(LSTM_rater, self).__init__()
        # Batch size should be 1
        # self.lstm_reviewer = nn.LSTM(in_ch, embeded, batch_first=True, bias=True)
        self._out_ch = out_ch

        self.dropout = nn.Dropout(p=dropout)
        self.out_fc = nn.Sequential(
            nn.LayerNorm(embed_ch),
            nn.ReLU(inplace=True),
            nn.Linear(embed_ch, out_ch)
        )
        self.lstm = nn.LSTM(in_ch, embed_ch, bidirectional=bidirectional, num_layers=2, batch_first=True)

        # for playback
        self.play_back = []

        # other settings
        self.iter_limit = iter_limit
        self._RECORD_ON = record
        self.register_buffer('_mode', torch.IntTensor([1])) # let it save the state too
        self.register_buffer('_embed_ch', torch.IntTensor([embed_ch]))
        self.register_buffer('_bidirectional', torch.IntTensor([bidirectional]))

    @property
    def RECORD_ON(self):
        return self._RECORD_ON

    @RECORD_ON.setter
    def RECORD_ON(self, r):
        self._RECORD_ON = r

    def forward(self, *args):
        if self._mode in (1, 2, 3, 4, 5):
            return self.forward_(*args)
        else:
            raise ValueError("There are only stage `1` or `2`, when mode = [1|2], this runs in stage 1, when "
                             "mode = [3|4], this runs in stage 2.")

    def forward_(self, x: torch.Tensor, seq_length: Union[torch.Tensor, list]):
        r"""


        Args:
            x (torch.Tensor):
                Input tensor of the encoded slice, assume the input already excluded the first and the bottom slice that
                were padded after the convolution.
            seq_length (list):
                A list of number of slices in each element of the input mini-batch. E.g., if the input is the padded
                sequence with [10, 15, 13] slices, it would be padded into a a tensor of (3 x 15 x C), the seq length
                would be [8, 13, 11] excluding the top and bottom slice of each element.

        Returns:

        """
        # assert x.shape[0] == 1, f"This rater can only handle one sample at a time, got input of dimension {x.shape}."
        # required input size: (B x C x S), padding_mask: (B x S)
        num_slice = x.shape[-1]

        # LSTM (B x C x S) -> (B x S x C)
        x = x.permute(0, 2, 1)
        x = pack_padded_sequence(x, seq_length, batch_first=True, enforce_sorted=False)
        # !note that bidirectional LSTM reorders reverse direction run of `output` (`_o`) already
        _o, (_h, _c) = self.lstm(x)
        _o = unpack_sequence(_o) # convert back into tensors, _o = list of B x (S x C)

        o = self.out_fc(self.dropout(_h[-1])) # _h: (L x B x C), o: (B x C_out)

        if self._RECORD_ON:
            # d = direction, s = slice_index
            # Note that RAN_25 eats top and bot slice, so `s` starts with 1
            s = [torch.arange(oo.shape[0]).view(-1, 1) + 1 for oo in _o]
            d = [torch.zeros(oo.shape[0]).view(-1, 1) for oo in _o]
            if self._out_ch > 1:
                # This is a hack, LSTM_rater should not know how the outer modules uses its output
                sigmoid_conf =  [torch.sigmoid(self.out_fc(oo)[..., 1]).cpu().view(-1, 1) for oo in _o]
                sw_pred = [self.out_fc(oo)[..., 0].cpu().view(-1, 1) for oo in _o]
                self.play_back.extend([torch.concat(row, dim=-1) for row in list(zip(sw_pred, sigmoid_conf, s, d))])
            else:
                raise ArithmeticError("Bidirectional not implemented.")
                sw_pred = [self.out_fc(oo).cpu().view(-1, 1) for oo in _o]
                self.play_back.extend([torch.concat(row, dim=-1) for row in list(zip(sw_pred, s, d))])
            # row = torch.cat([_o, d.expand_as(_o), slice_index.expand_as(_o)], dim=-1) # concat chans
            # self.play_back.append(row)
        return o # no need to deal with up or down afterwards

    def clean_playback(self) -> None:
        self.play_back.clear()

    def get_playback(self) -> list:
        r"""Return the playback list if :attr:`_RECORD_ON` is set to ``True`` during :func:`forward`.

        Each forward() call will add one element to ``self.play_back``, the elements should all be torch tensors
        that has the structure of (B x C x 1), where B is the mini-batch size. Note that the last elements might have
        a different B.

        Returns:
            list
        """
        return self.play_back