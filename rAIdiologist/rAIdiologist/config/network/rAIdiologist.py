import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .lstm_rater import LSTM_rater
from .slicewise_ran import SlicewiseAttentionRAN, RAN_25D
from mnts.mnts_logger import MNTSLogger
import types

class rAIdiologist(nn.Module):
    r"""
    This network is a CNN-RNN that combines the SWRAN with a simple LSTM network. The purpose was to imporve the
    interpretability as the SWRAN was already pretty good reaching accuracy of 95%. This network also has a benefit
    of not limiting the number of slices viewed such that scans with a larger field of view can also fit in.
    """
    def __init__(self, out_ch=2, record=False, iter_limit=5, dropout=0.2, lstm_dropout=0.1, bidirectional=False):
        super(rAIdiologist, self).__init__()
        self._RECORD_ON = None
        self.play_back = []
        # Create inception for 2D prediction
        #!   Note that sigmoid_out should be ``False`` when the loss is already `WithLogits`.
        self.cnn = SlicewiseAttentionRAN(1, out_ch, exclude_fc=True, sigmoid_out=False, dropout=dropout,
                                         reduce_strats='max')
        self.dropout = nn.Dropout(p=dropout)

        # LSTM for
        # self.lstm_prefc = nn.Linear(2048, 512)
        self.lstm_rater = LSTM_rater(2048, out_ch=out_ch + 1, embed_ch=512, record=record,
                                     iter_limit=iter_limit, dropout=lstm_dropout, bidirectional=False)

        # Mode
        self.register_buffer('_mode', torch.IntTensor([1]))
        self.lstm_rater._mode = self._mode

        # initialization
        self.innit()
        self.RECORD_ON = record

    @property
    def RECORD_ON(self):
        return self._RECORD_ON

    @RECORD_ON.setter
    def RECORD_ON(self, r):
        self._RECORD_ON = r
        self.lstm_rater.RECORD_ON = r

    def load_pretrained_swran(self, directory: str):
        return self.cnn.load_state_dict(torch.load(directory), strict=False)

    def clean_playback(self):
        r"""Call this after each forward run to clean the playback. Otherwise, you need to keep track of the order
        of data feeding into forward function."""
        self.play_back = []
        self.lstm_rater.clean_playback()

    def get_playback(self):
        return self.play_back

    def set_mode(self, mode: int):
        r"""
        ! When CNN is not to be trained the batchnorm in it should be turn of running stats track, the dropouts should
        ! be set to 0 too.
        """
        if mode == 0:
            for p in self.parameters():
                p.requires_grad = False
            for p in self.cnn.parameters():
                p.requires_grad = True
        elif mode in (1, 3):
            # pre-trained SRAN, train stage 1 RNN
            for p in self.cnn.parameters():
                p.requires_grad = False
            for p in self.lstm_rater.parameters():
                p.requires_grad = True
        elif mode in (2, 4):
            # fix RNN train SRAN
            for p in self.cnn.parameters():
                p.requires_grad = True
            for p in self.lstm_rater.parameters():
                p.requires_grad = False
            for p in self.lstm_rater.out_fc.parameters():
                p.requires_grad = True
        elif mode == 5:
            # Everything is on
            for p in self.parameters():
                p.requires_grad = True
        elif mode == -1: # inference
            mode = 5
        else:
            raise ValueError(f"Wrong mode input: {mode}, can only be one of [0|1|2|3|4|5].")

        # pretrain mode, don't exclude top
        self.cnn.exclude_top = mode != 0

        # fill up metadata
        self._mode.fill_(mode)
        self.lstm_rater._mode.fill_(mode)

    def innit(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward_(self, x, seg = None):
        # input is x: (B x 1 x H x W x S) seg: (B x 1 x H x W x S)
        while x.dim() < 5:
            x = x.unsqueeze(0)
        if not seg is None:
            raise DeprecationWarning("Focal mode is no longer available.")

        nonzero_slices, top_slices = RAN_25D.get_nonzero_slices(x)

        x = self.cnn(x)     # Shape -> (B x 2048 x S)
        x = self.dropout(x)
        while x.dim() < 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1).contiguous() # (B x C x S) -> (B x S x C)

        # o: (B x S x out_ch)
        o = self.lstm_rater(x,
                            torch.as_tensor(top_slices).int() + 1).contiguous() # +1 because top_slice is index

        # # Loop batch
        # _o = []
        #
        # num_ch = o.shape[-1]
        #
        # for i in nonzero_slices:
        #     # output of bidirectional LSTM: (B x S x 2 x C), where for 3rd dim, 0 is forward and 1 is reverse
        #     # say input is [0, 1, 2, 3, 4], output is arranged:
        #     #   [[0, 1, 2, 3, 4],
        #     #    [4, 3, 2, 1, 0]]
        #     # So if input is padded say [0, 1, 2, 3, 4, 0, 0], output is arranged:
        #     #   [[0, 1, 2, 3, 4, 0, 0],
        #     #    [0, 0, 4, 3, 2, 1, 0]]
        #     # Therefore, for forward run, gets the element before padding, and for the reverse run, get the
        #     # last element.
        #     # !Update:
        #     # Changed to use packed sequence, such that this now both forward and backward is at `top_slices`.
        #     _o_forward  = torch.narrow(o[i, :], 0, top_slices[i], 1) # forward run is padded after top_slices
        #     _o.append(_o_forward)
        #
        if self.RECORD_ON:
            self.play_back.extend(self.lstm_rater.get_playback())
            self.lstm_rater.clean_playback()
        #
        # if len(o) > 1:
        #     o = torch.cat(_o)
        # else:
        #     o = _o[0]
        # del _o
        return o

    @staticmethod
    def generate_mask_tensor(x):
        r"""Generates :class:`torch.MaskedTensor` that masks the zero padded slice.

        Args:
            x (torch.Tensor):
                Tensor input with size :math:`(B × C × H × W × S)`. The zero padding should be done for the :math:`S`
                dimension.

        Returns:
            torch.MaskedTensor: Tensor with size :math:`(B × S)`.

        """
        nonzero_slices, _ = RAN_25D.get_nonzero_slices(x)
        mask = torch.zeros([x.shape[0], x.shape[1]], dtype=torch.BoolType)
        for i, (bot_slice, top_slice) in nonzero_slices.items():
            mask[i, bot_slice:top_slice + 1].fill_(True)
        return torch.Masked

    def forward_swran(self, *args):
        if len(args) > 0: # restrict input to only a single image
            return self.cnn.forward(args[0])
        else:
            return self.cnn.forward(*args)

    def forward(self, *args):
        if self._mode == 0:
            return self.forward_swran(*args)
        else:
            return self.forward_(*args)


class rAIdiologist_v2(rAIdiologist):
    r"""This uses RAN_25D instead of SWRAN."""
    def __init__(self, out_ch=2, record=False, iter_limit=5, dropout=0.2, lstm_dropout=0.1, bidirectional=False,
                 reduce_strats = 'max'):
        super(rAIdiologist_v2, self).__init__(out_ch = out_ch,
                                              record = record,
                                              iter_limit = iter_limit,
                                              dropout = dropout,
                                              lstm_dropout = lstm_dropout,
                                              bidirectional = bidirectional,
                                              )

        # self.lstm_rater = LSTM_rater_v2(2048, record=record, iter_limit=iter_limit, dropout=lstm_dropout)
        self.cnn = RAN_25D(1, out_ch, exclude_fc=True, sigmoid_out=False, reduce_strats='mean', dropout=dropout)

class rAIdiologist_v3(rAIdiologist):
    r"""This is a new structure which attempts to weight the cnn and rnn output."""
    def __init__(self, *args, **kwargs):
        super(rAIdiologist_v3, self).__init__(*args, **kwargs)
        self.cnn.return_top = True
        self.cnn.exclude_top = False
        self.output_bn = nn.BatchNorm1d(2)
        self.final_fc = nn.Linear(2, 1, bias=False)
        self.final_fc.weight
        self.register_parameter('sw_weight', nn.Parameter(torch.Tensor([1.]))) # is weights the output between cnn and lstm decision

    def set_mode(self, mode: int):
        r"""The slicewise weights should only changes when the mode is not 0."""
        super(LSTM_rater, self).set_mode(mode)
        if not mode == 0:
            for mod in [self.final_fc, self.output_bn]:
                for p in mod.parameters():
                    p.requires_grad = True
        else:
            for mod in [self.final_fc, self.output_bn]:
                for p in mod.parameters():
                    p.requires_grad = False

    def forward_(self, x):
        # input is x: (B x 1 x H x W x S) seg: (B x 1 x H x W x S)
        while x.dim() < 5:
            x = x.unsqueeze(0)
        nonzero_slices, top_slices = RAN_25D.get_nonzero_slices(x)

        reduced_x, x = self.cnn(x)     # Shape -> (B x 2048 x S)
        x = self.dropout(x)
        while x.dim() < 3:
            x = x.unsqueeze(0)

        if self.RECORD_ON:
            # make a copy of cnn-encoded slices if record on.
            x_play_back = copy.deepcopy(x.detach())
        x = x.permute(0, 2, 1).contiguous() # (B x C x S) -> (B x S x C)

        # o: (B x S x out_ch + 1), lstm output dense layer will also decide its confidence
        o = self.lstm_rater(x[:, 1:], # discard the first and last slice
                            torch.as_tensor(top_slices).int() - 1).contiguous() # -1 discard the last slice

        if self.RECORD_ON:
            # If record on, extract playback from lstm_rater and clean the playback for each mini-batch
            lpb = self.lstm_rater.get_playback() # lpb: (list) B x (S x 4), channels are: (sw_pred, sigmoid_conf, s, d)
            # because the x_play_back was encoded, use cnn dense output layer to decode the predictions
            xpb = [xx for xx in self.cnn.out_fc1(self.cnn.out_bn(x_play_back).permute(0, 2, 1)).cpu()] # B x (S x 1)
            for _lpb, _reduced_x in zip(lpb, reduced_x.detach().cpu()):
                _lpb[:, 1] = _lpb[:, 0] * _lpb[:, 1] + _reduced_x * (1 - _lpb[:, 1])
            lpb = [_lpb[:, 1:].view(-1, 3) for _lpb in lpb]
            self.play_back.extend([torch.concat([xx[:ll.shape[0]], ll], dim=-1) for xx, ll in zip(xpb, lpb)])
            self.lstm_rater.clean_playback()

        if self._mode >= 3:
            # t = torch.concat([o, reduced_x], dim=1).view(-1, self.lstm_rater._out_ch + 1)
            weights = torch.sigmoid(o[:, 1])
            o[:, 0] = o[:, 0].flatten() * (1 - weights) + reduced_x.flatten() * 0.5
            o = torch.concat([torch.narrow(o, 1, 0, 1),
                              reduced_x.view(-1, 1),
                              torch.narrow(o, 1, 1, 1)], dim=1)
            o = o.view(-1, 3)
        return o

    def forward_swran(self, *args):
        if len(args) > 0: # restrict input to only a single image
            x, _ = self.cnn.forward(args[0])
            return x
        else:
            x, _ = self.cnn.forward(*args)
            return x

    def set_mode(self, mode):
        r"""For v3, `exclude_fc` of `self.cnn` is always `False`."""
        super(rAIdiologist_v3, self).set_mode(mode)
        self.cnn.exclude_top = False


class rAIdiologist_v4(rAIdiologist_v3):
    def __init__(self, *args, **kwargs):
        super(rAIdiologist_v4, self).__init__(*args, **kwargs)
        out_ch = kwargs.get('out_ch', 2)
        dropout = kwargs.get('dropout', 0.2)
        self.cnn = RAN_25D(1, out_ch, exclude_fc=True, sigmoid_out=False, reduce_strats='max',
                           dropout=dropout)
        self.cnn.return_top = True
        self.cnn.exclude_top = False

class rAIdiologist_v4mean(rAIdiologist_v3):
    def __init__(self, *args, **kwargs):
        super(rAIdiologist_v4mean, self).__init__(*args, **kwargs)
        out_ch = kwargs.get('out_ch', 2)
        dropout = kwargs.get('dropout', 0.2)
        self.cnn = RAN_25D(1, out_ch, exclude_fc=True, sigmoid_out=False, reduce_strats='mean',
                           dropout=dropout)
        self.cnn.return_top = True
        self.cnn.exclude_top = False