from typing import Optional, Union

import torch
from einops import rearrange, repeat
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, unpack_sequence

from pytorch_med_imaging.networks.layers import PositionalEncoding, NormLayers



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
    def __init__(
            self,
            in_ch           : int,
            embed_ch        : Optional[int]   = 1024,
            out_ch          : Optional[int]   = 2,
            record          : Optional[bool]  = False,
            iter_limit      : Optional[int]   = 5,
            dropout         : Optional[float] = 0.2,
            bidirectional   : Optional[bool]  = False,
            sum_slices      : Optional[int]   = 5,
    ) -> None:
        super(LSTM_rater, self).__init__()
        # Batch size should be 1
        # self.lstm_reviewer = nn.LSTM(in_ch, embeded, batch_first=True, bias=True)
        self._out_ch = out_ch
        self._sum_slices = sum_slices

        # define dropout
        self.dropout = nn.Dropout(p=dropout)

        # define FC
        fc_chan = embed_ch * self._sum_slices
        self.out_fc = nn.Sequential(
            nn.LayerNorm(fc_chan),
            nn.ReLU(inplace=True),
            nn.Linear(fc_chan, out_ch)
        )

        # define LSTM
        lstm_in_ch = in_ch
        self.lstm_norm = NormLayers.PaddedLayerNorm(lstm_in_ch)
        self.lstm = nn.LSTM(lstm_in_ch, embed_ch, bidirectional=bidirectional, num_layers=2, batch_first=True)

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

    def forward(self,
                x: torch.Tensor,
                seq_length: Union[torch.Tensor, list]):
        r"""Note that layer


        Args:
            x (torch.Tensor):
                Input tensor of the encoded slice, assume the input already excluded the first and the bottom slice that
                were padded after the convolution. Dimension should be (B x S x C).
            seq_length (list):
                A list of number of slices in each element of the input mini-batch. E.g., if the input is the padded
                sequence with [10, 15, 13] slices, it would be padded into a a tensor of (3 x 15 x C), the seq length
                would be [8, 13, 11] excluding the top and bottom slice of each element.
            cnn_pred (torch.FloatTensor, Optinal):
                This is the prediction of the CNN. If this is provided, the CNN value will be attached to the input
                of x along the channel axis

        Returns:

        """
        # required input size: (B x S x C)
        num_slice = x.shape[1]

        # Convert seq_length to tensor if it isn't
        if not isinstance(seq_length, torch.Tensor):
            seq_length = torch.Tensor(seq_length).int()

        # norm expects (B x S x C)
        x = self.lstm_norm(x, seq_length=seq_length)
        x = pack_padded_sequence(x, seq_length, batch_first=True, enforce_sorted=False)
        # !note that bidirectional LSTM reorders reverse direction run of `output` (`_o`) already
        _o, (_h, _c) = self.lstm(x)
        unpacked_o = unpack_sequence(_o) # convert back into tensors, _o = list of B x (S x C)
        last_slices = torch.stack([o[-self._sum_slices:].flatten() for o in unpacked_o])
        o = self.out_fc(self.dropout(last_slices)) # _h: (L x B x C), o: (B x C_out)

        if self._RECORD_ON and not self.training:
            # d = direction, s = slice_index
            # Note that RAN_25 eats top and bot slice, so `s` starts with 1
            for s in unpacked_o:
                # s: (S x C) -> ss: (S - [sum_slice - 1] x sum_slice x C)
                ss = torch.stack([torch.roll(s, -j, 0) for j in range(self._sum_slices)], dim=1)
                ss = rearrange(ss, 'b s c -> b (s c)')[:len(s) - self._sum_slices]
                risk_curve = self.out_fc(ss).cpu() # risk_curve: (S - [sum_slice - 1] x out_ch)
                risk_curve_index = torch.arange(len(risk_curve)) + (self._sum_slices - 1) // 2
                lstm_pb = torch.concat([risk_curve.view(-1, self._out_ch),  # 1st ch: lstm_pred; 2nd_ch: weight
                                        risk_curve_index.view(-1, 1),       # index of representing slice
                                        torch.zeros(len(risk_curve)).view(-1, 1)], dim=1)
                self.play_back.append(lstm_pb)
            # s = [torch.arange(oo.shape[0]).view(-1, 1) + 1 for oo in unpacked_o]
            # d = [torch.zeros(oo.shape[0]).view(-1, 1) for oo in unpacked_o]
        #     if self._out_ch > 1:
        #         # This is a hack, LSTM_rater should not know how the outer modules uses its output
        #         sigmoid_conf =  [torch.sigmoid(self.out_fc(oo)[..., 1]).cpu().view(-1, 1) for oo in unpacked_o]
        #         sw_pred = [self.out_fc(oo)[..., 0].cpu().view(-1, 1) for oo in unpacked_o]
        #         self.play_back.extend([torch.concat(row, dim=-1) for row in list(zip(sw_pred, sigmoid_conf, s, d))])
        #     else:
        #         sw_pred = [self.out_fc(oo).cpu().view(-1, 1) for oo in unpacked_o]
        #         self.play_back.extend([torch.concat(row, dim=-1) for row in list(zip(sw_pred, s, d))])
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


class Transformer_rater(nn.Module):
    r"""This LSTM rater receives inputs as a sequence of deep features extracted from each slice. This module has two
    operating mode, `stage_1` and `stage_2`. In `stage_1`, the module inspect the whole stack of features and directly
    return the output. In `stage_2`, the module will first inspect the whole stack, generating a prediction for each
    slice together with a confidence score. Then, starts at the middle slice, it will scan either upwards or downwards
    until the confidence score reaches a certain level for a successive number of times.

    This module also offers to record the predicted score, confidence score and which slice they are deduced. The
    playback are stored in the format:
        [(torch.Tensor([prediction, confidence, slice_index]),  # data 1
         (torch.Tensor([prediction, confidence, slice_index]),  # data 2
         ...]

    """
    def __init__(self, in_ch, embed_ch=1024, out_ch=2, record=False, iter_limit=5, dropout=0.2):
        super(Transformer_rater, self).__init__()
        # Batch size should be 1
        # self.lstm_reviewer = nn.LSTM(in_ch, embeded, batch_first=True, bias=True)

        trans_encoder_layer = nn.TransformerEncoderLayer(d_model=in_ch, nhead=4, dim_feedforward=embed_ch, dropout=dropout)
        self.embedding = nn.TransformerEncoder(trans_encoder_layer, num_layers=6)
        self.pos_encoder = PositionalEncoding(d_model=in_ch)

        self.dropout = nn.Dropout(p=dropout)
        self.out_fc = nn.Sequential(
            nn.LayerNorm(embed_ch),
            nn.ReLU(inplace=True),
            nn.Linear(embed_ch, out_ch)
        )
        self.lstm = nn.LSTM(in_ch, embed_ch, bidirectional=True, num_layers=2, batch_first=True)

        # for playback
        self.play_back = []

        # other settings
        self.iter_limit = iter_limit
        self._RECORD_ON = record
        self.register_buffer('_mode', torch.IntTensor([1])) # let it save the state too
        self.register_buffer('_embed_ch', torch.IntTensor([embed_ch]))

        # self.init() # initialization
        # if os.getenv('CUBLAS_WORKSPACE_CONFIG') not in [":16:8", ":4096:2"]:
        #     raise AttributeError("You are invoking LSTM without properly setting the environment CUBLAS_WORKSPACE_CONFIG"
        #                          ", which might result in non-deterministic behavior of LSTM. See "
        #                          "https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM for more.")

    @property
    def RECORD_ON(self):
        return self._record_on

    @RECORD_ON.setter
    def RECORD_ON(self, r):
        self._RECORD_ON = r

    def forward(self, *args):
        if self._mode in (1, 2, 3, 4, 5):
            return self.forward_(*args)
        else:
            raise ValueError("There are only stage `1` or `2`, when mode = [1|2], this runs in stage 1, when "
                             "mode = [3|4], this runs in stage 2.")

    def forward_(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        r"""


        Args:
            padding_mask:
                Passed to `self.embedding` attribute `src_key_padding_mask`. The masked portion should be `True`.

        Returns:

        """
        # assert x.shape[0] == 1, f"This rater can only handle one sample at a time, got input of dimension {x.shape}."
        # required input size: (B x C x S), padding_mask: (B x S)
        num_slice = x.shape[-1]

        # embed with transformer encoder
        # input (B x C x S), but pos_encoding request [S, B, C]
        # embed = self.embedding(self.pos_encoder(x.permute(2, 0, 1)), src_key_padding_mask=padding_mask)
        # embeded: (S, B, C) -> (B, S, C)
        # embed = embed.permute(1, 0, 2)

        # LSTM embed: (B, S, C) -> _o: (B, S, 2 x C) !!LSTM is bidirectional!!
        # !note that LSTM reorders reverse direction run of `output` already
        _o, (_h, _c) = self.lstm(x.permute(0, 2, 1))

        play_back = []

        # _o: (B, S, C) -> _o: (B x S x 2 x C)
        bsize, ssize, _ = _o.shape
        o = _o.view(bsize, ssize, 2, -1) # separate the outputs from two direction
        o = self.out_fc(self.dropout(o))    # _o: (B x S x 2 x fc)

        if self._RECORD_ON:
            # d = direction, s = slice_indexG
            d = torch.cat([torch.zeros(ssize), torch.ones(ssize)]).view(1, -1, 1)
            slice_index = torch.cat([torch.arange(ssize)] * 2).view(1 ,-1, 1)
            _o = o.view(bsize, ssize, -1).detach().cpu()
            _o = torch.cat([_o[..., 0], _o[..., 1]], dim=1).view(bsize, -1, 1)
            row = torch.cat([_o, d.expand_as(_o), slice_index.expand_as(_o)], dim=-1) # concat chans
            self.play_back.append(row)
        return o # no need to deal with up or down afterwards

    def clean_playback(self):
        self.play_back.clear()

    def get_playback(self):
        return self.play_back