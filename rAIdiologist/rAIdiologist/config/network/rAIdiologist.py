import warnings
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, unpack_sequence, pad_sequence
import copy
from .lstm_rater import LSTM_rater
from .slicewise_ran import SlicewiseAttentionRAN, RAN_25D
from .old.old_swran import SlicewiseAttentionRAN_old
from .ViT3d.models import ViTVNetImg2Pred, CONFIGS
from mnts.mnts_logger import MNTSLogger
import types

class rAIdiologist(nn.Module):
    r"""
    This network is a CNN-RNN that combines the SWRAN with a simple LSTM network. The purpose was to improve the
    interpretability as the SWRAN was already pretty good reaching accuracy of 95%. This network also has a benefit
    of not limiting the number of slices viewed such that scans with a larger field of view can also fit in.
    """
    def __init__(self, out_ch=2, record=False, iter_limit=5, dropout=0.2, lstm_dropout=0.1, bidirectional=False, mode=0,
                 reduce_strats='max', custom_cnn=None, custom_rnn=None):
        super(rAIdiologist, self).__init__()
        self._RECORD_ON = None
        self._dropout_pdict = {}
        self._out_ch = out_ch
        self.play_back = []

        # for guild hyper param tunning
        cnn_drop = os.getenv('cnn_drop') or dropout
        lstm_drop = os.getenv('lstm_drop') or lstm_dropout

        # Create inception for 2D prediction
        #!   Note that sigmoid_out should be ``False`` when the loss is already `WithLogits`.
        self.cnn = custom_cnn or SlicewiseAttentionRAN(1,
                                                       out_ch,
                                                       exclude_fc=False,
                                                       sigmoid_out=False,
                                                       dropout=dropout,
                                                       reduce_strats=reduce_strats)
        self.dropout = nn.Dropout(p=dropout)

        # LSTM for
        self.lstm_rater = custom_rnn or LSTM_rater(2048, out_ch=out_ch, embed_ch=128, record=record,
                                                   iter_limit=iter_limit, dropout=lstm_dropout, bidirectional=False,
                                                   sum_slices=3
                                                   )

        # Mode
        self.register_buffer('_mode', torch.IntTensor([mode]))
        self.set_mode(mode)

        # initialization
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
            self.requires_grad_(False)
            self.cnn.requires_grad_(True)
            self.cnn.return_top = False
            # This should not have any grad but set it anyways
            self.lstm_rater.eval()
        elif mode in (1, 3):
            # pre-trained SRAN, train stage 1 RNN
            self.requires_grad_(True)
            self.cnn.requires_grad_(False)
            self.lstm_rater.requires_grad_(True)

            # Set CNN to eval mode to disable batchnorm learning and dropouts
            self.cnn.eval()
            self.dropout.eval()

            # Set LSTM to train mode for training
            self.lstm_rater.train()
        elif mode in [2, 4]:
            # fix RNN train SRAN
            self.requires_grad_(True)
            self.cnn.requires_grad_(True)
            self.lstm_rater.requires_grad_(False)
            # self.lstm_rater.out_fc.requires_grad_(False)

            # Set LSTM to eval model to disable batchnorm learning and dropouts from affecting the results
            self.lstm_rater.train()

            # Set CNN to train mode for training
            self.cnn.train()
            self.dropout.train()
        elif mode == 5:
            # Everything is on
            self.requires_grad_(True)

            # Set everything to train mode
            self.cnn.train()
            self.dropout.train()
            self.lstm_rater.train()

        elif mode == -1: # inference
            mode = 5
            self.requires_grad_(False)
        else:
            raise ValueError(f"Wrong mode input: {mode}, can only be one of [0|1|2|3|4|5].")

        # pretrain mode, don't exclude top
        self.cnn.exclude_top = False
        self.cnn.return_top = mode != 0

        # fill up metadata
        self._mode.fill_(mode)
        self.lstm_rater._mode.fill_(mode)

    def train(self, mode: bool = True):
        r"""Override to ensure the train setting aligns with mode definitions. Because :func:`eval` will call this
        method, there's also an escape to ensure its not going to be a problem."""
        super(rAIdiologist, self).train(mode)
        # Reset mode only if in training loop
        if mode:
            self.set_mode(int(self._mode))
        return self

    def forward_(self, x):
        # input is x: (B x 1 x H x W x S) seg: (B x 1 x H x W x S)
        B = x.shape[0]
        while x.dim() < 5:
            x = x.unsqueeze(0)
        nonzero_slices, top_slices = RAN_25D.get_nonzero_slices(x.detach().cpu())

        reduced_x, x = self.cnn(x)     # Shape -> x: (B x 2048 x S); reduced_x: (B x 1)
        x = self.dropout(x)
        while x.dim() < 3:
            x = x.unsqueeze(0)

        if self.RECORD_ON and not self.training:
            # make a copy of cnn-encoded slices if record on.
            x_play_back = x.detach()
        x = x.permute(0, 2, 1).contiguous() # (B x C x S) -> (B x S x C)

        # o: (B x S x [out_ch + 1]), lstm output dense layer will also decide its confidence
        o = self.lstm_rater(x[:, 1:], # discard the first and last slice
                            torch.as_tensor(top_slices).int() - 1) # -1 discard the last slice

        if self.RECORD_ON and not self.training:
            # If record on, extract playback from lstm_rater and clean the playback for each mini-batch
            lpb = self.lstm_rater.get_playback() # lpb: (list) B x (S x 4), channels are: (sw_pred, sigmoid_conf, s, d)
            # because the x_play_back (B x C x S) was encoded, use cnn dense output layer to decode the predictions
            xpb = [xx for xx in self.cnn.out_fc1(x_play_back.permute(0, 2, 1)).cpu()] # B x (S x 1)
            pb = []
            for _lpb, _xpb, _reduced_x in zip(lpb, xpb, reduced_x.detach().cpu()):
                _pb = _lpb[:,0]                 # risk curve
                # concat the CNN predictions;
                pb_row = torch.concat([
                 _xpb[_lpb[:, -2].long()].view(-1, 1),   # CNN_prediction
                 _pb.view(-1, 1),                        # risk_curve
                 _lpb[:, -2:]                            # [index, direction]
                ], dim=1)
                pb.append(pb_row)
            self.play_back.extend(pb)
            self.lstm_rater.clean_playback()

        # if self.training:
        if False:
            return o.view(B, -1, self._out_ch)
        else:
            seq_len = self.lstm_rater.current_seq_len
            out = torch.stack([oo[-1] for oo in unpack_sequence(
                pack_padded_sequence(o, seq_len, batch_first=True, enforce_sorted=False)
            )]).view(B, self._out_ch)
            if any(torch.isclose(out, torch.zeros_like(out).float())):
                raise ValueError("Detect output is padded.")
            return out

    def forward_cnn(self, *args):
        if len(args) > 0: # restrict input to only a single image
            return self.cnn.forward(args[0])
        else:
            return self.cnn.forward(*args)

    def forward(self, *args):
        if self._mode == 0:
            return self.forward_cnn(*args)
        else:
            return self.forward_(*args)


def create_rAIdiologist_v1():
    return rAIdiologist(1, 1, dropout=0.1, lstm_dropout=0.1)

def create_rAIdiologist_v2():
    cnn = RAN_25D(1, 1, dropout=0.35)
    return rAIdiologist(1, 1, lstm_dropout=0.1, custom_cnn=cnn)

def create_rAIdiologist_v3():
    cnn = SlicewiseAttentionRAN(1, 1, dropout=0.35)
    return rAIdiologist(1, 1, lstm_dropout=0.1, custom_cnn=cnn)

def create_rAIdiologist_v4():
    cnn = ViTVNetImg2Pred(CONFIGS['ViT3d-Img2Pred'], num_classes=1, img_size=(320, 320, 25))
    return rAIdiologist(1, 1, lstm_dropout=0.1, custom_cnn=cnn)

def create_rAIdiologist_v41():
    config = CONFIGS['ViT3d-Img2Pred']
    config.patches.grid = (4, 4, 25)
    cnn = ViTVNetImg2Pred(config, num_classes=1, img_size=(320, 320, 25))
    return rAIdiologist(1, 1, lstm_dropout=0.1, custom_cnn=cnn)

def create_rAIdiologist_v42():
    config = CONFIGS['ViT3d-Img2Pred']
    config.patches.grid = (20, 20, 5)
    cnn = ViTVNetImg2Pred(config, num_classes=1, img_size=(320, 320, 25))
    return rAIdiologist(1, 1, lstm_dropout=0.1, custom_cnn=cnn)

def create_old_rAI():
    cnn = SlicewiseAttentionRAN_old(1, 1, exclude_fc=False, return_top=False)
    return rAIdiologist(1, 1, dropout=0.25, lstm_dropout=0.2, custom_cnn=cnn)

def create_old_rAI_rmean():
    cnn = SlicewiseAttentionRAN_old(1, 1, exclude_fc=False, return_top=False, reduce_by_mean=True)
    return rAIdiologist(1, 1, dropout=0.15, lstm_dropout=0.15, custom_cnn=cnn)


