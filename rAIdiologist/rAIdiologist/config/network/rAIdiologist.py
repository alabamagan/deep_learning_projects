import warnings
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, unpack_sequence, pad_sequence
import copy
from .lstm_rater import LSTM_rater, Transformer_rater
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

    .. deprecated::
        - Option `bidirection` is deprecated.

    """
    def __init__(self, out_ch=2, record=False, iter_limit=5, cnn_dropout=0.2, rnn_dropout=0.1, bidirectional=False, mode=0,
                 reduce_strats='max', custom_cnn=None, custom_rnn=None):
        super(rAIdiologist, self).__init__()
        self._RECORD_ON = None
        self._dropout_pdict = {}
        self._out_ch = out_ch
        self.play_back = []

        # for guild hyper param tunning
        cnn_drop = os.getenv('cnn_drop') or cnn_dropout
        lstm_drop = os.getenv('lstm_drop') or rnn_dropout

        # Create inception for 2D prediction
        #!   Note that sigmoid_out should be ``False`` when the loss is already `WithLogits`.
        self.cnn = custom_cnn or SlicewiseAttentionRAN(1,
                                                       out_ch,
                                                       exclude_fc=False,
                                                       sigmoid_out=False,
                                                       dropout=cnn_dropout,
                                                       reduce_strats=reduce_strats)
        self.dropout = nn.Dropout(p=cnn_dropout)

        # LSTM for
        self.rnn = custom_rnn or LSTM_rater(2048, out_ch=out_ch, embed_ch=128, record=record,
                                            iter_limit=iter_limit, dropout=rnn_dropout, bidirectional=False,
                                            sum_slices=3
                                            )

        # Mode
        self.register_buffer('_mode', torch.IntTensor([mode]))
        self.set_mode(mode)

        # initialization
        self.RECORD_ON = record

        # The learning rate of cnn and rnn is different, we need to give it a proper weight
        # Previously, we set rnn's weight to be 5 times the CNN,
        self.lr_weights = {
            'cnn': 1.,
            'rnn': 0.1
        }


    @property
    def RECORD_ON(self):
        return self._RECORD_ON

    @RECORD_ON.setter
    def RECORD_ON(self, r):
        self._RECORD_ON = r
        self.rnn.RECORD_ON = r

    def load_pretrained_CNN(self, directory: str):
        try:
            return self.load_state_dict(torch.load(directory), strict=True)
        except:
            return self.cnn.load_state_dict(torch.load(directory), strict=False)

    def state_dict(self, *args, **kwargs):
        r"""Override to make sure CNN is saved stand alone during pretrain phase."""
        if self._mode == 0:
            return self.cnn.state_dict(*args, **kwargs)
        else:
            return super().state_dict(*args, **kwargs)

    def clean_playback(self):
        r"""Call this after each forward run to clean the playback. Otherwise, you need to keep track of the order
        of data feeding into forward function."""
        self.play_back = []
        self.rnn.clean_playback()

    def get_playback(self):
        return self.play_back

    def set_mode(self, mode: int):
        r"""
        ! When CNN is not to be trained the batchnorm in it should be turn of running stats track, the dropouts should
        ! be set to 0 too.
        """
        if mode == 0: # pretrain
            self.requires_grad_(False)
            self.cnn.requires_grad_(True)
            self.cnn.return_top = False
            self.cnn.train()
            self.dropout.train()
            # This should not have any grad but set it anyways
            self.rnn.eval()
        elif mode in (1, 3):
            # pre-trained SRAN, train stage 1 RNN
            # Freeze CNN, train RNN
            self.requires_grad_(True)
            self.cnn.requires_grad_(False)
            self.rnn.requires_grad_(True)

            # Set CNN to eval mode to disable batchnorm learning and dropouts
            self.cnn.eval()
            self.dropout.eval()

            # Set LSTM to train mode for training
            self.rnn.train()
        elif mode in [2, 4]:
            # Freeze RNN train CNN again
            self.requires_grad_(True)
            self.cnn.requires_grad_(True)
            self.rnn.requires_grad_(False)
            self.rnn.out_fc.requires_grad_(True) # Keep output layer learnt

            # Set LSTM to eval model to disable batchnorm learning and dropouts from affecting the results
            self.rnn.eval()

            # Set CNN to train mode for training
            self.cnn.train()
            self.dropout.train()
        elif mode == 5:
            # Everything is on
            self.requires_grad_(True)

            # Set everything to train mode
            self.cnn.train()
            self.dropout.train()
            self.rnn.train()

        elif mode == -1: # inference
            mode = 5
            self.requires_grad_(False)
            self.eval() # term off all dropouts
        else:
            raise ValueError(f"Wrong mode input: {mode}, can only be one of [0|1|2|3|4|5].")

        # pretrain mode, don't exclude top
        self.cnn.exclude_top = False
        self.cnn.return_top = mode != 0

        # fill up metadata
        self._mode.fill_(mode)
        self.rnn._mode.fill_(mode)

    def train(self, mode: bool = True):
        r"""Override to ensure the train setting aligns with mode definitions. Because :func:`eval` will call this
        method, there's also an escape to ensure its not going to be a problem."""
        super(rAIdiologist, self).train(mode)
        # Reset mode only if in training loop
        if mode:
            self.set_mode(int(self._mode))
        return self

    def forward_(self, x):
        r"""This should not be ran for pre-train"""
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
        o = self.rnn(x[:, 1:],  # discard the first and last slice because CNN kernel is [3x3x3]
                     torch.as_tensor(top_slices).int() - 1) # -1 discard the last slice

        if self.RECORD_ON and not self.training:
            self.collect_playback(reduced_x, x_play_back)

        # if self.training:
        if False:
            return o.view(B, -1, self._out_ch)
        else:
            out = self.inference_forward(B, o)
            return out

    def inference_forward(self, B, o):
        r"""Forward function for inference. This stacks the output back into desired output dimension of (B x out_ch).

        TODO: A more general base class should be written for rAIdiologist setting.

        .. note::
            This inference function is written for :class:`lstm_rater`. The purpose is to collect the inference output,
            which requires the knowledge of each of the length before padding. If this is not necessary, you can
            inherit this function.

        Args:
            B (int): Batchsize
            o (torch.Tensor): Output tensor


        """
        seq_len = self.rnn.current_seq_len
        out = torch.stack([oo[-1] for oo in unpack_sequence(
            pack_padded_sequence(o, seq_len, batch_first=True, enforce_sorted=False)
        )]).view(B, self._out_ch)
        if any(torch.isclose(out, torch.zeros_like(out).float())):
            raise ValueError("Detect output is padded.")
        return out

    def collect_playback(self, reduced_x, x_play_back) -> None:
        """Collects and processes playback data for each mini-batch during training.

        This method is primarily involved in the extraction and processing of playback data from LSTM and CNN models.
        It retrieves the playback from the LSTM rater, decodes it using a CNN's dense output layer, and then concatenates
        various components including CNN predictions and risk curves along with their indices and directions. This
        processed data is appended to `self.play_back` for further analysis or training purposes.

        .. note::
            * The expected dimensions for the LSTM playback data are (B, S x 4), where B is the batch size and S x 4
              represents the sequence length multiplied by four channels: (sw_pred, sigmoid_conf, s, d).
            * The method involves CPU-intensive operations such as data transfer and manipulation, which can impact
              performance especially in larger models or datasets.
            * You can inherit this function to customize the behavior for collecting attention for visualization.

        Args:
            reduced_x (torch.Tensor):
                The tensor containing reduced features from an earlier layer of the model. It is not directly used
                in the concatenation but is utilized here for syncing with the batch processing.
            x_play_back (torch.Tensor):
                The playback tensor from the CNN, expected to have dimensions (B, C, S) where B is the batch size,
                C is the number of channels, and S is the sequence length.

        """
        # If record on, extract playback from lstm_rater and clean the playback for each mini-batch
        # Expected playback dimension: lpb: (list) B x (S x 4), channels are: (sw_pred, sigmoid_conf, s, d)
        lpb = self.rnn.get_playback()
        # because the x_play_back (B x C x S) was encoded, use cnn dense output layer to decode the predictions
        xpb = [xx for xx in self.cnn.out_fc1(x_play_back.permute(0, 2, 1)).cpu()]  # B x (S x 1)
        pb = []
        for _lpb, _xpb, _reduced_x in zip(lpb, xpb, reduced_x.detach().cpu()):
            _pb = _lpb[:, 0]  # risk curve
            # concat the CNN predictions;
            pb_row = torch.concat([
                _xpb[_lpb[:, -2].long()].view(-1, 1),  # CNN_prediction
                _pb.view(-1, 1),  # risk_curve
                _lpb[:, -2:]  # [index, direction]
            ], dim=1)
            pb.append(pb_row)
        self.play_back.extend(pb)
        self.rnn.clean_playback()

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


class rAIdiologist_Transformer(rAIdiologist):
    def __init__(self, in_ch = 1, out_ch=2, record=False, cnn_dropout=0.15, rnn_dropout=0.1, bidirectional=False, mode=0,
                 reduce_strats='max', custom_cnn=None, custom_rnn=None):
        # * Define CNN and RNN
        # TODO: Debug for the following
        # - There's a fixed prediction during inference
        # - The network does not converge at all
        cnn = SlicewiseAttentionRAN(in_ch, out_ch, dropout=cnn_dropout, return_top=True)
        rnn = Transformer_rater(in_ch=2048, dropout=rnn_dropout, out_ch=out_ch)

        # Useless variables
        # - iter_limit
        # - bidirectional
        super().__init__(out_ch, record, 5, cnn_dropout, rnn_dropout, False, mode, reduce_strats,
                         custom_cnn=cnn, custom_rnn=rnn)

    def collect_playback(self, reduced_x, x_play_back) -> None:
        # Do nothing because its not implemented
        pass

    def inference_forward(self, B, o):
        return o

    def forward_(self, x):
        r"""Override to change the default behavior, the follwoing changes were made to cater with the use of
        transformer:
        - the first and the last slice of the image is not discarded.
        - changed to use einops for more intuitive operation

        Expected input: (B x C x H x W x S)
        """
        # input is x: (B x 1 x H x W x S)
        B = x.shape[0]
        while x.dim() < 5:
            x = x.unsqueeze(0)

        # It is expected the output for the cnn part is already zero padded for inputs with different number of slices.
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
        o = self.rnn(x[:, 1:],  # discard the first and last slice because CNN kernel is [3x3x3]
                     torch.as_tensor(top_slices).int() - 1) # -1 discard the last slice

        if self.RECORD_ON and not self.training:
            self.collect_playback(reduced_x, x_play_back)

        # if self.training:
        if False:
            return o.view(B, -1, self._out_ch)
        else:
            out = self.inference_forward(B, o)
            return out

    def forward(self, *args):
        if self._mode == 0:
            return self.forward_cnn(*args)
        else:
            # * Padding the input if its a list

            return self.forward_(*args)


def create_rAIdiologist_v1():
    return rAIdiologist(1, 1, cnn_dropout=0.1, rnn_dropout=0.1)

def create_rAIdiologist_v2():
    cnn = RAN_25D(1, 1, dropout=0.35)
    return rAIdiologist(1, 1, rnn_dropout=0.1, custom_cnn=cnn)

def create_rAIdiologist_v3():
    cnn = SlicewiseAttentionRAN(1, 1, dropout=0.35)
    return rAIdiologist(1, 1, rnn_dropout=0.1, custom_cnn=cnn)

def create_rAIdiologist_v4():
    cnn = ViTVNetImg2Pred(CONFIGS['ViT3d-Img2Pred'], num_classes=1, img_size=(320, 320, 25))
    return rAIdiologist(1, 1, rnn_dropout=0.1, custom_cnn=cnn)

def create_rAIdiologist_v41():
    config = CONFIGS['ViT3d-Img2Pred']
    config.patches.grid = (4, 4, 25)
    cnn = ViTVNetImg2Pred(config, num_classes=1, img_size=(320, 320, 25))
    return rAIdiologist(1, 1, rnn_dropout=0.1, custom_cnn=cnn)

def create_rAIdiologist_v42():
    config = CONFIGS['ViT3d-Img2Pred']
    config.patches.grid = (10, 10, 25) # How many grid per axis, not pixel sizes
    cnn = ViTVNetImg2Pred(config, num_classes=1, img_size=(320, 320, 25))
    return rAIdiologist(1, 1, rnn_dropout=0.1, custom_cnn=cnn)

def create_rAIdiologist_v43():
    config = CONFIGS['ViT3d-Img2Pred']
    config.patches.grid = (10, 10, 25) # How many grid per axis, not pixel sizes
    cnn = ViTVNetImg2Pred(config, num_classes=1, img_size=(320, 320, 25))
    return rAIdiologist(1, 1, rnn_dropout=0.15, custom_cnn=cnn)

def create_rAIdiologist_v5():
    return rAIdiologist_Transformer(1, 1)

def create_rAIdiologist_v5():
    return rAIdiologist_Transformer(1, 1)

def create_rAIdiologist_v5_1():
    r"""This version does output channel = 2"""
    return rAIdiologist_Transformer(1, 2)

def create_old_rAI():
    cnn = SlicewiseAttentionRAN_old(1, 1, exclude_fc=False, return_top=False)
    return rAIdiologist(1, 1, cnn_dropout=0.25, rnn_dropout=0.2, custom_cnn=cnn)

def create_old_rAI_rmean():
    cnn = SlicewiseAttentionRAN_old(1, 1, exclude_fc=False, return_top=False, reduce_by_mean=True)
    return rAIdiologist(1, 1, cnn_dropout=0.15, rnn_dropout=0.15, custom_cnn=cnn)



