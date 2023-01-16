import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional
from  pytorch_med_imaging.networks.layers import ResidualBlock3d, DoubleConv3d, Conv3d
from  pytorch_med_imaging.networks.AttentionResidual import AttentionModule_Modified
from  pytorch_med_imaging.networks.third_party_nets.spacecutter import LogisticCumulativeLink


__all__ = ['SlicewiseAttentionRAN', 'AttentionRAN_25D', 'OrdinalSlicewiseAttentionRAN', 'AttentionRAN_25D_MSE', 'RAN_25D']

class RAN_25D(nn.Module):
    r"""


    Attributes:
        in_ch (int):
            Number of in channels.
        out_ch (int):
            Number of out channels.
        first_conv_ch (int, Optional):
            Number of output channels of the first conv layer.
        save_mask (bool, Optional):
            If `True`, the attention mask of the attention modules will be saved to the CPU memory.
            Default to `False`.
        save_weight (bool, Optional):
            If `True`, the slice attention would be saved for use (CPU memory). Default to `False`.
        exclude_fc (bool, Optional):
            If `True`, the output FC layer would be excluded.

    """
    _strats_dict = {
        'max': 0,
        'min': 1,
        'mean': 2,
        'avg': 2,
    }
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 first_conv_ch: Optional[int] = 64,
                 exclude_fc: Optional[bool] = False,
                 sigmoid_out: Optional[bool] = False,
                 reduce_strats: Optional[str] = 'max',
                 dropout: Optional[float] = 0.1):

        super(RAN_25D, self).__init__()

        self.in_conv1 = Conv3d(in_ch, first_conv_ch, kern_size=[3, 3, 1], stride=[1, 1, 1], padding=[1, 1, 0])
        self.exclude_top = exclude_fc # Normally you don't have to use this.
        self.sigmoid_out = sigmoid_out

        # RAN
        self.in_conv2 = ResidualBlock3d(first_conv_ch, 256)
        self.att1 = AttentionModule_Modified(256, 256)
        self.r1 = ResidualBlock3d(256, 512, p=dropout)
        self.att2 = AttentionModule_Modified(512, 512)
        self.r2 = ResidualBlock3d(512, 1024, p=dropout)
        self.att3 = AttentionModule_Modified(1024, 1024)
        self.out_conv1 = ResidualBlock3d(1024, 2048, p=dropout)

        # Output layer
        self.out_fc1 = nn.Sequential(
            nn.Linear(2048, out_ch),
        )

        # Reduce strategy
        if reduce_strats not in self._strats_dict:
            raise AttributeError("Reduce strategy is incorrect.")
        self.register_buffer('reduce_strats', torch.Tensor([self._strats_dict[reduce_strats]]))

    @staticmethod
    def get_nonzero_slices(x: torch.Tensor):
        r"""This method computes the nonzero slices in the input. Assumes input shape follows the convention of
        :math:`(B × C × H × W × S)` and that the padding is done only for dimension :math:`S`.

        Args:
            x (torch.Tensor):
                Tensor with shape :math:`(B × C × H × W × S)`.

        Returns:
            dict: ``nonzero_slices`` - Keys are mini-batch index and values are the bottom and top slices pair
            list: ``top_slices`` - List of top index of non-zero slices.
        """
        # compute non-zero slices from seg if it is not None
        _tmp = x.detach()
        sum_slice = _tmp.sum(dim=[1, 2, 3])  # (B x S)
        # Raise error if everything is zero in any of the minibatch
        if 0 in list(sum_slice.sum(dim=[1])):
            msg = f"An item in the mini-batch is completely empty or have no segmentation:\n" \
                  f"{sum_slice.sum(dim=[1])}"
            raise ArithmeticError(msg)
        padding_mask = sum_slice != 0  # (B x S)
        where_non0 = torch.argwhere(padding_mask)
        nonzero_slices = {i.cpu().item(): (where_non0[where_non0[:, 0] == i][:, 1].min().cpu().item(),
                                           where_non0[where_non0[:, 0] == i][:, 1].max().cpu().item()) for i in
                          where_non0[:, 0]}
        # Calculate the loading bounds
        added_radius = 0
        bot_slices = [max(0, nonzero_slices[i][0] - added_radius) for i in range(len(nonzero_slices))]
        top_slices = [min(x.shape[-1], nonzero_slices[i][1] + added_radius) for i in range(len(nonzero_slices))]
        return nonzero_slices, top_slices

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
        nonzero_slices, _ = self.get_nonzero_slices(x)
        mask = torch.zeros([x.shape[0], x.shape[1]], dtype=torch.BoolType)
        for i, (bot_slice, top_slice) in nonzero_slices.items():
            mask[i, bot_slice:top_slice + 1].fill_(True)
        return torch.Masked

    def forward(self,
                x: torch.Tensor):
        r"""Expect input :math:`(B × in_{ch} × H × W × S)`, output (B × out_ch)

        Args:
            x (torch.Tensor):
                This should be a float tensor input with shape :math:`(B × in_{ch} × H × W × S)`.

        """
        B = x.shape[0]
        while x.dim() < 5:
            x = x.unsqueeze(0)
        nonzero_slice, _ = self.get_nonzero_slices(x)


        x = self.in_conv1(x)

        x = F.max_pool3d(x, [2, 2, 1], stride=[2, 2, 1])

        # Resume dimension
        x = self.in_conv2(x)

        x = self.att1(x)
        x = self.r1(x)
        x = self.att2(x)
        x = self.r2(x)
        x = self.att3(x)

        x = self.out_conv1(x)

        # order of slicewise attention and max pool makes no differences because pooling is within slice
        x = F.adaptive_max_pool3d(x, [1, 1, None]).squeeze()
        if x.dim() < 3:
            x = x.unsqueeze(0)


        if not self.exclude_top:
            x = self.out_fc1(x.permute(0, 2, 1))

            # Get best prediction across the slices
            if self.reduce_strats == 0:
                x = torch.stack([x[i, p[0]:p[1]+1].max(dim=0).values for i, p in nonzero_slice.items()])
            elif self.reduce_strats == 1:
                x = torch.stack([x[i, p[0]:p[1]+1].min(dim=0).values for i, p in nonzero_slice.items()])
            elif self.reduce_strats == 2:
                x = torch.stack([x[i, p[0]:p[1]+1].mean(dim=0) for i, p in nonzero_slice.items()])

            if x.dim() < 2:
                x.view(B, -1)
        if self.sigmoid_out:
            x = torch.sigmoid(x)
        return x

class SlicewiseAttentionRAN(RAN_25D):
    r"""


    Attributes:
        in_ch (int):
            Number of in channels.
        out_ch (int):
            Number of out channels.
        first_conv_ch (int, Optional):
            Number of output channels of the first conv layer.
        save_mask (bool, Optional):
            If `True`, the attention mask of the attention modules will be saved to the CPU memory.
            Default to `False`.
        save_weight (bool, Optional):
            If `True`, the slice attention would be saved for use (CPU memory). Default to `False`.
        exclude_fc (bool, Optional):
            If `True`, the output FC layer would be excluded.

    """
    _strats_dict = {
        'max': 0,
        'min': 1,
        'mean': 2,
        'avg': 2,
        'attn': 3,
    }
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 first_conv_ch: Optional[int] = 64,
                 save_mask: Optional[bool] = False,
                 save_weight: Optional[bool] = False,
                 exclude_fc: Optional[bool] = False,
                 sigmoid_out: Optional[bool] = False,
                 reduce_strats: Optional[str] = 'max',
                 dropout: Optional[float] = 0.1):

        super(SlicewiseAttentionRAN, self).__init__(
                 in_ch        = in_ch        ,
                 out_ch       = out_ch       ,
                 first_conv_ch= first_conv_ch,
                 exclude_fc   = exclude_fc   ,
                 sigmoid_out  = sigmoid_out  ,
                 reduce_strats= reduce_strats,
                 dropout      = dropout)

        self.save_weight=save_weight
        self.in_conv1 = Conv3d(in_ch, first_conv_ch, kern_size=[3, 3, 1], stride=[1, 1, 1], padding=[1, 1, 0])
        self.exclude_top = exclude_fc # Normally you don't have to use this.
        self.sigmoid_out = sigmoid_out

        # Slicewise attention layer
        self.in_sw = nn.Sequential(
            nn.MaxPool3d([2, 2, 1]),
            DoubleConv3d(int(first_conv_ch),
                         int(first_conv_ch * 2),
                         kern_size=[3, 3, 1], padding=0, dropout=dropout, activation='leaky_relu'),
            nn.MaxPool3d([2, 2, 1]),
            DoubleConv3d(int(first_conv_ch * 2), 1, kern_size=1, padding=0, dropout=dropout, activation='leaky_relu'),
            nn.AdaptiveAvgPool3d([1, 1, None])
        )
        self.x_w = None

    def forward(self, x):
        r"""Expect input (B x in_ch x H x W x S), output (B x out_ch)"""
        B = x.shape[0]
        while x.dim() < 5:
            x = x.unsqueeze(0)
        nonzero_slice, _ = self.get_nonzero_slices(x)

        x = self.in_conv1(x)


        # Construct slice weight
        x_w = self.in_sw(x).view(B, -1)
        if self.save_weight:
            self.x_w = x_w.data.cpu()

        # Permute the axial dimension to the last
        x = F.max_pool3d(x, [2, 2, 1], stride=[2, 2, 1])
        x = x * x_w.view([x.shape[0], 1, 1, 1, -1]).expand_as(x)

        # Resume dimension
        x = self.in_conv2(x)

        x = self.att1(x)
        x = self.r1(x)
        x = self.att2(x)
        x = self.r2(x)
        x = self.att3(x)

        x = self.out_conv1(x)
        # order of slicewise attention and max pool makes no differences because pooling is within slice
        # x = x = x * x_w.view([x.shape[0], 1, 1, 1, -1]).expand_as(x)
        x = F.adaptive_max_pool3d(x, [1, 1, None]).squeeze()
        # else:
        # x = F.adaptive_avg_pool3d(x, [1, 1, None]).squeeze()

        if x.dim() < 3:
            x = x.unsqueeze(0)

        if not self.exclude_top:
            x = self.out_fc1(x.permute(0, 2, 1))

            # Get best prediction across the slices
            if self.reduce_strats == 0:
                x = torch.stack([x[i, p[0]:p[1]+1].max(dim=0).values for i, p in nonzero_slice.items()])
            elif self.reduce_strats == 1:
                x = torch.stack([x[i, p[0]:p[1]+1].min(dim=0).values for i, p in nonzero_slice.items()])
            elif self.reduce_strats == 2:
                x = torch.stack([x[i, p[0]:p[1]+1].mean(dim=0).values for i, p in nonzero_slice.items()])

            if x.dim() < 2:
                x.view(B, -1)
        if self.sigmoid_out:
            x = torch.sigmoid(x)
        return x


    def get_mask(self):
        #[[B,H,W,D],[B,H,W,D],[B,H,W,]]
        return [r.get_mask() for r in [self.att1, self.att2, self.att3]]

    def get_slice_attention(self):
        if not self.x_w is None:
            while self.x_w.dim() < 2:
                self.x_w = self.x_w.unsqueeze(0)
            return self.x_w
        else:
            print("Attention weight was not saved!")
            return None

class AttentionRAN_25D(nn.Module):
    r"""


    Attributes:
        in_ch (int):
            Number of in channels.
        out_ch (int):
            Number of out channels.
        first_conv_ch (int, Optional):
            Number of output channels of the first conv layer.
        save_mask (bool, Optional):
            If `True`, the attention mask of the attention modules will be saved to the CPU memory.
            Default to `False`.
        save_weight (bool, Optional):
            If `True`, the slice attention would be saved for use (CPU memory). Default to `False`.

    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 first_conv_ch: Optional[int] = 64,
                 save_mask: Optional[bool] = False,
                 sigmoid_out: Optional[bool] = True):
        super(AttentionRAN_25D, self).__init__()

        self.in_conv1 = Conv3d(in_ch, first_conv_ch, kern_size=[3, 3, 1], stride=[1, 1, 1], padding=[1, 1, 0])


        # RAN
        self.in_conv2 = ResidualBlock3d(first_conv_ch, 256)
        self.att1 = AttentionModule_Modified(256, 256, save_mask=save_mask)
        self.r1 = ResidualBlock3d(256, 512, p=0.2)
        self.att2 = AttentionModule_Modified(512, 512, save_mask=save_mask)
        self.r2 = ResidualBlock3d(512, 1024, p=0.2)
        self.att3 = AttentionModule_Modified(1024, 1024, save_mask=save_mask)
        self.out_conv1 = ResidualBlock3d(1024, 2048, p=0.2)
        self.sigmoid_out = sigmoid_out

        # slice down
        self.slice_down = nn.Conv1d(out_ch, out_ch, kernel_size=5)

        # Output layer
        self.out_fc1 = nn.Sequential(
            nn.Linear(2048, 1024),
            # nn.BatchNorm1d(1024),
            # nn.LayerNorm(1024),
            nn.LeakyReLU()
        )
        self.out_fc2 = nn.Linear(1024, out_ch)

    def forward(self, x):
        while x.dim() < 5:
            x = x.unsqueeze(0)
        x = self.in_conv1(x)

        # Permute the axial dimension to the last
        x = F.max_pool3d(x, [2, 2, 1], stride=[2, 2, 1])
        x = self.in_conv2(x)

        x = self.att1(x)
        x = self.r1(x)
        x = self.att2(x)
        x = self.r2(x)
        x = self.att3(x)

        x = self.out_conv1(x)
        x = F.adaptive_max_pool3d(x, [1, 1, None]).squeeze() # (B x 2048 x 1 x 1 x S)

        if x.dim() < 3:
            x = x.unsqueeze(0)

        # Original way to reduce the slice dimension
        # x = x.max(dim=-1).values

        x = self.out_fc1(x.view([x.shape[0], x.shape[1], x.shape[-1]]).permute(0, 2, 1)) # (B x S x 2048)
        x = self.out_fc2(x).permute(0, 2, 1) # (B x out_ch x S)
        while x.shape[-1]>= 5:
            x = self.slice_down(x)
        x = x.mean(dim=-1)
        while x.dim() < 2:
            x = x.unsqueeze(0)
        if self.sigmoid_out:
            x = torch.sigmoid(x)
        return x

class OrdinalSlicewiseAttentionRAN(SlicewiseAttentionRAN):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 **kwargs):
        super(OrdinalSlicewiseAttentionRAN, self).__init__(in_ch, out_ch, **kwargs)

        # new output layer
        self.LCL = LogisticCumulativeLink(num_classes=out_ch) # note this performs its own sigmoid
        self.out_fc1 = nn.Linear(2048, 1)
        self.sigmoid_out = False

    def _batch_callback(self):
        r"""Copied and translated from
        https://github.com/EthanRosenthal/spacecutter/blob/master/spacecutter/callbacks.py"""
        margin  = 0
        min_val = -1.0e6
        cutpoints = self.LCL.cutpoints.data
        for i in range(cutpoints.shape[0] - 1):
            cutpoints[i].clamp_(min_val,
                                cutpoints[i + 1] - margin)

    def forward(self, x):
        x = super(OrdinalSlicewiseAttentionRAN, self).forward(x)
        # scale the output to adapt for the initial range of the LCL layer
        return self.LCL(x)

class OriginalAttentionRAN_25D(AttentionRAN_25D):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 **kwargs):
        super(OriginalAttentionRAN_25D, self).__init__(in_ch, out_ch, **kwargs)

        # new output layer
        self.LCL = LogisticCumulativeLink(num_classes=out_ch) # note this performs its own sigmoid
        self.out_fc2 = nn.Linear(1024, 1)
        self.slice_down = nn.Conv1d(1, 1, kernel_size=5)
        self.sigmoid_out = False

    def _batch_callback(self):
        r"""Copied and translated from
        https://github.com/EthanRosenthal/spacecutter/blob/master/spacecutter/callbacks.py"""
        margin  = 0.1
        min_val = -1.0e6
        cutpoints = self.LCL.cutpoints.data
        for i in range(cutpoints.shape[0] - 1):
            cutpoints[i].clamp_(min_val,
                                cutpoints[i + 1] - margin)

    def forward(self, x):
        return self.LCL(super(OriginalAttentionRAN_25D, self).forward(x))

class AttentionRAN_25D_MSE(AttentionRAN_25D):
    def __init__(self,
                 in_ch,
                 out_ch,
                 val_range: list,
                 **kwargs):
        super(AttentionRAN_25D_MSE, self).__init__(in_ch, out_ch, **kwargs)
        assert len(val_range) == 2
        self.register_buffer('val_range', torch.Tensor(val_range))
        self.sigmoid_out = True

    def forward(self, x):
        x = super(AttentionRAN_25D_MSE, self).forward(x)
        return x * (self.val_range[1] - self.val_range[0]) + self.val_range[0]