"""
Code copied from https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch
Modified by alabamagan, redistributed under the MIT License.

Summary of modifications:
    * Add docstrings and comments

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm
from torch.nn.modules.utils import _pair, _triple
from .configs import *
from torch.distributions.normal import Normal
from einops import rearrange

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size):
        super(Embeddings, self).__init__()
        self.config = config
        in_ch = config.in_ch
        down_25d = config.config_as_25d
        down_factor = config.down_factor
        patch_size = _triple(config.patch_size)

        # Calculate the number of patches
        n_patches = int((img_size[0]/2**down_factor // patch_size[0]) * (img_size[1]/2**down_factor // patch_size[1]) *
                        (img_size[2]/2**down_factor // patch_size[2] if not down_25d else img_size[2] // patch_size[2]))

        # Initialize CNN encoder
        self.hybrid_model = CNNEncoder(config, n_channels=in_ch)
        in_channels = config['encoder_channels'][-1]

        # Initialize patch embeddings with 3D convolution
        self.patch_embeddings = Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)

        # Initialize position embeddings as learnable parameters
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        # Initialize dropout layer
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        # Get CNN features
        x, features = self.hybrid_model(x)

        # Generate patch embeddings (B, hidden, n_patches^(1/2), n_patches^(1/2))
        x = self.patch_embeddings(x)

        # Flatten and transpose dimensions (B, n_patches, hidden)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        # Combine patch embeddings with position embeddings (B, n_patches, hidden)
        embeddings = x + self.position_embeddings

        # Apply dropout
        embeddings = self.dropout(embeddings)

        # Return embeddings and features
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x

        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        """Decoder block for a 3D U-Net architecture.

        This class represents a single decoder block of a 3D U-Net, which
        upsamples the input tensor and performs two 3D convolutions with ReLU
        activations. Optional skip connections can also be provided.

        Attributes:
            conv1 (Conv3dReLU):
                First 3D convolution layer followed by ReLU activation.
            conv2 (Conv3dReLU):
                Second 3D convolution layer followed by ReLU activation.
            up (nn.Upsample):
                Upsampling layer with trilinear interpolation.

        Args:
            in_channels (int):
                Number of input channels.
            out_channels (int):
                Number of output channels.
            skip_channels (int, optional):
                Number of additional channels from the skip connection. Defaults to 0.
            use_batchnorm (bool, optional):
                Whether to use batch normalization in the convolution layers. Defaults to True.
        """
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DecoderCup(nn.Module):
    def __init__(self, config, img_size):
        super().__init__()
        self.config = config
        self.down_factor = config.down_factor
        head_channels = config.conv_first_channel
        self.img_size = img_size
        self.conv_more = Conv3dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        self.patch_size = _triple(config.patch_size)
        skip_channels = self.config.skip_channels
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        l, h, w = (self.img_size[0]//2**self.down_factor//self.patch_size[0]), (self.img_size[1]//2**self.down_factor//self.patch_size[1]), (self.img_size[2]//2**self.down_factor//self.patch_size[2])
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, l, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
                #print(skip.shape)
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer

    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Down25d(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 1)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class CNNEncoder(nn.Module):
    def __init__(self, config, n_channels=2):
        """CNN Encoder module for generating hierarchical features.

        This class represents a 3D CNN encoder that generates hierarchical features
        from the input data. The encoder consists of an initial double convolution
        followed by a series of downsampling layers and additional max-pooling
        for deeper features.

        Attributes:
            n_channels (int):
                Number of input channels.
            down_num (int):
                Number of additional max-pooling layers.
            inc (DoubleConv
                ): Initial double convolution layer.
            down1 (Down):
                First downsampling layer.
            down2 (Down):
                Second downsampling layer.
            width (int):
                Width of the last encoder channel.

        Args:
            config (Config):
                Configuration object containing model parameters.
            n_channels (int, optional):
                Number of input channels. Defaults to 2.
        """
        super(CNNEncoder, self).__init__()
        down25d = config.config_as_25d
        down = Down if not down25d else Down25d

        self.n_channels = n_channels
        encoder_channels = config.encoder_channels
        self.down_num = config.down_num
        self.inc = DoubleConv(n_channels, encoder_channels[0])
        self.down1 = down(encoder_channels[0], encoder_channels[1])
        self.down2 = down(encoder_channels[1], encoder_channels[2])
        self.width = encoder_channels[-1]
    def forward(self, x):
        """Forward pass of the CNN encoder.

        Args:
            x (torch.Tensor):
                Input tensor of shape (B, C, D, H, W), where
                B is the batch size, C is the number of channels, D is the depth,
                H is the height, and W is the width.

        Returns:
            torch.Tensor:
                Output tensor of shape (B, C_out, D, H, W),
                where C_out is the number of output channels.
            List[torch.Tensor]:
                List of hierarchical features generated by
                the CNN encoder, ordered from high to low resolution. They can
                be fed to the decoder through shortcuts for img2img usage.
        """
        features = []
        x1 = self.inc(x)
        features.append(x1)
        x2 = self.down1(x1)
        features.append(x2)
        feats = self.down2(x2)
        features.append(feats)
        feats_down = feats
        for i in range(self.down_num):
            feats_down = nn.MaxPool3d(2)(feats_down)
            features.append(feats_down)
        return feats, features[::-1]


class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)

class ViTVNetImg2Img(nn.Module):
    def __init__(self, config, img_size=(64, 256, 256), int_steps=7, vis=False):
        super(ViTVNetImg2Img, self).__init__()
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config, img_size)
        self.reg_head = RegistrationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=config['n_dims'],
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(img_size)
        self.config = config
        #self.integrate = VecInt(img_size, int_steps)
    def forward(self, x):

        source = x[:,0:1,:,:]

        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        flow = self.reg_head(x)
        #flow = self.integrate(flow)
        out = self.spatial_trans(source, flow)
        return out, flow


class ViTVNetImg2Pred(nn.Module):
    def __init__(self, config, num_classes=1, img_size=(64, 256, 256), int_steps=7, vis=False):
        super(ViTVNetImg2Pred, self).__init__()
        self.transformer = Transformer(config, img_size, vis)
        self.head = nn.Linear(config.hidden_size, num_classes)
        self.config = config
        self.vis = vis
        self.num_classes = num_classes
        #self.integrate = VecInt(img_size, int_steps)
    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        x, attention, _ = self.transformer(x)  # (B, n_patch, hidden)
        x = self.head(x[:, 0])
        x = x.view(B, self.num_classes)
        if self.vis:
            return x, attention
        else:
            return x


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.

    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

CONFIGS = {
    'ViT3d-V-Net': get_3DReg_config(),
    'ViT3d-Img2Pred': get_3DImg2Pred_config()
}

if __name__ == '__main__':
    import torch
    import pytorch_model_summary
    with torch.no_grad():
        net = ViTVNetImg2Pred(CONFIGS['ViT3d-Img2Pred'], img_size=(320, 320, 32), num_classes=1)
        net = net.cuda()
        pytorch_model_summary.summary(net, torch.rand(1, 1, 320, 320, 32).cuda(),
                                      show_input=True,
                                      show_parent_layers=True,
                                      max_depth=3,
                                      print_summary=True)

