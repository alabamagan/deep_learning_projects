import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


def _make_divisible(v: int, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    """
    Standard 3D convolution with batch norm and ReLU activation.
    """
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[nn.Module] = None
    ):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        super(ConvBNReLU, self).__init__(
            nn.Conv3d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual3D(nn.Module):
    """
    Inverted Residual block for MobileNetV2 with 3D convolutions.
    """
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[nn.Module] = None
    ):
        super(InvertedResidual3D, self).__init__()
        self.stride = stride
        assert stride in [1, 2], f"Stride must be 1 or 2, got {stride}"
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
            
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        
        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # Pointwise convolution
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        
        # Depthwise convolution
        layers.extend([
            # Depthwise convolution
            ConvBNReLU(
                hidden_dim,
                hidden_dim,
                stride=stride,
                groups=hidden_dim,
                norm_layer=norm_layer
            ),
            # Pointwise-linear convolution
            nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_3D(nn.Module):
    """
    3D MobileNetV2 implementation for medical image classification.
    
    Args:
        num_classes: Number of classification classes
        width_mult: Width multiplier - adjusts number of channels in each layer
        inverted_residual_setting: Network structure
        round_nearest: Round the number of channels in each layer to be a multiple of this number
        block: Module specifying inverted residual building block for mobilenet
        norm_layer: Module specifying the normalization layer to use
        in_channels: Number of input channels
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[nn.Module] = None,
        norm_layer: Optional[nn.Module] = None,
        in_channels: int = 1
    ):
        super(MobileNetV2_3D, self).__init__()
        
        if block is None:
            block = InvertedResidual3D
            
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
            
        input_channel = 32
        last_channel = 1280
        
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        
        # Only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )
        
        # Building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvBNReLU(in_channels, input_channel, stride=2, norm_layer=norm_layer)]
        
        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        norm_layer=norm_layer
                    )
                )
                input_channel = output_channel
        
        # Building last several layers
        features.append(
            ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer)
        )
        
        # Combine feature layers
        self.features = nn.Sequential(*features)
        
        # Building classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool3d((None, None, 1)),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        
        # For compatibility
        self.register_buffer('_mode', torch.IntTensor([0]))
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # This exists since TorchScript doesn't support hybrid scripting
        # so we need to separate the forward method
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unsqueeze during inference
        if x.dim() == 4:
            x = x.unsqueeze(0)
            
        return self._forward_impl(x)


class MobileNetV3_3D(nn.Module):
    """
    3D MobileNetV3 implementation for medical image classification.
    
    Args:
        num_classes: Number of classification classes
        width_mult: Width multiplier - adjusts number of channels in each layer
        reduced_tail: Boolean to determine if the tail of the network should be reduced
        dilated: Boolean to determine if dilation should be used
        in_channels: Number of input channels
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        width_mult: float = 1.0,
        reduced_tail: bool = False,
        dilated: bool = False,
        in_channels: int = 1
    ):
        super(MobileNetV3_3D, self).__init__()
        
        # Building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [ConvBNReLU(in_channels, input_channel, stride=2, norm_layer=nn.BatchNorm3d)]
        
        # Building inverted residual blocks
        # MobileNetV3-Large configuration
        bneck_conf = [
            # k, exp, c, se, nl, s
            [3, 16, 16, False, 'RE', 1],
            [3, 64, 24, False, 'RE', 2],
            [3, 72, 24, False, 'RE', 1],
            [5, 72, 40, True, 'RE', 2],
            [5, 120, 40, True, 'RE', 1],
            [5, 120, 40, True, 'RE', 1],
            [3, 240, 80, False, 'HS', 2],
            [3, 200, 80, False, 'HS', 1],
            [3, 184, 80, False, 'HS', 1],
            [3, 184, 80, False, 'HS', 1],
            [3, 480, 112, True, 'HS', 1],
            [3, 672, 112, True, 'HS', 1],
            [5, 672, 160, True, 'HS', 2],
            [5, 960, 160, False, 'HS', 1],
            [5, 960, 160, True, 'HS', 1],
        ]
        
        for k, exp, c, se, nl, s in bneck_conf:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_channel = _make_divisible(exp * width_mult, 8)
            layers.append(
                InvertedResidual3D(
                    input_channel,
                    output_channel,
                    s,
                    exp_channel // input_channel
                )
            )
            input_channel = output_channel
        
        # Building last several layers
        last_conv_input_channel = input_channel
        last_conv_output_channel = _make_divisible(960 * width_mult, 8)
        layers.append(
            ConvBNReLU(
                last_conv_input_channel,
                last_conv_output_channel,
                kernel_size=1,
                norm_layer=nn.BatchNorm3d
            )
        )
        
        self.features = nn.Sequential(*layers)
        
        # Building classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool3d((None, None, 1)),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(last_conv_output_channel, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        
        # For compatibility
        self.register_buffer('_mode', torch.IntTensor([0]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unsqueeze during inference
        if x.dim() == 4:
            x = x.unsqueeze(0)
            
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_mobilenet3d(model_name: str = 'mobilenet_v2', **kwargs):
    """
    Factory function to create MobileNet3D models.
    
    Args:
        model_name (str): One of 'mobilenet_v2' or 'mobilenet_v3'
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        MobileNetV2_3D or MobileNetV3_3D: The requested model
    """
    if model_name == 'mobilenet_v2':
        return MobileNetV2_3D(**kwargs)
    elif model_name == 'mobilenet_v3':
        return MobileNetV3_3D(**kwargs)
    else:
        raise ValueError(f"Unsupported MobileNet model: {model_name}. Choose from 'mobilenet_v2' or 'mobilenet_v3'.")


def get_mobilenet_v2_3d():
    """Returns a MobileNetV2-3D model with default parameters for medical imaging."""
    m = get_mobilenet3d('mobilenet_v2', in_channels=1, num_classes=1)
    m.set_mode = lambda x: 0  # Does nothing
    return m


def get_mobilenet_v3_3d():
    """Returns a MobileNetV3-3D model with default parameters for medical imaging."""
    m = get_mobilenet3d('mobilenet_v3', in_channels=1, num_classes=1)
    m.set_mode = lambda x: 0  # Does nothing
    return m