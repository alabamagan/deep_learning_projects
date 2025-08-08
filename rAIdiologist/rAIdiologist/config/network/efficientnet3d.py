import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional


def round_filters(filters: int, width_coefficient: float, depth_divisor: int = 8) -> int:
    """
    Calculate and round number of filters based on width coefficient and depth divisor.
    
    Args:
        filters: Number of filters
        width_coefficient: Width coefficient
        depth_divisor: Depth divisor
        
    Returns:
        Rounded number of filters
    """
    filters *= width_coefficient
    new_filters = max(depth_divisor, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats: int, depth_coefficient: float) -> int:
    """
    Calculate and round number of repeats based on depth coefficient.
    
    Args:
        repeats: Number of repeats
        depth_coefficient: Depth coefficient
        
    Returns:
        Rounded number of repeats
    """
    return int(math.ceil(depth_coefficient * repeats))


class Swish(nn.Module):
    """
    Swish activation function.
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class MBConv3D(nn.Module):
    """
    Mobile Inverted Residual Bottleneck block for 3D inputs.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float,
        drop_rate: float = 0.0
    ):
        super(MBConv3D, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.has_se = se_ratio is not None and 0 < se_ratio <= 1
        self.expand_ratio = expand_ratio
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Conv3d(in_channels, expanded_channels, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm3d(expanded_channels)
        
        # Depthwise convolution phase
        self.depthwise_conv = nn.Conv3d(
            expanded_channels, expanded_channels, kernel_size=kernel_size,
            stride=stride, padding=kernel_size // 2, groups=expanded_channels, bias=False
        )
        self.bn1 = nn.BatchNorm3d(expanded_channels)
        
        # Squeeze and Excitation phase
        if self.has_se:
            num_squeezed_channels = max(1, int(in_channels * se_ratio))
            self.se_reduce = nn.Conv3d(expanded_channels, num_squeezed_channels, kernel_size=1)
            self.se_expand = nn.Conv3d(num_squeezed_channels, expanded_channels, kernel_size=1)
        
        # Output phase
        self.project_conv = nn.Conv3d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Skip connection if in_channels == out_channels and stride == 1
        self.use_skip = (stride == 1 and in_channels == out_channels)
        
        # Activation function
        self.swish = Swish()
        
    def forward(self, x):
        identity = x
        
        # Expansion phase
        if self.expand_ratio != 1:
            x = self.expand_conv(x)
            x = self.bn0(x)
            x = self.swish(x)
        
        # Depthwise convolution phase
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.swish(x)
        
        # Squeeze and Excitation phase
        if self.has_se:
            # Squeeze
            se = F.adaptive_avg_pool3d(x, 1)
            se = self.se_reduce(se)
            se = self.swish(se)
            # Excitation
            se = self.se_expand(se)
            se = torch.sigmoid(se)
            x = x * se
        
        # Output phase
        x = self.project_conv(x)
        x = self.bn2(x)
        
        # Skip connection
        if self.use_skip:
            if self.training and self.drop_rate > 0:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            x = x + identity
        
        return x


class EfficientNet3D(nn.Module):
    """
    3D EfficientNet implementation for medical image classification.
    
    Args:
        width_coefficient: Width coefficient for network scaling
        depth_coefficient: Depth coefficient for network scaling
        dropout_rate: Dropout rate before final classifier
        num_classes: Number of classification classes
        in_channels: Number of input channels
        model_name: Name of the EfficientNet model (e.g., 'efficientnet-b0')
    """
    
    def __init__(
        self,
        width_coefficient: float = 1.0,
        depth_coefficient: float = 1.0,
        dropout_rate: float = 0.2,
        num_classes: int = 1,
        in_channels: int = 1,
        model_name: str = 'efficientnet-b0'
    ):
        super(EfficientNet3D, self).__init__()
        
        # Model configuration
        self.model_name = model_name
        
        # Base configuration for EfficientNet-B0
        base_channels = 32
        base_depths = [1, 2, 2, 3, 3, 4, 1]
        base_channels_list = [16, 24, 40, 80, 112, 192, 320]
        base_kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
        base_expand_ratios = [1, 6, 6, 6, 6, 6, 6]
        base_strides = [1, 2, 2, 2, 1, 2, 1]
        base_se_ratio = 0.25
        
        # Scale channels and depths based on coefficients
        channels_list = [round_filters(c, width_coefficient) for c in base_channels_list]
        depths = [round_repeats(d, depth_coefficient) for d in base_depths]
        
        # Stem
        out_channels = round_filters(base_channels, width_coefficient)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            Swish()
        )
        
        # Build blocks
        in_channels = out_channels
        blocks = []
        for i in range(len(depths)):
            out_channels = channels_list[i]
            kernel_size = base_kernel_sizes[i]
            expand_ratio = base_expand_ratios[i]
            stride = base_strides[i]
            se_ratio = base_se_ratio
            
            # Add blocks
            for _ in range(depths[i]):
                # Use stride 1 for all blocks except the first one in each stage
                block_stride = stride if _ == 0 else 1
                blocks.append(
                    MBConv3D(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=block_stride,
                        expand_ratio=expand_ratio,
                        se_ratio=se_ratio,
                        drop_rate=dropout_rate
                    )
                )
                in_channels = out_channels
        
        self.blocks = nn.Sequential(*blocks)
        
        # Head
        out_channels = round_filters(1280, width_coefficient)
        self.head = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            Swish()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool3d((None, None, 1)),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_channels, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # For compatibility
        self.register_buffer('_mode', torch.IntTensor([0]))
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Unsqueeze during inference
        if x.dim() == 4:
            x = x.unsqueeze(0)
            
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.classifier(x)
        return x


def get_efficientnet3d(model_name: str = 'efficientnet-b0', **kwargs):
    """
    Factory function to create EfficientNet3D models of different sizes.
    
    Args:
        model_name (str): One of 'efficientnet-b0' through 'efficientnet-b7'
        **kwargs: Additional arguments to pass to EfficientNet3D
        
    Returns:
        EfficientNet3D: The requested model
    """
    # Model configurations
    model_configs = {
        'efficientnet-b0': (1.0, 1.0, 0.2),
        'efficientnet-b1': (1.0, 1.1, 0.2),
        'efficientnet-b2': (1.1, 1.2, 0.3),
        'efficientnet-b3': (1.2, 1.4, 0.3),
        'efficientnet-b4': (1.4, 1.8, 0.4),
        'efficientnet-b5': (1.6, 2.2, 0.4),
        'efficientnet-b6': (1.8, 2.6, 0.5),
        'efficientnet-b7': (2.0, 3.1, 0.5),
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Unsupported EfficientNet model: {model_name}. "
                         f"Choose from {list(model_configs.keys())}.")
    
    width_coefficient, depth_coefficient, dropout_rate = model_configs[model_name]
    
    return EfficientNet3D(
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        dropout_rate=dropout_rate,
        model_name=model_name,
        **kwargs
    )


def get_efficientnet3d_b0():
    """Returns an EfficientNet3D-B0 model with default parameters for medical imaging."""
    m = get_efficientnet3d('efficientnet-b0', in_channels=1, num_classes=1)
    m.set_mode = lambda x: 0  # Does nothing
    return m