import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class Bottleneck3D(nn.Module):
    """
    Bottleneck block for ResNeXt with 3D convolutions.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for the convolution
        downsample: Downsample layer
        cardinality: Cardinality (number of groups)
        base_width: Base width of the bottleneck
        expansion: Expansion factor for the bottleneck
    """
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        cardinality: int = 32,
        base_width: int = 4,
        expansion: int = 4
    ):
        super(Bottleneck3D, self).__init__()
        width = int(out_channels * (base_width / 64.)) * cardinality
        
        # First convolution
        self.conv1 = nn.Conv3d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(width)
        
        # Grouped convolution
        self.conv2 = nn.Conv3d(
            width, width, kernel_size=3, stride=stride, padding=1,
            groups=cardinality, bias=False
        )
        self.bn2 = nn.BatchNorm3d(width)
        
        # Third convolution
        self.conv3 = nn.Conv3d(width, out_channels * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels * expansion)
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
        
        # Downsample layer
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNeXt3D(nn.Module):
    """
    3D ResNeXt implementation for medical image classification.
    
    Args:
        block: Block type (Bottleneck3D)
        layers: Number of layers in each block
        cardinality: Cardinality (number of groups)
        base_width: Base width of the bottleneck
        num_classes: Number of classification classes
        in_channels: Number of input channels
        zero_init_residual: Zero initialize residual branches
    """
    
    def __init__(
        self,
        block,
        layers: List[int],
        cardinality: int = 32,
        base_width: int = 4,
        num_classes: int = 1,
        in_channels: int = 1,
        zero_init_residual: bool = False
    ):
        super(ResNeXt3D, self).__init__()
        self.in_channels = 64
        self.cardinality = cardinality
        self.base_width = base_width
        
        # Initial convolution
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool3d((None, None, 1)),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(512 * block.expansion, num_classes)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck3D):
                    nn.init.constant_(m.bn3.weight, 0)
        
        # For compatibility
        self.register_buffer('_mode', torch.IntTensor([0]))
    
    def _make_layer(
        self,
        block: Bottleneck3D,
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """
        Create a layer with multiple blocks.
        
        Args:
            block: Block type
            out_channels: Number of output channels
            blocks: Number of blocks in the layer
            stride: Stride for the first block
            
        Returns:
            nn.Sequential: A sequential container of blocks
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, downsample,
                self.cardinality, self.base_width
            )
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels, out_channels,
                    cardinality=self.cardinality,
                    base_width=self.base_width
                )
            )
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unsqueeze during inference
        if x.dim() == 4:
            x = x.unsqueeze(0)
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.classifier(x)
        
        return x


def get_resnext3d(model_depth: str = '50', **kwargs):
    """
    Factory function to create ResNeXt3D models of different depths.
    
    Args:
        model_depth (str): One of '50', '101', or '152'
        **kwargs: Additional arguments to pass to ResNeXt3D
        
    Returns:
        ResNeXt3D: The requested model
    """
    if model_depth == '50':
        model = ResNeXt3D(Bottleneck3D, [3, 4, 6, 3], **kwargs)
    elif model_depth == '101':
        model = ResNeXt3D(Bottleneck3D, [3, 4, 23, 3], **kwargs)
    elif model_depth == '152':
        model = ResNeXt3D(Bottleneck3D, [3, 8, 36, 3], **kwargs)
    else:
        raise ValueError(f"Unsupported ResNeXt depth: {model_depth}. Choose from '50', '101', or '152'.")
    
    return model


def get_resnext3d_50():
    """Returns a ResNeXt3D-50 model with default parameters for medical imaging."""
    m = get_resnext3d('50', in_channels=1, num_classes=1)
    m.set_mode = lambda x: 0  # Does nothing
    return m