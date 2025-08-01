import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['get_ResNet3d_101', 'get_vgg16']

# -- ResNet3d
def get_inplanes():
    return [64, 128, 256, 512]

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(7, 7, conv1_t_size),
                               stride=(2, 2, conv1_t_stride),
                               padding=(3, 3, conv1_t_size // 2),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool3d((None, None, 1)),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # For compatibility
        self.register_buffer('_mode', torch.IntTensor([0]))

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Unsqueeze during inference
        if x.dim() == 4:
            x = x.unsqueeze(0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.classifier(x)

        return x


def get_ResNet3d(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet3D(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet3D(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet3D(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet3D(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet3D(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet3D(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet3D(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

def get_ResNet3d_101():
    r"""Expected input size"""
    m = get_ResNet3d(101, n_input_channels=1, n_classes=1, conv1_t_size=1)
    m.set_mode = lambda x: 0 # Does nothing
    return m

# -- 3D VGG
import torch
import torch.nn as nn


class VGG16_3D(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super(VGG16_3D, self).__init__()

        # VGG16 3D configuration (channels, number of conv layers)
        self.arch_config = [
            (64, 2),  # Block 1
            (128, 2),  # Block 2
            (256, 3),  # Block 3
            (512, 3),  # Block 4
            (512, 3),  # Block 5
        ]

        # Build feature layers
        self.features = self._make_layers(in_channels)

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool3d((None, None, 1)), # Max pool each slice
            nn.AdaptiveAvgPool3d((1, 1, 1)), # Average all
            nn.Flatten(),
            nn.Linear(512, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(2048, num_classes)
        )

        self._initialize_weights()
        # For compatibility
        self.register_buffer('_mode', torch.IntTensor([0]))

    def _make_layers(self, in_channels):
        layers = []
        input_channels = in_channels

        for out_channels, num_convs in self.arch_config:
            layers.extend(self._make_conv_block(
                input_channels,
                out_channels,
                num_convs
            ))
            input_channels = out_channels

        return nn.Sequential(*layers)

    def _make_conv_block(self, in_channels, out_channels, num_convs):
        layers = []

        # First conv layer of the block
        layers.extend([
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ])

        # Remaining conv layers in the block
        for _ in range(num_convs - 1):
            layers.extend([
                nn.Conv3d(out_channels, out_channels, kernel_size=[3,3,1], padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            ])

        # Add maxpool at the end of each block
        layers.append(nn.MaxPool3d(kernel_size=2, stride=2))

        return layers

    def forward(self, x):
        # Unsqueeze if 4D during inference
        if x.dim() == 4:
            x = x.unsqueeze(0)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def get_vgg16():
    r"""Assume input size"""
    m = VGG16_3D(num_classes=1, in_channels=1)
    m.set_mode = lambda x: 0 # Does nothing
    return m