"""Sparse Convolution.

References:
    https://github.com/mit-han-lab/torchsparse/blob/master/examples/example.py
    https://github.com/mit-han-lab/e3d/blob/master/spvnas/core/models/semantic_kitti/spvcnn.py
"""

import torch
import torch.nn as nn
import numpy as np

import torchsparse
import torchsparse.nn as spnn
from torchsparse import PointTensor
from torchsparse.utils import sparse_collate

from spvcnn_utils import *


class BatchNorm(nn.Module):
    """A workaround for distributed training."""

    def __init__(self,
                 num_features: int,
                 *,
                 eps: float = 1e-5,
                 momentum: float = 0.1) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=num_features, eps=eps, momentum=momentum)

    def forward(self, inputs):
        feats = inputs.F
        coords = inputs.C
        stride = inputs.s
        feats = self.bn(feats)
        outputs = torchsparse.SparseTensor(coords=coords, feats=feats, stride=stride)
        outputs.coord_maps = inputs.coord_maps
        outputs.kernel_maps = inputs.kernel_maps
        return outputs


spnn.BatchNorm = BatchNorm


class BasicConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        self.conv = spnn.Conv3d(in_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                dilation=dilation,
                                stride=stride)
        self.bn = spnn.BatchNorm(out_channels)
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        self.deconv = spnn.Conv3d(in_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  dilation=dilation,
                                  stride=stride,
                                  transpose=True)
        self.bn = spnn.BatchNorm(out_channels)
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(True),
            spnn.Conv3d(out_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        stride=1),
            spnn.BatchNorm(out_channels),
        )

        if in_channels == out_channels and stride == 1:
            self.downsample = nn.Sequential()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(in_channels, out_channels, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(out_channels)
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class MinkUNet(nn.Module):
    def __init__(self, in_channels, voxel_size, num_classes, cr=1.0):
        super().__init__()

        self.voxel_size = voxel_size
        self.num_classes = num_classes
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]

        self.stem = nn.Sequential(
            spnn.Conv3d(in_channels, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], kernel_size=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], kernel_size=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], kernel_size=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], kernel_size=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], kernel_size=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], kernel_size=3, stride=1, dilation=1),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], kernel_size=2, stride=2, dilation=1),
            ResidualBlock(cs[2], cs[3], kernel_size=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], kernel_size=3, stride=1, dilation=1),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], kernel_size=2, stride=2, dilation=1),
            ResidualBlock(cs[3], cs[4], kernel_size=3, stride=1, dilation=1),
            ResidualBlock(cs[4], cs[4], kernel_size=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], kernel_size=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[5] + cs[3], cs[5], kernel_size=3, stride=1, dilation=1),
                ResidualBlock(cs[5], cs[5], kernel_size=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], kernel_size=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[6] + cs[2], cs[6], kernel_size=3, stride=1, dilation=1),
                ResidualBlock(cs[6], cs[6], kernel_size=3, stride=1, dilation=1),
            )
        ])

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], kernel_size=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[7] + cs[1], cs[7], kernel_size=3, stride=1, dilation=1),
                ResidualBlock(cs[7], cs[7], kernel_size=3, stride=1, dilation=1),
            )
        ])

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], kernel_size=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[8] + cs[0], cs[8], kernel_size=3, stride=1, dilation=1),
                ResidualBlock(cs[8], cs[8], kernel_size=3, stride=1, dilation=1),
            )
        ])

        self.classifier = nn.Linear(cs[8], num_classes)

    def forward(self, x: torchsparse.PointTensor):
        x0 = initial_voxelize(x, 1.0, self.voxel_size)

        x0 = self.stem(x0)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        y1 = self.up1[0](x4)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)

        out = voxel_to_point(y4, x)
        out = self.classifier(out.F)

        return out


def test():

    # spvoxelize only wraps GPU
    model = MinkUNet(in_channels=1, voxel_size=0.2, num_classes=80).cuda()

    pc = np.random.randn(10000, 3)
    pc[:, :3] = pc[:, :3] * 10
    pc = np.hstack([pc, np.zeros([10000, 1])])
    coords, feats = sparse_collate([pc], [np.random.randn(10000, 1)], coord_float=True)
    x = PointTensor(feats, coords).cuda()

    out = model(x)
    print(out.shape)
    assert out.shape == (10000, 80)


if __name__ == "__main__":
    test()
