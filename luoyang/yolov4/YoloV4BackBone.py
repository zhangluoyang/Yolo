import math
import torch
from typing import *
import torch.nn as nn
import torch.nn.functional as F
import luoyang.utils.file_utils as file_utils
from collections import OrderedDict


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class BasicConv(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              (kernel_size, kernel_size),
                              (stride, stride),
                              kernel_size // 2,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

    def forward_fuse(self, x: torch.Tensor):
        if self.activation is None:
            return self.conv(x)
        return self.activation(self.conv(x))


class ResBlock(nn.Module):
    def __init__(self, channels: int,
                 hidden_channels: int = None):
        super(ResBlock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResBlockBody(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 num_blocks: int,
                 first: bool):
        super(ResBlockBody, self).__init__()

        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)

        if first:
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)
            self.blocks_conv = nn.Sequential(
                ResBlock(channels=out_channels, hidden_channels=out_channels // 2),
                BasicConv(out_channels, out_channels, 1))
            self.concat_conv = BasicConv(out_channels * 2, out_channels, 1)
        else:
            self.split_conv0 = BasicConv(out_channels, out_channels // 2, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels // 2, 1)
            self.blocks_conv = nn.Sequential(
                *[ResBlock(out_channels // 2) for _ in range(num_blocks)],
                BasicConv(out_channels // 2, out_channels // 2, 1))
            self.concat_conv = BasicConv(out_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)
        return x


class CSPDarkNet(nn.Module):
    def __init__(self, layers: List[int]):
        super(CSPDarkNet, self).__init__()
        self.in_planes = 32
        self.conv1 = BasicConv(3, self.in_planes, kernel_size=3, stride=1)
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.stages = nn.ModuleList([
            ResBlockBody(self.in_planes, self.feature_channels[0], layers[0], first=True),
            ResBlockBody(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            ResBlockBody(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            ResBlockBody(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            ResBlockBody(self.feature_channels[3], self.feature_channels[4], layers[4], first=False)])

        self.num_features = 1
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.stages[0](x)
        x = self.stages[1](x)
        out3 = self.stages[2](x)
        out4 = self.stages[3](out3)
        out5 = self.stages[4](out4)

        return out3, out4, out5


def darknet53(darknet_csp_weight_path):
    model = CSPDarkNet([1, 2, 8, 8, 4])
    if darknet_csp_weight_path is not None and file_utils.file_is_exists(darknet_csp_weight_path):
        model.load_state_dict(torch.load(darknet_csp_weight_path))
    return model


def conv2d(filter_in: int,
           filter_out: int,
           kernel_size: int,
           stride: int = 1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out,
                           kernel_size=(kernel_size, kernel_size),
                           stride=(stride, stride),
                           padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1, inplace=True))]))


class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=None):
        super(SpatialPyramidPooling, self).__init__()

        if pool_sizes is None:
            pool_sizes = [5, 9, 13]
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)
        return features


class UpSample(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int):
        super(UpSample, self).__init__()

        self.up_sample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_sample(x)
        return x


def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1))
    return m


def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1))
    return m


def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1))
    return m


class YoloV4BackBone(nn.Module):
    def __init__(self, num_classes, darknet_csp_weight_path: str, anchor_num: int = 3):
        super(YoloV4BackBone, self).__init__()
        self.backbone = darknet53(darknet_csp_weight_path)

        self.conv1 = make_three_conv([512, 1024], 1024)
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512, 1024], 2048)

        self.up_sample1 = UpSample(512, 256)
        self.conv_for_P4 = conv2d(512, 256, 1)
        self.make_five_conv1 = make_five_conv([256, 512], 512)

        self.up_sample2 = UpSample(256, 128)
        self.conv_for_P3 = conv2d(256, 128, 1)
        self.make_five_conv2 = make_five_conv([128, 256], 256)

        self.yolo_head3 = yolo_head([256, anchor_num * (5 + num_classes)], 128)

        self.down_sample1 = conv2d(128, 256, 3, stride=2)
        self.make_five_conv3 = make_five_conv([256, 512], 512)

        self.yolo_head2 = yolo_head([512, anchor_num * (5 + num_classes)], 256)

        self.down_sample2 = conv2d(256, 512, 3, stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024], 1024)

        self.yolo_head1 = yolo_head([1024, anchor_num * (5 + num_classes)], 512)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x2, x1, x0 = self.backbone(x)

        p_5 = self.conv1(x0)
        p_5 = self.SPP(p_5)
        p_5 = self.conv2(p_5)

        p5_up_sample = self.up_sample1(p_5)
        p4 = self.conv_for_P4(x1)
        p4 = torch.cat([p4, p5_up_sample], dim=1)
        p4 = self.make_five_conv1(p4)

        p4_up_sample = self.up_sample2(p4)
        p3 = self.conv_for_P3(x2)
        p3 = torch.cat([p3, p4_up_sample], dim=1)
        p3 = self.make_five_conv2(p3)

        p3_down_sample = self.down_sample1(p3)
        p4 = torch.cat([p3_down_sample, p4], dim=1)
        p4 = self.make_five_conv3(p4)

        p4_down_sample = self.down_sample2(p4)
        p_5 = torch.cat([p4_down_sample, p_5], dim=1)
        p_5 = self.make_five_conv4(p_5)

        out2 = self.yolo_head3(p3)
        out1 = self.yolo_head2(p4)
        out0 = self.yolo_head1(p_5)

        return out0, out1, out2
