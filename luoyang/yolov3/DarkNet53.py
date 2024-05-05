"""
darknet53 network
"""
import torch
import torch.nn as nn
from typing import Tuple, List
from collections import OrderedDict


class BasicBlock(nn.Module):
    """
    resnet
    """

    def __init__(self, in_planes: int, planes: Tuple[int, int]):
        """
        :param in_planes: 输入通道数目
        :param planes: (输入通道数目, 输出通道数目)
        """
        super(BasicBlock, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_planes,
                                            out_channels=planes[0],
                                            kernel_size=(1, 1),
                                            stride=(1, 1),
                                            padding=0,
                                            bias=False),
                                  nn.BatchNorm2d(planes[0]),
                                  nn.LeakyReLU(0.1, inplace=True),
                                  nn.Conv2d(planes[0],
                                            planes[1],
                                            kernel_size=(3, 3),
                                            stride=(1, 1),
                                            padding=1,
                                            bias=False),
                                  nn.BatchNorm2d(planes[1]),
                                  nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        residual = x
        output = self.conv(x)
        return residual + output


def _make_layer(planes: Tuple[int, int], block: int) -> nn.Module:
    layers = [("conv", nn.Conv2d(planes[0],
                                 planes[1],
                                 kernel_size=(3, 3),
                                 stride=(2, 2),
                                 padding=1,
                                 bias=False)),
              ("bn", nn.BatchNorm2d(planes[1])),
              ("l_relu", nn.LeakyReLU(0.1))]
    for i in range(block):
        layers.append(("residual_{}".format(i), BasicBlock(planes[1], planes)))
    return nn.Sequential(OrderedDict(layers))


class DarkNet53(nn.Module):
    """
    darknet
    """

    def __init__(self, blocks: List[int],
                 output_filters: List[int]):
        super(DarkNet53, self).__init__()
        self.blocks = blocks
        self.output_filters = output_filters

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                             out_channels=32,
                                             kernel_size=(3, 3),
                                             stride=(1, 1),
                                             padding=1,
                                             bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.LeakyReLU(0.1, inplace=True))

        self.layer_1 = _make_layer(planes=(32, self.output_filters[0]), block=blocks[0])
        self.layer_2 = _make_layer(planes=(64, self.output_filters[1]), block=blocks[1])
        self.layer_3 = _make_layer(planes=(128, self.output_filters[2]), block=blocks[2])
        self.layer_4 = _make_layer(planes=(256, self.output_filters[3]), block=blocks[3])
        self.layer_5 = _make_layer(planes=(512, self.output_filters[4]), block=blocks[4])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param x:
        :return:
        """
        cnn_out = self.conv1(x)

        darknet_out1 = self.layer_1(cnn_out)
        darknet_out2 = self.layer_2(darknet_out1)
        # 8倍下采样
        darknet_out3 = self.layer_3(darknet_out2)
        # 16倍下采样
        darknet_out4 = self.layer_4(darknet_out3)
        # 32倍下采样
        darknet_out5 = self.layer_5(darknet_out4)

        return darknet_out3, darknet_out4, darknet_out5


def create_darknet53() -> DarkNet53:
    return DarkNet53(blocks=[1, 2, 8, 8, 4],
                     output_filters=[64, 128, 256, 512, 1024])
