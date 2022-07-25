from collections import OrderedDict
from typing import Tuple, Union, List
import src.model.ex.utils as ex_utils

import torch
import torch.nn as nn

from src.model.ex.Darknet53 import create_darknet53


def _conv(filter_in: int,
          filter_out: int,
          kernel_size: int) -> nn.Module:
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(in_channels=filter_in,
                           out_channels=filter_out,
                           kernel_size=(kernel_size, kernel_size),
                           stride=(1, 1),
                           padding=pad,
                           bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("l_relu", nn.LeakyReLU(0.1)),
    ]))


def _make_last_layers(filters_list: Tuple[int, int],
                      in_filters: int,
                      out_filter: int) -> nn.Module:
    return nn.Sequential(
        _conv(filter_in=in_filters, filter_out=filters_list[0], kernel_size=1),
        _conv(filter_in=filters_list[0], filter_out=filters_list[1], kernel_size=3),
        _conv(filter_in=filters_list[1], filter_out=filters_list[0], kernel_size=1),
        _conv(filter_in=filters_list[0], filter_out=filters_list[1], kernel_size=3),
        _conv(filter_in=filters_list[1], filter_out=filters_list[0], kernel_size=1),
        _conv(filter_in=filters_list[0], filter_out=filters_list[1], kernel_size=3),
        nn.Conv2d(in_channels=filters_list[1],
                  out_channels=out_filter,
                  kernel_size=(1, 1),
                  stride=(1, 1),
                  padding=0,
                  bias=True))


class YoloV3Body(nn.Module):

    def __init__(self,
                 output_size: int,
                 darknet_pretrain_path: Union[str, None]):
        super(YoloV3Body, self).__init__()
        self._output_size = output_size
        self._darknet_pretrain_path = darknet_pretrain_path

        self.darknet53 = create_darknet53()

        output_filters: List[int] = self.darknet53.output_filters

        self.last_layer0 = _make_last_layers(filters_list=(512, 1024),
                                             in_filters=output_filters[-1],
                                             out_filter=output_size)

        self.last_layer1_conv = _conv(filter_in=512,
                                      filter_out=256,
                                      kernel_size=1)

        self.last_layer1_up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = _make_last_layers(filters_list=(256, 512),
                                             in_filters=output_filters[-2] + 256,
                                             out_filter=output_size)

        self.last_layer2_conv = _conv(filter_in=256,
                                      filter_out=128,
                                      kernel_size=1)
        self.last_layer2_up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = _make_last_layers(filters_list=(128, 256),
                                             in_filters=output_filters[-3] + 128,
                                             out_filter=output_size)

        if darknet_pretrain_path is not None:
            self.darknet53.load_state_dict(torch.load(darknet_pretrain_path, map_location="cpu"))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param x:
        :return:
        """
        x2, x1, x0 = self.darknet53(x)

        out0_branch = self.last_layer0[:5](x0)
        # [size/8]
        out0 = self.last_layer0[5:](out0_branch)

        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_up_sample(x1_in)

        x1_in = torch.cat([x1_in, x1], 1)

        out1_branch = self.last_layer1[:5](x1_in)
        # [size/16]
        out1 = self.last_layer1[5:](out1_branch)

        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_up_sample(x2_in)

        x2_in = torch.cat([x2_in, x2], 1)
        # [size/32]
        out2 = self.last_layer2(x2_in)
        return out0, out1, out2
