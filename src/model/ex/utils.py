import torch
import torch.nn as nn
from src.model.ex.Activate import *
from typing import Tuple, List, Dict, Union


def make_conv_bn_l(in_channel: int,
                   out_channel: int,
                   k_size: int,
                   activate: str = "leaky_relu",
                   stride: int = 1,
                   groups: int = 1):
    pad = (k_size - 1) // 2 if k_size else 0
    return nn.Sequential(nn.Conv2d(in_channels=in_channel,
                                   out_channels=out_channel,
                                   kernel_size=(k_size, k_size),
                                   stride=(stride, stride),
                                   groups=groups,
                                   padding=pad,
                                   bias=False),
                         nn.BatchNorm2d(out_channel),
                         get_activate(activate=activate))
