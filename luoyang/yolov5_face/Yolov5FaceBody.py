"""
yolo v5 face
"""
import torch
from typing import *
import torch.nn as nn
import luoyang.utils.file_utils as file_utils


def _auto_pad(k: Union[int, List[int]], p: Union[int, None]):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SiLU(nn.Module):

    def __init__(self):
        super(SiLU, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(x)


class Conv(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 k: int = 1,
                 s: int = 1,
                 p: int = None,
                 g: int = 1,
                 act: bool = True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(k, k),
                              stride=(s, s),
                              padding=_auto_pad(k, p),
                              groups=g,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x: torch.Tensor):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor):
        if self.act is None:
            return self.conv(x)
        return self.act(self.conv(x))


class StemBlock(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, act=True):
        super(StemBlock, self).__init__()
        self.stem_1 = Conv(c1, c2, k, s, p, g, act)
        self.stem_2a = Conv(c2, c2 // 2, 1, 1, 0)
        self.stem_2b = Conv(c2 // 2, c2, 3, 2, 1)
        self.stem_2p = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.stem_3 = Conv(c2 * 2, c2, 1, 1, 0)

    def forward(self, x):
        stem_1_out = self.stem_1(x)
        stem_2a_out = self.stem_2a(stem_1_out)
        stem_2b_out = self.stem_2b(stem_2a_out)
        stem_2p_out = self.stem_2p(stem_1_out)
        out = self.stem_3(torch.cat((stem_2b_out, stem_2p_out), 1))
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 shortcut: bool = True,
                 g: int = 1,
                 e: float = 0.5):
        super(Bottleneck, self).__init__()
        c_ = int(in_channels * e)  # hidden channels
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(c_, out_channels, 3, 1, g=g)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 n: int = 1,
                 shortcut: bool = True,
                 g: int = 1,
                 e: float = 0.5):
        super(C3, self).__init__()
        c_ = int(out_channels * e)
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(in_channels, c_, 1, 1)
        self.cv3 = Conv(2 * c_, out_channels, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 k: Tuple[int, int, int] = (3, 5, 7)):
        """

        :param in_channels:
        :param out_channels:
        :param k:
        """
        super(SPPF, self).__init__()
        c_ = in_channels // 2
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), out_channels, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class CSPDarknet(nn.Module):
    def __init__(self, base_channels: int,
                 base_depth: int):
        super(CSPDarknet, self).__init__()
        self.stem = StemBlock(3, base_channels, k=3)

        self.dark2 = nn.Sequential(
            C3(base_channels, base_channels * 2, base_depth))

        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C3(base_channels * 4, base_channels * 4, base_depth * 3))

        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C3(base_channels * 8, base_channels * 8, base_depth * 3))

        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2),
            SPPF(base_channels * 16, base_channels * 16),
            C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.dark2(x)
        x = self.dark3(x)
        feat1 = x
        x = self.dark4(x)
        feat2 = x
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3


def create_body(base_channels: int,
                base_depth: int,
                pretrained_path: str):
    """

    :param base_channels:
    :param base_depth:
    :param pretrained_path:
    :return:
    """
    net = CSPDarknet(base_channels=base_channels,
                     base_depth=base_depth)
    if pretrained_path is not None and file_utils.file_is_exists(path=pretrained_path):
        print("pretrained_path{0}".format(pretrained_path))
        info = net.load_state_dict(torch.load(pretrained_path, map_location="cpu"), strict=False)
        print(info)
    return net


class YoloV5FaceBody(nn.Module):

    def __init__(self, anchor_num: int,
                 num_classes: int,
                 point_num: int,
                 wid_mul: float,
                 dep_mul: float,
                 pretrained_path: str):
        super(YoloV5FaceBody, self).__init__()
        self.point_num = point_num
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        self.backbone = create_body(base_channels=base_channels,
                                    base_depth=base_depth,
                                    pretrained_path=pretrained_path)

        self.up_sample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3 = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_up_sample1 = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        self.conv_for_feat2 = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_up_sample2 = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        self.down_sample1 = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_down_sample1 = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        self.down_sample2 = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_down_sample2 = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

        self.down_sample3 = Conv(base_channels * 16, base_channels * 16, 3, 2)
        self.conv3_for_down_sample3 = C3(base_channels * 32, base_channels * 32, base_depth, shortcut=False)

        output_size = 5 + point_num * 2 + num_classes
        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, anchor_num * output_size, kernel_size=(1, 1))
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, anchor_num * output_size, kernel_size=(1, 1))
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, anchor_num * output_size, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param x:
        :return:
        """
        feat1, feat2, feat3 = self.backbone(x)

        p5 = self.conv_for_feat3(feat3)
        p5_up_sample = self.up_sample(p5)
        p4 = torch.cat([p5_up_sample, feat2], 1)
        p4 = self.conv3_for_up_sample1(p4)

        p4 = self.conv_for_feat2(p4)
        p4_up_sample = self.up_sample(p4)
        p3 = torch.cat([p4_up_sample, feat1], 1)
        p3 = self.conv3_for_up_sample2(p3)

        p3_down_sample = self.down_sample1(p3)
        p4 = torch.cat([p3_down_sample, p4], 1)
        p4 = self.conv3_for_down_sample1(p4)

        p4_down_sample = self.down_sample2(p4)
        p5 = torch.cat([p4_down_sample, p5], 1)
        p5 = self.conv3_for_down_sample2(p5)

        out2 = self.yolo_head_P3(p3)
        out1 = self.yolo_head_P4(p4)
        out0 = self.yolo_head_P5(p5)
        return out0, out1, out2
