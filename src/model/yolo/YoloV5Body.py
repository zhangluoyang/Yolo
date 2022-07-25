"""
yolo v5
"""
import torch
from typing import *
import torch.nn as nn


def _auto_pad(k: int, p: Tuple[int, None]):
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

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Focus(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 k: int = 1,
                 s: int = 1,
                 p: int = None,
                 g: int = 1,
                 act: bool = True):
        super(Focus, self).__init__()
        self.conv = Conv(in_channels=in_channels * 4,
                         out_channels=out_channels,
                         k=k,
                         s=s,
                         p=p,
                         g=g,
                         act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(
            torch.cat([
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2]], dim=1))


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


class SPP(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 k: Tuple[int, int, int] = (5, 9, 13)):
        """

        :param in_channels:
        :param out_channels:
        :param k:
        """
        super(SPP, self).__init__()
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
        super().__init__()
        self.stem = Focus(3, base_channels, k=3)

        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C3(base_channels * 2, base_channels * 2, base_depth),
        )

        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C3(base_channels * 4, base_channels * 4, base_depth * 3))

        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C3(base_channels * 8, base_channels * 8, base_depth * 3))

        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2),
            SPP(base_channels * 16, base_channels * 16),
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
    print("pretrained_path{0}".format(pretrained_path))
    net.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
    return net


class YoloV5Body(nn.Module):
    def __init__(self, anchor_num,
                 num_classes,
                 wid_mul: float,
                 dep_mul: float,
                 pretrained_path: str):
        super(YoloV5Body, self).__init__()

        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)
        # ---------------------------------------------------#
        self.backbone = create_body(base_channels, base_depth, pretrained_path=pretrained_path)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3 = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1 = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        self.conv_for_feat2 = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2 = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        self.down_sample1 = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1 = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        self.down_sample2 = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2 = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, anchor_num * (5 + num_classes), kernel_size=(1, 1))
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, anchor_num * (5 + num_classes), kernel_size=(1, 1))
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, anchor_num * (5 + num_classes), kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param x:
        :return:
        """
        feat1, feat2, feat3 = self.backbone(x)

        P5 = self.conv_for_feat3(feat3)
        P5_up_sample = self.upsample(P5)
        P4 = torch.cat([P5_up_sample, feat2], 1)
        P4 = self.conv3_for_upsample1(P4)

        P4 = self.conv_for_feat2(P4)
        P4_up_sample = self.upsample(P4)
        P3 = torch.cat([P4_up_sample, feat1], 1)
        P3 = self.conv3_for_upsample2(P3)

        P3_down_sample = self.down_sample1(P3)
        P4 = torch.cat([P3_down_sample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)

        P4_down_sample = self.down_sample2(P4)
        P5 = torch.cat([P4_down_sample, P5], 1)
        P5 = self.conv3_for_downsample2(P5)

        out2 = self.yolo_head_P3(P3)
        out1 = self.yolo_head_P4(P4)
        out0 = self.yolo_head_P5(P5)
        return out0, out1, out2
