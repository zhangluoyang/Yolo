import torch
import numpy as np
from typing import *
import torch.nn as nn
import pkg_resources as pkg
import luoyang.utils.file_utils as file_utils
import luoyang.utils.torch_utils as torch_utils


def auto_pad(k: Union[int, tuple],
             p: Union[int, None] = None,
             d: int = 1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Conv(nn.Module):
    default_act = SiLU()

    def __init__(self, c1: int,
                 c2: int,
                 k: Union[int, Tuple[int, int]] = 1,
                 s: Union[int, Tuple[int, int]] = 1,
                 p: int = None,
                 g: int = 1,
                 d: int = 1,
                 act=True):
        super().__init__()

        if isinstance(k, int):
            k = (k, k)
        if isinstance(s, int):
            s = (s, s)

        self.conv = nn.Conv2d(c1,
                              c2,
                              k,
                              s,
                              auto_pad(k, p, d),
                              groups=g,
                              dilation=(d, d),
                              bias=False)
        self.bn = nn.BatchNorm2d(c2,
                                 eps=0.001,
                                 momentum=0.03,
                                 affine=True,
                                 track_running_stats=True)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, c1: int,
                 c2: int,
                 k: List[Tuple[int, int]],
                 shortcut: bool = True,
                 g: int = 1,
                 e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(c1=self.c,
                                          c2=self.c,
                                          shortcut=shortcut,
                                          g=g,
                                          k=[(3, 3), (3, 3)],
                                          e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Backbone(nn.Module):

    def __init__(self, base_channels: int,
                 base_depth: int,
                 deep_mul: float):
        super().__init__()

        self.stem = Conv(3, base_channels, 3, 2)

        self.dark2 = nn.Sequential(Conv(base_channels, base_channels * 2, 3, 2),
                                   C2f(base_channels * 2, base_channels * 2, base_depth, True))

        self.dark3 = nn.Sequential(Conv(base_channels * 2, base_channels * 4, 3, 2),
                                   C2f(base_channels * 4, base_channels * 4, base_depth * 2, True))

        self.dark4 = nn.Sequential(Conv(base_channels * 4, base_channels * 8, 3, 2),
                                   C2f(base_channels * 8, base_channels * 8, base_depth * 2, True))

        self.dark5 = nn.Sequential(Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
                                   C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul),
                                       base_depth, True),
                                   SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor,
                                                torch.Tensor,
                                                torch.Tensor]:
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
                deep_mul: float,
                pretrained_path: str):
    """

    :param base_channels:
    :param base_depth:
    :param deep_mul:
    :param pretrained_path:
    :return:
    """
    net = Backbone(base_channels=base_channels,
                   base_depth=base_depth,
                   deep_mul=deep_mul)
    if pretrained_path is not None and file_utils.file_is_exists(path=pretrained_path):
        print("pretrained_path{0}".format(pretrained_path))
        net.load_state_dict(torch.load(pretrained_path, map_location="cpu"), strict=False)
    return net


class DFL(nn.Module):
    """
    distribution focal loss
    """

    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


def check_version(current: str = "0.0.0",
                  minimum: str = "0.0.0",
                  name: str = "version ",
                  pinned: bool = False) -> bool:
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    return result


TORCH_1_10 = check_version(torch.__version__, '1.10.0')


def make_anchors(feature_size: List[int],
                 strides: List[int],
                 grid_cell_offset: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    未每一个特征层生成 anchor
    :param feature_size: (特征层 tensor对应的尺寸大小)
    :param strides: (特征层下采样步长)
    :param grid_cell_offset: 偏移量 (中心点距离左上角偏移0.5)
    :return:
    """
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        # 当前特征层尺寸
        _size = feature_size[i]
        sx = torch.arange(end=_size) + grid_cell_offset
        sy = torch.arange(end=_size) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((_size * _size, 1), stride))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


class YoloV8Body(nn.Module):

    def __init__(self, num_classes: int,
                 phi: str,
                 pretrained_path: Union[None, str]):
        super(YoloV8Body, self).__init__()
        depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.00, }
        width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        deep_width_dict = {'n': 1.00, 's': 1.00, 'm': 0.75, 'l': 0.50, 'x': 0.50}
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]

        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        self.backbone = create_body(base_channels=base_channels,
                                    base_depth=base_depth,
                                    deep_mul=deep_mul,
                                    pretrained_path=pretrained_path)

        self.up_sample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv3_for_up_sample1 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8,
                                        base_channels * 8,
                                        base_depth,
                                        shortcut=False)

        self.conv3_for_up_sample2 = C2f(base_channels * 8 + base_channels * 4,
                                        base_channels * 4,
                                        base_depth,
                                        shortcut=False)

        self.down_sample1 = Conv(base_channels * 4, base_channels * 4, 3, 2)

        self.conv3_for_down_sample1 = C2f(base_channels * 8 + base_channels * 4,
                                          base_channels * 8,
                                          base_depth,
                                          shortcut=False)

        self.down_sample2 = Conv(base_channels * 8, base_channels * 8, 3, 2)

        self.conv3_for_down_sample2 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8,
                                          int(base_channels * 16 * deep_mul),
                                          base_depth,
                                          shortcut=False)

        ch = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
        self.shape = None
        self.nl = len(ch)
        self.reg_max = 16
        self.output_size = num_classes + self.reg_max * 4
        self.num_classes = num_classes

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes)  # channels
        # 目标框回归输出层
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        # 类别输出
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, num_classes, 1)) for x in ch)
        # dfl 结果汇总
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # 主干网络 分别获取 8、16、32倍下采样的特征层 (特征金字塔)
        feat1, feat2, feat3 = self.backbone.forward(x)
        # 路径聚合网络 底层与顶层特征
        p5_up_sample = self.up_sample(feat3)
        p4 = torch.cat([p5_up_sample, feat2], 1)
        p4 = self.conv3_for_up_sample1(p4)
        p4_up_sample = self.up_sample(p4)
        p3 = torch.cat([p4_up_sample, feat1], 1)
        p3 = self.conv3_for_up_sample2(p3)
        p3_down_sample = self.down_sample1(p3)
        p4 = torch.cat([p3_down_sample, p4], 1)
        p4 = self.conv3_for_down_sample1(p4)
        p4_down_sample = self.down_sample2(p4)
        p5 = torch.cat([p4_down_sample, feat3], 1)
        p5 = self.conv3_for_down_sample2(p5)
        x = [p3, p4, p5]
        for i in range(self.nl):
            # 回归与分类
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        return x
