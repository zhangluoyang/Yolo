import copy
import math
import torch
from typing import *
import torch.nn as nn
import pkg_resources as pkg
import luoyang.utils.file_utils as file_utils
from luoyang.utils.torch_utils import fuse_conv_and_bn

def autopad(k: int, p: Union[None, int] = None, d: int = 1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    def __init__(self, c1: int,
                 c2: int,
                 k: int = 1,
                 s: int = 1,
                 p: Union[int, None] = None,
                 g: int = 1,
                 d: int = 1,
                 act: bool = True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def fuse_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class C2f(nn.Module):
    def __init__(self, c1: int,
                 c2: int,
                 n: int = 1,
                 shortcut: bool = False,
                 g: int = 1,
                 e: float = 0.5):
        super(C2f, self).__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    def __init__(self, c1: int,
                 c2: int,
                 shortcut: bool = True,
                 g: int = 1,
                 k: Tuple[int, int] = (3, 3),
                 e: float = 0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SCDown(nn.Module):
    def __init__(self, c1: int,
                 c2: int,
                 k: int,
                 s: int):
        super(SCDown, self).__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.cv1(x))


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed: int) -> None:
        super(RepVGGDW, self).__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):

    def __init__(self, c1: int,
                 c2: int,
                 shortcut: bool = True,
                 e: float = 0.5,
                 lk: bool = False):
        super(CIB, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Sequential(Conv(c1, c1, 3, g=c1),
                                 Conv(c1, 2 * c_, 1),
                                 Conv(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else RepVGGDW(2 * c_),
                                 Conv(2 * c_, c2, 1),
                                 Conv(c2, c2, 3, g=c2))

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):

    def __init__(self, c1: int,
                 c2: int,
                 n: int = 1,
                 shortcut: bool = False,
                 lk: bool = False,
                 g: int = 1,
                 e: float = 0.5):
        super(C2fCIB, self).__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class SPPF(nn.Module):
    def __init__(self, c1: int,
                 c2: int,
                 k: int = 5):
        super(SPPF, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Attention(nn.Module):
    def __init__(self, dim: int,
                 num_heads=8,
                 attn_ratio=0.5):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        # 位置编码 使用了
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (
                (q.transpose(-2, -1) @ k) * self.scale
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSA(nn.Module):

    def __init__(self, c1: int,
                 c2: int,
                 e: float = 0.5):
        super(PSA, self).__init__()
        assert (c1 == c2)
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1),
                                 Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class Backbone(nn.Module):

    def __init__(self,
                 depth_scale: float,
                 width_scale: float,
                 max_channel: int,
                 num_classes: int,
                 phi: str):
        super(Backbone, self).__init__()
        self.num_classes = num_classes
        self.phi = phi

        # # 640 -> 320
        self.conv_p1 = Conv(c1=3,
                            c2=_make_divisible(min(64, max_channel) * width_scale, 8),
                            k=3,
                            s=2)
        # 320 -> 160
        self.conv_p2 = Conv(c1=_make_divisible(min(64, max_channel) * width_scale, 8),
                            c2=_make_divisible(min(128, max_channel) * width_scale, 8),
                            k=3,
                            s=2)

        self.c2f_p3 = C2f(c1=_make_divisible(min(128, max_channel) * width_scale, 8),
                          c2=_make_divisible(min(128, max_channel) * width_scale, 8),
                          n=max(round(3 * depth_scale), 1),
                          shortcut=True)
        # 160 -> 80
        self.conv_p3 = Conv(c1=_make_divisible(min(128, max_channel) * width_scale, 8),
                            c2=_make_divisible(min(256, max_channel) * width_scale, 8),
                            k=3,
                            s=2)

        self.c3f_p4 = C2f(c1=_make_divisible(min(256, max_channel) * width_scale, 8),
                          c2=_make_divisible(min(256, max_channel) * width_scale, 8),
                          n=max(round(6 * depth_scale), 1),
                          shortcut=True)
        # 80 -> 40
        self.sc_down_p4 = SCDown(c1=_make_divisible(min(256, max_channel) * width_scale, 8),
                                 c2=_make_divisible(min(512, max_channel) * width_scale, 8),
                                 k=3,
                                 s=2)

        if phi in ["n", "s", "b", "l", "m"]:
            self.c_p5 = C2f(c1=_make_divisible(min(512, max_channel) * width_scale, 8),
                            c2=_make_divisible(min(512, max_channel) * width_scale, 8),
                            n=max(round(6 * depth_scale), 1),
                            shortcut=True)
        elif phi in ["x"]:
            self.c_p5 = C2fCIB(c1=_make_divisible(min(512, max_channel) * width_scale, 8),
                               c2=_make_divisible(min(512, max_channel) * width_scale, 8),
                               n=max(round(6 * depth_scale), 1),
                               shortcut=True)

        else:
            raise NotImplemented

        self.sc_down_p5 = SCDown(c1=_make_divisible(min(512, max_channel) * width_scale, 8),
                                 c2=_make_divisible(min(1024, max_channel) * width_scale, 8),
                                 k=3,
                                 s=2)
        if phi in ["n"]:
            self.c = C2f(c1=_make_divisible(min(1024, max_channel) * width_scale, 8),
                         c2=_make_divisible(min(1024, max_channel) * width_scale, 8),
                         n= max(round(3 * depth_scale), 1),
                         shortcut=True)
        elif phi in ["x", "b", "l", "m"]:
            self.c = C2fCIB(c1=_make_divisible(min(1024, max_channel) * width_scale, 8),
                            c2=_make_divisible(min(1024, max_channel) * width_scale, 8),
                            n= max(round(3 * depth_scale), 1),
                            shortcut=True)
        elif phi in ["s"]:
            self.c = C2fCIB(c1=_make_divisible(min(1024, max_channel) * width_scale, 8),
                            c2=_make_divisible(min(1024, max_channel) * width_scale, 8),
                            n= max(round(3 * depth_scale), 1),
                            shortcut=True,
                            lk=True)
        else:
            raise NotImplemented

        self.sppf = SPPF(c1=_make_divisible(min(1024, max_channel) * width_scale, 8),
                         c2=_make_divisible(min(1024, max_channel) * width_scale, 8),
                         k=5)

        self.psa = PSA(c1=_make_divisible(min(1024, max_channel) * width_scale, 8),
                       c2=_make_divisible(min(1024, max_channel) * width_scale, 8),
                       e=0.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor,
                                                torch.Tensor,
                                                torch.Tensor]:
        x = self.conv_p1(x)
        x = self.conv_p2(x)
        x = self.c2f_p3(x)

        x = self.conv_p3(x)
        p3_feature = self.c3f_p4(x)

        x = self.sc_down_p4(p3_feature)
        p4_feature = self.c_p5(x)

        x = self.sc_down_p5(p4_feature)
        x = self.c(x)
        x = self.sppf(x)
        p5_feature = self.psa(x)

        return p3_feature, p4_feature, p5_feature


class DFL(nn.Module):

    def __init__(self, c1=16):
        super(DFL, self).__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

def _make_divisible(x, divisor=8):
    return math.ceil(x / divisor) * divisor


def create_body(depth_scale: float,
                width_scale: float,
                max_channel: int,
                num_classes: int,
                phi: str,
                pretrained_path: str):
    """

    :param depth_scale:
    :param width_scale:
    :param max_channel:
    :param num_classes:
    :param phi:
    :param pretrained_path:
    :return:
    """
    net = Backbone(depth_scale=depth_scale,
                   width_scale=width_scale,
                   max_channel=max_channel,
                   num_classes=num_classes,
                   phi=phi)
    if pretrained_path is not None and file_utils.file_is_exists(path=pretrained_path):
        print("pretrained_path{0}".format(pretrained_path))
        net.load_state_dict(torch.load(pretrained_path, map_location="cpu"), strict=False)
    return net

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
    为每一个特征层生成 anchor
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

class YoloV10Body(nn.Module):
    def __init__(self, num_classes: int,
                 phi: str,
                 pretrained_path: Union[None, str]):
        super(YoloV10Body, self).__init__()
        depth_dict = {'b': 0.33, "n": 0.33, "s": 0.33, "m": 0.67, "l": 1.0, "x": 1.00}
        width_dict = {'b': 1.00, "n": 0.25, "s": 0.5, "m": 0.75, "l": 1.0, "x": 1.25}
        max_channels = {'b': 512, "n": 1024, "s": 1024, "m": 768, "l": 512, "x": 512}
        depth, width, max_channel = depth_dict[phi], width_dict[phi], max_channels[phi]

        self.backbone = create_body(depth_scale=depth,
                                    width_scale=width,
                                    max_channel=max_channel,
                                    num_classes=num_classes,
                                    phi=phi,
                                    pretrained_path=pretrained_path)

        self.up_sample = nn.Upsample(scale_factor=2, mode="nearest")

        if phi in ["b", "l", "x"]:
            self.p_13  = C2fCIB(c1=_make_divisible(min(1024, max_channel) * width + _make_divisible(min(512, max_channel) * width)),
                                c2=_make_divisible(min(512, max_channel) * width),
                                n=max(round(3 * depth), 1),
                                shortcut=True)
        elif phi in ["m", "n", "s"]:
            self.p_13 = C2f(c1=_make_divisible(min(1024, max_channel) * width + _make_divisible(min(512, max_channel) * width)),
                            c2=_make_divisible(min(512, max_channel) * width),
                            n=max(round(3 * depth), 1))
        else:
            raise NotImplemented

        self.p_16 = C2f(c1=_make_divisible(min(512, max_channel) * width) + _make_divisible(min(256, max_channel) * width),
                        c2=_make_divisible(min(256, max_channel) * width),
                        n=max(round(3 * depth), 1))

        self.p_17 = Conv(c1=_make_divisible(min(256, max_channel) * width),
                         c2=_make_divisible(min(256, max_channel) * width),
                         k=3,
                         s=2)

        if phi in ["b", "l", "m", "x"]:
            self.p_19 = C2fCIB(c1=_make_divisible(min(256, max_channel) * width)+_make_divisible(min(512, max_channel) * width),
                               c2=_make_divisible(min(512, max_channel) * width),
                               n=max(round(3 * depth), 1),
                               shortcut=True)
        elif phi in ["n", "s"]:
            self.p_19 = C2f(c1=_make_divisible(min(256, max_channel) * width)+_make_divisible(min(512, max_channel) * width),
                            c2=_make_divisible(min(512, max_channel) * width),
                            n=max(round(3 * depth), 1))
        else:
            raise NotImplemented

        self.p_20 =  SCDown(c1=_make_divisible(min(512, max_channel) * width, 8),
                                 c2=_make_divisible(min(512, max_channel) * width, 8),
                                 k=3,
                                 s=2)

        if phi in ["b", "l", "m", "x"]:
            self.p_22 = C2fCIB(
                c1=_make_divisible(min(512, max_channel) * width) + _make_divisible(min(1024, max_channel) * width),
                c2=_make_divisible(min(1024, max_channel) * width),
                n=max(round(3 * depth), 1),
                shortcut=True)

        elif phi in ["n", "s"]:
            self.p_22 =C2fCIB(
                c1=_make_divisible(min(512, max_channel) * width) + _make_divisible(min(1024, max_channel) * width),
                c2=_make_divisible(min(1024, max_channel) * width),
                n=max(round(3 * depth), 1),
                shortcut=True,
                lk=True)
        else:
            raise NotImplemented
        # 通道
        ch = [_make_divisible(min(256, max_channel) * width),
              _make_divisible(min(512, max_channel) * width),
              _make_divisible(min(1024, max_channel) * width)]

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
        self.cv3 = nn.ModuleList(nn.Sequential(nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                                               nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                                               nn.Conv2d(c3, self.num_classes, 1)) for i, x in enumerate(ch))

        # dfl 结果汇总
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        self.one2one_cv2 = copy.deepcopy(self.cv2)

        self.one2one_cv3 = copy.deepcopy(self.cv3)


    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # 主干网络 分别获取 8、16、32倍下采样的特征层 (特征金字塔)
        feat1, feat2, feat3 = self.backbone(x)
        # 路径聚合网络 底层与顶层特征
        p5_up_sample = self.up_sample(feat3)
        p4 = torch.cat([p5_up_sample, feat2], 1)
        p4 = self.p_13(p4)

        p4_up_sample = self.up_sample(p4)
        p3 = torch.cat([p4_up_sample, feat1], 1)
        p3 = self.p_16(p3)

        p3_down_sample = self.p_17(p3)
        p4 = torch.cat([p3_down_sample, p4], 1)
        p4 = self.p_19(p4)

        p4_down_sample = self.p_20(p4)
        p5 = torch.cat([p4_down_sample, feat3], 1)
        p5 = self.p_22(p5)
        x = [p3, p4, p5]
        for i in range(self.nl):
            # 回归与分类
            x[i] = torch.cat((self.one2one_cv2[i](x[i]), self.one2one_cv3[i](x[i])), 1)
        return x
