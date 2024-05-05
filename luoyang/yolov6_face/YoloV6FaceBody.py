"""
yolo v6 版本
"""
import math
import torch
import numpy as np
from typing import *
import torch.nn as nn
import torch.nn.functional as F
from luoyang.param.Param import Yolo6FaceParam

activation_table = {'relu': nn.ReLU(),
                    'silu': nn.SiLU(),
                    'hardswish': nn.Hardswish()}

# 配置
_config = {"s": {"depth_multiple": 0.33,
                 "width_multiple": 0.50,
                 "backbone": {"num_repeats": [1, 6, 12, 18, 6, 6],
                              "out_channels": [64, 128, 256, 512, 768, 1024]},
                 "neck": {"num_repeats": [12, 12, 12, 12, 12, 12],
                          "out_channels": [512, 256, 128, 256, 512, 1024]},
                 "head": {"in_channels": [128, 256, 512, 1024],
                          "num_layers": 3,
                          "anchors": 1,
                          "strides": [8, 16, 32, 64],
                          "atss_warmup_epoch": 4,
                          "iou_type": 'giou',
                          "use_dfl": False,
                          "reg_max": 0}
                 },
           "n": {"depth_multiple": 0.33,
                 "width_multiple": 0.25,
                 "backbone": {"num_repeats": [1, 6, 12, 18, 6, 6],
                              "out_channels": [64, 128, 256, 512, 768, 1024]},
                 "neck": {"num_repeats": [12, 12, 12, 12, 12, 12],
                          "out_channels": [512, 256, 128, 256, 512, 1024]},
                 "head": {"in_channels": [128, 256, 512, 1024],
                          "num_layers": 3,
                          "anchors": 1,
                          "strides": [8, 16, 32, 64],
                          "atss_warmup_epoch": 4,
                          "iou_type": 'siou',
                          "use_dfl": False,
                          "reg_max": 0}
                 },
           "m": {"depth_multiple": 0.6,
                 "width_multiple": 0.75,
                 "backbone": {"num_repeats": [1, 6, 12, 18, 6, 6],
                              "out_channels": [64, 128, 256, 512, 768, 1024]},
                 "neck": {"num_repeats": [12, 12, 12, 12, 12, 12],
                          "out_channels": [512, 256, 128, 256, 512, 1024]},
                 "head": {"in_channels": [128, 256, 512, 1024],
                          "num_layers": 3,
                          "anchors": 1,
                          "strides": [8, 16, 32, 64],
                          "atss_warmup_epoch": 4,
                          "iou_type": 'giou',
                          "use_dfl": True,
                          "reg_max": 16,
                          "distill_weight": {
                              'class': 1.0,
                              'dfl': 1.0}
                          }
                 },
           "l": {"depth_multiple": 1.0,
                 "width_multiple": 1.0,
                 "backbone": {"num_repeats": [1, 6, 12, 18, 6, 6],
                              "out_channels": [64, 128, 256, 512, 768, 1024]},
                 "neck": {"num_repeats": [12, 12, 12, 12, 12, 12],
                          "out_channels": [512, 256, 128, 256, 512, 1024]},
                 "head": {"in_channels": [128, 256, 512, 1024],
                          "num_layers": 3,
                          "anchors": 1,
                          "strides": [8, 16, 32, 64],
                          "atss_warmup_epoch": 4,
                          "iou_type": 'giou',
                          "use_dfl": True,
                          "reg_max": 16,
                          "distill_weight": {
                              'class': 1.0,
                              'dfl': 1.0}
                          }
                 }
           }


class SiLU(nn.Module):

    def __init__(self):
        super(SiLU, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        return x * self.sigmoid(x)


class ConvModule(nn.Module):
    """
    cnn + bn +activate 网络结构
    """

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 activation_type: Union[str, None],
                 padding=None,
                 groups: int = 1,
                 bias=False):
        super(ConvModule, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(kernel_size, kernel_size),
                              stride=(stride, stride),
                              padding=padding,
                              groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        if activation_type is not None:
            self.act = activation_table.get(activation_type)
        self.activation_type = activation_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_type is None:
            return self.bn(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_type is None:
            return self.conv(x)
        return self.act(self.conv(x))


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 padding_mode: str = 'zeros',
                 deploy: bool = False,
                 use_se: bool = False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2
        # 仅仅保留 大于0的值
        self.non_linearity = nn.ReLU()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            # 重参化之后的结果
            self.rbr_re_param = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=(kernel_size, kernel_size),
                                          stride=(stride, stride),
                                          padding=padding,
                                          dilation=(dilation, dilation),
                                          groups=groups,
                                          bias=True,
                                          padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = ConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, activation_type=None, padding=padding, groups=groups)
            self.rbr_1x1 = ConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                      activation_type=None, padding=padding_11, groups=groups)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """

        :param inputs:
        :return:
        """
        if hasattr(self, 'rbr_re_param'):
            # 重参化之后的结果
            return self.non_linearity(self.se(self.rbr_re_param(inputs)))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.non_linearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        """
        重参化后 获取等价的 卷积核 偏置
        :return:
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel_id, bias_id = self._fuse_bn_tensor(self.rbr_identity)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernel_id, bias3x3 + bias1x1 + bias_id

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.in_channels
        groups = self.groups
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """
        将 1*1 的卷积核 扩充至 3*3
        :param kernel1x1:
        :return:
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: nn.Module):
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvModule):
            # 卷积 返回卷积核权重即可
            kernel = branch.conv.weight
            bias = branch.conv.bias
            if branch.conv.bias is None:
                bias = torch.zeros(kernel.size(0), dtype=kernel.dtype).to(kernel.device)
            return kernel, bias
        elif isinstance(branch, nn.BatchNorm2d):
            # 归一化层 转换为 卷积核的形式
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                # 卷积核
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """
        转换为部署模式
        :return:`
        """
        if hasattr(self, 'rbr_re_param'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_re_param = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                      out_channels=self.rbr_dense.conv.out_channels,
                                      kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                      padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                      groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_re_param.weight.data = kernel
        self.rbr_re_param.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class ConvBNReLU(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: Union[None, int] = None,
                 groups: int = 1,
                 bias: bool = False):
        super(ConvBNReLU, self).__init__()
        self.block = ConvModule(in_channels, out_channels, kernel_size, stride, 'relu', padding, groups, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBNSiLU(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: Union[int, None] = None,
                 groups: int = 1,
                 bias: bool = False):
        super(ConvBNSiLU, self).__init__()
        self.block = ConvModule(in_channels, out_channels, kernel_size, stride, 'silu', padding, groups, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SPPFModule(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int,
                 block: nn.Module,
                 kernel_size: int = 5):
        super(SPPFModule, self).__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = block(in_channels, c_, 1, 1)
        self.cv2 = block(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class SimSPPF(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size=5,
                 block=ConvBNReLU):
        super(SimSPPF, self).__init__()
        self.spp_f = SPPFModule(in_channels=in_channels,
                                out_channels=out_channels,
                                block=block,
                                kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spp_f(x)


class SPPF(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 5,
                 block: ConvBNSiLU = ConvBNSiLU):
        super(SPPF, self).__init__()
        self.sppf = SPPFModule(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               block=block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sppf(x)


class CSPSPPFModule(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 5,
                 e: float = 0.5,
                 block: Union[ConvBNReLU, ConvBNSiLU] = ConvBNReLU):
        super(CSPSPPFModule, self).__init__()
        c_ = int(out_channels * e)
        self.cv1 = block(in_channels, c_, 1, 1)
        self.cv2 = block(in_channels, c_, 1, 1)
        self.cv3 = block(c_, c_, 3, 1)
        self.cv4 = block(c_, c_, 1, 1)

        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.cv5 = block(4 * c_, c_, 1, 1)
        self.cv6 = block(c_, c_, 3, 1)
        self.cv7 = block(2 * c_, out_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y0 = self.cv2(x)
        y1 = self.m(x1)
        y2 = self.m(y1)
        y3 = self.cv6(self.cv5(torch.cat([x1, y1, y2, self.m(y2)], 1)))
        return self.cv7(torch.cat((y0, y3), dim=1))


class CSPSPPF(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 5,
                 e: float = 0.5,
                 block: Union[ConvBNReLU, ConvBNSiLU] = ConvBNSiLU):
        super(CSPSPPF, self).__init__()
        self.csp_spp_f = CSPSPPFModule(in_channels, out_channels, kernel_size, e, block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.csp_spp_f(x)


class SimCSPSPPF(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 5,
                 e: float = 0.5,
                 block: Union[ConvBNReLU, ConvBNSiLU] = ConvBNReLU):
        super().__init__()
        self.cspsppf = CSPSPPFModule(in_channels, out_channels, kernel_size, e, block)

    def forward(self, x):
        return self.cspsppf(x)


class RepBlock(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int,
                 n: int = 1,
                 block=RepVGGBlock,
                 basic_block=RepVGGBlock):
        super(RepBlock, self).__init__()
        self.conv1 = block(in_channels, out_channels)
        self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None
        if block == BottleRep:
            self.conv1 = BottleRep(in_channels, out_channels, basic_block=basic_block, weight=True)
            n = n // 2
            self.block = nn.Sequential(
                *(BottleRep(out_channels, out_channels, basic_block=basic_block, weight=True) for _ in
                  range(n - 1))) if n > 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class BottleRep(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int,
                 basic_block=RepVGGBlock,
                 weight: bool = False):
        super(BottleRep, self).__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs


class Transpose(nn.Module):
    """
    用于上采样 (转置卷积)
    """

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 2,
                 stride: int = 2):
        super(Transpose, self).__init__()
        self.up_sample_transpose = torch.nn.ConvTranspose2d(in_channels=in_channels,
                                                            out_channels=out_channels,
                                                            kernel_size=(kernel_size, kernel_size),
                                                            stride=(stride, stride),
                                                            bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_sample_transpose(x)


class EfficientRep6(nn.Module):

    def __init__(self,
                 channels_list: List[int],
                 num_repeats: List[int],
                 in_channels: int = 3,
                 block=Type[RepVGGBlock],
                 fuse_P2: bool = False,
                 csp_spp_f: bool = False):
        super(EfficientRep6, self).__init__()
        self.fuse_P2 = fuse_P2
        self.csp_spp_f = csp_spp_f
        self.stem = block(in_channels=in_channels,
                          out_channels=channels_list[0],
                          kernel_size=3,
                          stride=2)

        self.ERBlock_2 = nn.Sequential(
            block(in_channels=channels_list[0],
                  out_channels=channels_list[1],
                  kernel_size=3,
                  stride=2),
            RepBlock(in_channels=channels_list[1],
                     out_channels=channels_list[1],
                     n=num_repeats[1],
                     block=block))

        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2),
            RepBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                block=block))

        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2),
            RepBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                block=block))

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2),
            RepBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                block=block))

        channel_merge_layer = SimSPPF if not csp_spp_f else SimCSPSPPF

        self.ERBlock_6 = nn.Sequential(block(in_channels=channels_list[4],
                                             out_channels=channels_list[5],
                                             kernel_size=3,
                                             stride=2),
                                       RepBlock(in_channels=channels_list[5],
                                                out_channels=channels_list[5],
                                                n=num_repeats[5],
                                                block=block),
                                       channel_merge_layer(in_channels=channels_list[5],
                                                           out_channels=channels_list[5],
                                                           kernel_size=5))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)
        x = self.ERBlock_6(x)
        outputs.append(x)
        return tuple(outputs)


class BiFusion(nn.Module):

    def __init__(self, in_channels: List[int],
                 out_channels: int):
        super(BiFusion, self).__init__()
        self.cv1 = ConvBNReLU(in_channels[0], out_channels, 1, 1)
        self.cv2 = ConvBNReLU(in_channels[1], out_channels, 1, 1)
        self.cv3 = ConvBNReLU(out_channels * 3, out_channels, 1, 1)

        self.up_sample = Transpose(in_channels=out_channels,
                                   out_channels=out_channels)
        self.down_sample = ConvBNReLU(in_channels=out_channels,
                                      out_channels=out_channels,
                                      kernel_size=3,
                                      stride=2)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x0 = self.up_sample(x[0])
        x1 = self.cv1(x[1])
        x2 = self.down_sample(self.cv2(x[2]))
        return self.cv3(torch.cat((x0, x1, x2), dim=1))


class RepBiFPANNeck6(nn.Module):

    def __init__(self, channels_list: List[int],
                 num_repeats: List[int],
                 block=RepVGGBlock):
        super(RepBiFPANNeck6, self).__init__()

        self.reduce_layer0 = ConvBNReLU(in_channels=channels_list[5],
                                        out_channels=channels_list[6],
                                        kernel_size=1,
                                        stride=1)

        self.Bi_fusion_0 = BiFusion(in_channels=[channels_list[4], channels_list[6]],
                                    out_channels=channels_list[6])

        self.Rep_p5 = RepBlock(in_channels=channels_list[6],
                               out_channels=channels_list[6],
                               n=num_repeats[6],
                               block=block)

        self.reduce_layer1 = ConvBNReLU(in_channels=channels_list[6],
                                        out_channels=channels_list[7],
                                        kernel_size=1,
                                        stride=1)

        self.Bi_fusion_1 = BiFusion(in_channels=[channels_list[3], channels_list[7]],
                                    out_channels=channels_list[7])

        self.Rep_p4 = RepBlock(in_channels=channels_list[7],
                               out_channels=channels_list[7],
                               n=num_repeats[7],
                               block=block)

        self.reduce_layer2 = ConvBNReLU(in_channels=channels_list[7],
                                        out_channels=channels_list[8],
                                        kernel_size=1,
                                        stride=1)

        self.Bi_fusion_2 = BiFusion(in_channels=[channels_list[2], channels_list[8]],
                                    out_channels=channels_list[8])

        self.Rep_p3 = RepBlock(in_channels=channels_list[8],
                               out_channels=channels_list[8],
                               n=num_repeats[8],
                               block=block)

        self.down_sample_2 = ConvBNReLU(in_channels=channels_list[8],
                                        out_channels=channels_list[8],
                                        kernel_size=3,
                                        stride=2)

        self.Rep_n4 = RepBlock(in_channels=channels_list[8] + channels_list[8],
                               out_channels=channels_list[9],
                               n=num_repeats[9],
                               block=block)

        self.down_sample_1 = ConvBNReLU(in_channels=channels_list[9],
                                        out_channels=channels_list[9],
                                        kernel_size=3,
                                        stride=2)

        self.Rep_n5 = RepBlock(in_channels=channels_list[7] + channels_list[9],
                               out_channels=channels_list[10],
                               n=num_repeats[10],
                               block=block)

        self.down_sample_0 = ConvBNReLU(in_channels=channels_list[10],
                                        out_channels=channels_list[10],
                                        kernel_size=3,
                                        stride=2)

        self.Rep_n6 = RepBlock(in_channels=channels_list[6] + channels_list[10],
                               out_channels=channels_list[11],
                               n=num_repeats[11],
                               block=block)

    def forward(self, input_list: Tuple[torch.Tensor, ...]) -> List[torch.Tensor]:
        (x4, x3, x2, x1, x0) = input_list

        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bi_fusion_0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p5(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bi_fusion_1([fpn_out1, x2, x3])
        f_out1 = self.Rep_p4(f_concat_layer1)

        fpn_out2 = self.reduce_layer2(f_out1)
        f_concat_layer2 = self.Bi_fusion_2([fpn_out2, x3, x4])
        pan_out3 = self.Rep_p3(f_concat_layer2)

        down_feat2 = self.down_sample_2(pan_out3)
        p_concat_layer2 = torch.cat([down_feat2, fpn_out2], 1)
        pan_out2 = self.Rep_n4(p_concat_layer2)

        down_feat1 = self.down_sample_1(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n5(p_concat_layer1)

        down_feat0 = self.down_sample_0(pan_out1)
        p_concat_layer0 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n6(p_concat_layer0)

        outputs = [pan_out3, pan_out2, pan_out1, pan_out0]
        return outputs


class BepC3(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int,
                 n: int = 1,
                 e: float = 0.5,
                 block=RepVGGBlock):
        super(BepC3, self).__init__()
        c_ = int(out_channels * e)
        self.cv1 = ConvBNReLU(in_channels, c_, 1, 1)
        self.cv2 = ConvBNReLU(in_channels, c_, 1, 1)
        self.cv3 = ConvBNReLU(2 * c_, out_channels, 1, 1)
        if block == ConvBNSiLU:
            self.cv1 = ConvBNSiLU(in_channels, c_, 1, 1)
            self.cv2 = ConvBNSiLU(in_channels, c_, 1, 1)
            self.cv3 = ConvBNSiLU(2 * c_, out_channels, 1, 1)
        self.m = RepBlock(in_channels=c_, out_channels=c_, n=n, block=BottleRep, basic_block=block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class BottleRep3(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int,
                 basic_block: nn.Module = RepVGGBlock,
                 weight: bool = False):
        super(BottleRep3, self).__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        self.conv3 = basic_block(out_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs


class MBLABlock(nn.Module):

    def __init__(self, in_channels: int,
                 out_channels: int,
                 n: int = 1,
                 e: float = 0.5,
                 block=RepVGGBlock):
        super(MBLABlock, self).__init__()
        n = n // 2
        if n <= 0:
            n = 1
        if n == 1:
            n_list = [0, 1]
        else:
            extra_branch_steps = 1
            while extra_branch_steps * 2 < n:
                extra_branch_steps *= 2
            n_list = [0, extra_branch_steps, n]
        branch_num = len(n_list)
        c_ = int(out_channels * e)
        self.c = c_
        self.cv1 = ConvModule(in_channels, branch_num * self.c, 1, 1, 'relu', bias=False)
        self.cv2 = ConvModule((sum(n_list) + branch_num) * self.c, out_channels, 1, 1, 'relu', bias=False)

        if block == ConvBNSiLU:
            self.cv1 = ConvModule(in_channels, branch_num * self.c, 1, 1, 'silu', bias=False)
            self.cv2 = ConvModule((sum(n_list) + branch_num) * self.c, out_channels, 1, 1, 'silu', bias=False)

        self.m = nn.ModuleList()
        for n_list_i in n_list[1:]:
            self.m.append(
                nn.Sequential(*(BottleRep3(self.c, self.c, basic_block=block, weight=True) for _ in range(n_list_i))))

        self.split_num = tuple([self.c] * branch_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split(self.split_num, 1))
        all_y = [y[0]]
        for m_idx, m_i in enumerate(self.m):
            all_y.append(y[m_idx + 1])
            all_y.extend(m(all_y[-1]) for m in m_i)
        return self.cv2(torch.cat(all_y, 1))


class CSPBepBackboneP6(nn.Module):

    def __init__(self, channels_list: List[int],
                 num_repeats: List[int],
                 in_channels: int = 3,
                 block=RepVGGBlock,
                 csp_e: float = float(1) / 2,
                 fuse_P2: bool = False,
                 csp_spp_f: bool = False,
                 stage_block_type: str = "BepC3"):
        super(CSPBepBackboneP6, self).__init__()
        if stage_block_type == "BepC3":
            stage_block = BepC3
        elif stage_block_type == "MBLABlock":
            stage_block = MBLABlock
        else:
            raise NotImplementedError
        self.fuse_P2 = fuse_P2
        self.stem = block(in_channels=in_channels,
                          out_channels=channels_list[0],
                          kernel_size=3,
                          stride=2)

        self.ERBlock_2 = nn.Sequential(block(in_channels=channels_list[0],
                                             out_channels=channels_list[1],
                                             kernel_size=3,
                                             stride=2),
                                       stage_block(
                                           in_channels=channels_list[1],
                                           out_channels=channels_list[1],
                                           n=num_repeats[1],
                                           e=csp_e,
                                           block=block))

        self.ERBlock_3 = nn.Sequential(block(in_channels=channels_list[1],
                                             out_channels=channels_list[2],
                                             kernel_size=3,
                                             stride=2),
                                       stage_block(in_channels=channels_list[2],
                                                   out_channels=channels_list[2],
                                                   n=num_repeats[2],
                                                   e=csp_e,
                                                   block=block))

        self.ERBlock_4 = nn.Sequential(block(in_channels=channels_list[2],
                                             out_channels=channels_list[3],
                                             kernel_size=3,
                                             stride=2),
                                       stage_block(in_channels=channels_list[3],
                                                   out_channels=channels_list[3],
                                                   n=num_repeats[3],
                                                   e=csp_e,
                                                   block=block))

        channel_merge_layer = SPPF if block == ConvBNSiLU else SimSPPF
        if csp_spp_f:
            channel_merge_layer = CSPSPPF if block == ConvBNSiLU else SimCSPSPPF

        self.ERBlock_5 = nn.Sequential(block(in_channels=channels_list[3],
                                             out_channels=channels_list[4],
                                             kernel_size=3,
                                             stride=2),
                                       stage_block(in_channels=channels_list[4],
                                                   out_channels=channels_list[4],
                                                   n=num_repeats[4],
                                                   e=csp_e,
                                                   block=block))
        self.ERBlock_6 = nn.Sequential(block(in_channels=channels_list[4],
                                             out_channels=channels_list[5],
                                             kernel_size=3,
                                             stride=2),
                                       stage_block(in_channels=channels_list[5],
                                                   out_channels=channels_list[5],
                                                   n=num_repeats[5],
                                                   e=csp_e,
                                                   block=block),
                                       channel_merge_layer(
                                           in_channels=channels_list[5],
                                           out_channels=channels_list[5],
                                           kernel_size=5))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)
        x = self.ERBlock_6(x)
        outputs.append(x)

        return tuple(outputs)


class CSPRepBiFPANNeckP6(nn.Module):

    def __init__(self,
                 channels_list: List[int],
                 num_repeats: List[int],
                 block,
                 csp_e: float = float(1) / 2,
                 stage_block_type: str = "BepC3"):
        super(CSPRepBiFPANNeckP6, self).__init__()
        if stage_block_type == "BepC3":
            stage_block = BepC3
        elif stage_block_type == "MBLABlock":
            stage_block = MBLABlock
        else:
            raise NotImplementedError
        self.reduce_layer0 = ConvBNReLU(in_channels=channels_list[5],
                                        out_channels=channels_list[6],
                                        kernel_size=1,
                                        stride=1)

        self.bi_fusion_0 = BiFusion(in_channels=[channels_list[4], channels_list[6]],
                                    out_channels=channels_list[6])

        self.Rep_p5 = stage_block(in_channels=channels_list[6],
                                  out_channels=channels_list[6],
                                  n=num_repeats[6],
                                  e=csp_e,
                                  block=block)

        self.reduce_layer1 = ConvBNReLU(in_channels=channels_list[6],
                                        out_channels=channels_list[7],
                                        kernel_size=1,
                                        stride=1)

        self.bi_fusion_1 = BiFusion(in_channels=[channels_list[3], channels_list[7]],
                                    out_channels=channels_list[7])

        self.Rep_p4 = stage_block(in_channels=channels_list[7],
                                  out_channels=channels_list[7],
                                  n=num_repeats[7],
                                  e=csp_e,
                                  block=block)

        self.reduce_layer2 = ConvBNReLU(in_channels=channels_list[7],
                                        out_channels=channels_list[8],
                                        kernel_size=1,
                                        stride=1)

        self.bi_fusion_2 = BiFusion(in_channels=[channels_list[2], channels_list[8]],
                                    out_channels=channels_list[8])

        self.Rep_p3 = stage_block(in_channels=channels_list[8],
                                  out_channels=channels_list[8],
                                  n=num_repeats[8],
                                  e=csp_e,
                                  block=block)

        self.down_sample_2 = ConvBNReLU(in_channels=channels_list[8],
                                        out_channels=channels_list[8],
                                        kernel_size=3,
                                        stride=2)

        self.Rep_n4 = stage_block(in_channels=channels_list[8] + channels_list[8],
                                  out_channels=channels_list[9],
                                  n=num_repeats[9],
                                  e=csp_e,
                                  block=block)

        self.down_sample_1 = ConvBNReLU(in_channels=channels_list[9],
                                        out_channels=channels_list[9],
                                        kernel_size=3,
                                        stride=2)

        self.Rep_n5 = stage_block(
            in_channels=channels_list[7] + channels_list[9],
            out_channels=channels_list[10],
            n=num_repeats[10],
            e=csp_e,
            block=block)

        self.down_sample_0 = ConvBNReLU(in_channels=channels_list[10],
                                        out_channels=channels_list[10],
                                        kernel_size=3,
                                        stride=2)

        self.Rep_n6 = stage_block(in_channels=channels_list[6] + channels_list[10],
                                  out_channels=channels_list[11],
                                  n=num_repeats[11],
                                  e=csp_e,
                                  block=block)

    def forward(self, _input: Tuple[torch.Tensor, ...]) -> List[torch.Tensor]:
        (x4, x3, x2, x1, x0) = _input
        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.bi_fusion_0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p5(f_concat_layer0)
        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.bi_fusion_1([fpn_out1, x2, x3])
        f_out1 = self.Rep_p4(f_concat_layer1)
        fpn_out2 = self.reduce_layer2(f_out1)
        f_concat_layer2 = self.bi_fusion_2([fpn_out2, x3, x4])
        pan_out3 = self.Rep_p3(f_concat_layer2)
        down_feat2 = self.down_sample_2(pan_out3)
        p_concat_layer2 = torch.cat([down_feat2, fpn_out2], 1)
        pan_out2 = self.Rep_n4(p_concat_layer2)
        down_feat1 = self.down_sample_1(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n5(p_concat_layer1)
        down_feat0 = self.down_sample_0(pan_out1)
        p_concat_layer0 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n6(p_concat_layer0)
        outputs = [pan_out3, pan_out2, pan_out1, pan_out0]
        return outputs


def make_divisible(x: float, divisor: int):
    return math.ceil(x / divisor) * divisor


def create_body(channels_list: List[int],
                num_repeats: List[int],
                in_channels: int,
                block: Type[RepVGGBlock],
                m_type: str):
    assert m_type in ["l", "m", "n", "s"]
    if m_type in ["n", "s"]:
        return EfficientRep6(channels_list=channels_list,
                             num_repeats=num_repeats,
                             in_channels=in_channels,
                             block=block,
                             fuse_P2=True,
                             csp_spp_f=True)
    else:
        csp_e = 0.5 if m_type == "l" else 2.0 / 3
        return CSPBepBackboneP6(channels_list=channels_list,
                                num_repeats=num_repeats,
                                in_channels=in_channels,
                                block=block,
                                csp_e=csp_e,
                                fuse_P2=True,
                                csp_spp_f=False,
                                stage_block_type="BepC3")


def create_neck(channels_list: List[int],
                num_repeats: List[int],
                block: Type[RepVGGBlock],
                m_type: str):
    assert m_type in ["l", "m", "n", "s"]
    if m_type in ["n", "s"]:
        return RepBiFPANNeck6(channels_list=channels_list,
                              num_repeats=num_repeats,
                              block=block)
    else:
        csp_e = 0.5 if m_type == "l" else 2.0 / 3
        return CSPRepBiFPANNeckP6(channels_list=channels_list,
                                  num_repeats=num_repeats,
                                  block=block,
                                  csp_e=csp_e)


def create_head(channels_list: List[int],
                num_classes: int,
                point_num: int,
                reg_max: int = 16,
                num_layers: int = 3):
    chx = [6, 8, 10] if num_layers == 3 else [8, 9, 10, 11]
    num_anchors = 1
    head_layers = nn.Sequential(ConvBNSiLU(in_channels=channels_list[chx[0]],
                                           out_channels=channels_list[chx[0]],
                                           kernel_size=1,
                                           stride=1),
                                ConvBNSiLU(in_channels=channels_list[chx[0]],
                                           out_channels=channels_list[chx[0]],
                                           kernel_size=3,
                                           stride=1),
                                ConvBNSiLU(in_channels=channels_list[chx[0]],
                                           out_channels=channels_list[chx[0]],
                                           kernel_size=3,
                                           stride=1),
                                nn.Conv2d(in_channels=channels_list[chx[0]],
                                          out_channels=num_classes * num_anchors,
                                          kernel_size=(1, 1)),
                                nn.Conv2d(in_channels=channels_list[chx[0]],
                                          out_channels=4 * (reg_max + num_anchors),
                                          kernel_size=(1, 1)),
                                ConvBNSiLU(in_channels=channels_list[chx[0]],
                                           out_channels=channels_list[chx[0]],
                                           kernel_size=3,
                                           stride=1),
                                nn.Conv2d(in_channels=channels_list[chx[0]],
                                          out_channels=2 * point_num,
                                          kernel_size=(1, 1)),

                                # 第1层
                                ConvBNSiLU(in_channels=channels_list[chx[1]],
                                           out_channels=channels_list[chx[1]],
                                           kernel_size=1,
                                           stride=1),
                                ConvBNSiLU(in_channels=channels_list[chx[1]],
                                           out_channels=channels_list[chx[1]],
                                           kernel_size=3,
                                           stride=1),
                                ConvBNSiLU(in_channels=channels_list[chx[1]],
                                           out_channels=channels_list[chx[1]],
                                           kernel_size=3,
                                           stride=1),
                                nn.Conv2d(in_channels=channels_list[chx[1]],
                                          out_channels=num_classes * num_anchors,
                                          kernel_size=(1, 1)),
                                nn.Conv2d(in_channels=channels_list[chx[1]],
                                          out_channels=4 * (reg_max + num_anchors),
                                          kernel_size=(1, 1)),
                                ConvBNSiLU(in_channels=channels_list[chx[1]],
                                           out_channels=channels_list[chx[1]],
                                           kernel_size=3,
                                           stride=1),
                                nn.Conv2d(in_channels=channels_list[chx[1]],
                                          out_channels=2 * point_num,
                                          kernel_size=(1, 1)),
                                # 第2层
                                ConvBNSiLU(in_channels=channels_list[chx[2]],
                                           out_channels=channels_list[chx[2]],
                                           kernel_size=1,
                                           stride=1),
                                ConvBNSiLU(in_channels=channels_list[chx[2]],
                                           out_channels=channels_list[chx[2]],
                                           kernel_size=3,
                                           stride=1),
                                ConvBNSiLU(in_channels=channels_list[chx[2]],
                                           out_channels=channels_list[chx[2]],
                                           kernel_size=3,
                                           stride=1),
                                nn.Conv2d(in_channels=channels_list[chx[2]],
                                          out_channels=num_classes * num_anchors,
                                          kernel_size=(1, 1)),
                                nn.Conv2d(in_channels=channels_list[chx[2]],
                                          out_channels=4 * (reg_max + num_anchors),
                                          kernel_size=(1, 1)),
                                ConvBNSiLU(in_channels=channels_list[chx[2]],
                                           out_channels=channels_list[chx[2]],
                                           kernel_size=3,
                                           stride=1),
                                nn.Conv2d(in_channels=channels_list[chx[2]],
                                          out_channels=2 * point_num,
                                          kernel_size=(1, 1)))
    if num_layers == 4:
        head_layers.add_module('stem3', ConvBNSiLU(in_channels=channels_list[chx[3]],
                                                   out_channels=channels_list[chx[3]],
                                                   kernel_size=1,
                                                   stride=1))
        head_layers.add_module('cls_conv3', ConvBNSiLU(in_channels=channels_list[chx[3]],
                                                       out_channels=channels_list[chx[3]],
                                                       kernel_size=3,
                                                       stride=1))
        head_layers.add_module('reg_conv3', ConvBNSiLU(in_channels=channels_list[chx[3]],
                                                       out_channels=channels_list[chx[3]],
                                                       kernel_size=3,
                                                       stride=1))
        head_layers.add_module('cls_pred3', nn.Conv2d(in_channels=channels_list[chx[3]],
                                                      out_channels=num_classes * num_anchors,
                                                      kernel_size=(1, 1)))
        head_layers.add_module('reg_pred3', nn.Conv2d(in_channels=channels_list[chx[3]],
                                                      out_channels=4 * (reg_max + num_anchors),
                                                      kernel_size=(1, 1)))
        head_layers.add_module('reg_point_conv3', ConvBNSiLU(in_channels=channels_list[chx[3]],
                                                             out_channels=channels_list[chx[3]],
                                                             kernel_size=3,
                                                             stride=1))
        head_layers.add_module('reg_point_pred3', nn.Conv2d(in_channels=channels_list[chx[3]],
                                                            out_channels=2 * point_num,
                                                            kernel_size=(1, 1)))

    return head_layers


def build_network(m_type: str,
                  point_num: int,
                  num_classes: int,
                  num_layers: Union[int, None] = None):
    assert m_type in ["s", "n", "m", "l"]
    config = _config[m_type]
    depth_mul = config["depth_multiple"]
    width_mul = config["width_multiple"]
    num_repeat_backbone = config["backbone"]["num_repeats"]
    channels_list_backbone = config["backbone"]["out_channels"]
    num_repeat_neck = config["neck"]["num_repeats"]
    channels_list_neck = config["neck"]["out_channels"]
    use_dfl = config["head"]["use_dfl"]
    reg_max = config["head"]["reg_max"]
    if num_layers is None:
        num_layers = config["head"]["num_layers"]

    num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]

    if m_type in ["s", "n", "m"]:
        block = RepVGGBlock
    else:
        block = ConvBNSiLU

    back_bone_network = create_body(channels_list=channels_list,
                                    num_repeats=num_repeat,
                                    in_channels=3,
                                    block=block,
                                    m_type=m_type)

    neck_network = create_neck(channels_list=channels_list,
                               num_repeats=num_repeat,
                               block=block,
                               m_type=m_type)

    head_network = create_head(channels_list=channels_list,
                               num_classes=num_classes,
                               point_num=point_num,
                               reg_max=reg_max,
                               num_layers=num_layers)

    return back_bone_network, neck_network, head_network, reg_max


class YoloV6FaceHead(nn.Module):

    def __init__(self,
                 num_classes: int,
                 point_num: int,
                 head_layers: Union[None, nn.Sequential] = None,
                 num_layers: int = 3,
                 inplace: bool = True,
                 use_dfl: bool = False,
                 reg_max: int = 16):
        super(YoloV6FaceHead, self).__init__()

        self.num_classes = num_classes
        self.point_num = point_num
        self.num_output = num_classes + 4
        self.num_layers = num_layers
        self.num_anchor = 1

        self.inplace = inplace
        self.use_dfl = use_dfl
        self.reg_max = reg_max

        self.grid = [torch.zeros(1)] * num_layers
        self.prior_prob = 1e-2
        self.inplace = inplace
        stride = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64]  # strides computed during build
        self.stride = stride
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        # self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, kernel_size=(1, 1), bias=False)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0  # 固定的大小 相当于 anchor 尺寸

        self.stems = nn.ModuleList()

        self.cls_conv_list = nn.ModuleList()
        self.reg_conv_list = nn.ModuleList()
        self.reg_point_conv_list = nn.ModuleList()

        self.cls_predict_list = nn.ModuleList()
        self.reg_predict_list = nn.ModuleList()
        self.reg_point_predict_list = nn.ModuleList()

        for i in range(num_layers):
            idx = i * 7
            self.stems.append(head_layers[idx])
            self.cls_conv_list.append(head_layers[idx + 1])
            self.reg_conv_list.append(head_layers[idx + 2])

            self.cls_predict_list.append(head_layers[idx + 3])
            self.reg_predict_list.append(head_layers[idx + 4])

            self.reg_point_conv_list.append(head_layers[idx + 5])
            self.reg_point_predict_list.append(head_layers[idx + 6])

        self.proj = NotImplemented

        self.initialize_biases()

    def initialize_biases(self):

        for conv in self.cls_predict_list:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1),
                                           requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w,
                                             requires_grad=True)

        for conv in self.reg_predict_list:
            b = conv.bias.view(-1, )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1),
                                           requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w,
                                             requires_grad=True)

        self.proj = nn.Parameter(torch.linspace(0,
                                                self.reg_max,
                                                self.reg_max + 1),
                                 requires_grad=False)

    def __generate_train_anchors__(self, features: List[torch.Tensor]):
        """
        训练过程 使用的 anchor
        :param features:
        :return:
        """

        anchors_xx_yy = []
        anchor_points_xy = []
        num_anchors_list = []
        stride_tensor_list = []
        device = features[0].device
        for i, stride in enumerate(self.stride):
            _, _, h, w = features[i].size()
            cell_half_size = self.grid_cell_size * stride * 0.5

            shift_x = (torch.arange(end=w, device=device) + self.grid_cell_offset) * stride

            shift_y = (torch.arange(end=h, device=device) + self.grid_cell_offset) * stride

            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')

            anchor_xx_yy = torch.stack([shift_x - cell_half_size, shift_y - cell_half_size,
                                        shift_x + cell_half_size, shift_y + cell_half_size],
                                       dim=-1).clone().to(features[0].dtype)

            anchor_point = torch.stack([shift_x, shift_y], dim=-1).clone().to(features[0].dtype)
            anchors_xx_yy.append(anchor_xx_yy.reshape([-1, 4]))
            anchor_points_xy.append(anchor_point.reshape([-1, 2]))
            num_anchors_list.append(anchors_xx_yy[-1].size(0))
            stride_tensor_list.append(torch.full([num_anchors_list[-1], 1], stride, dtype=features[i].dtype))

        anchors_xx_yy = torch.cat(anchors_xx_yy)
        anchor_points_xy = torch.cat(anchor_points_xy).to(device)
        stride_tensor = torch.cat(stride_tensor_list).to(device)

        return anchors_xx_yy, anchor_points_xy, num_anchors_list, stride_tensor

    def bbox_decode(self,
                    scale_anchor_points_x_y: torch.Tensor,
                    reg_distribute_tensor: torch.Tensor) -> torch.Tensor:
        """

        :param scale_anchor_points_x_y:
        :param reg_distribute_tensor:
        :return:
        """
        if self.use_dfl:
            batch_size, n_anchors, _ = reg_distribute_tensor.shape
            # 回归坐标 变成 加权求平均
            # 计算 softmax [batch, anchor_num, 4, 17] 计算权重
            # 加权求平均 计算最终值
            pred_dist = F.softmax(reg_distribute_tensor.view(batch_size,
                                                             n_anchors,
                                                             4,
                                                             self.reg_max + 1),
                                  dim=-1).matmul(self.proj.to(reg_distribute_tensor.device))
        else:
            pred_dist = reg_distribute_tensor
        # left top right bottom
        lt, rb = torch.split(pred_dist, 2, -1)

        x1y1 = scale_anchor_points_x_y - lt
        x2y2 = scale_anchor_points_x_y + rb

        return torch.cat([x1y1, x2y2], dim=-1)

    def point_decode(self,
                     scale_anchor_points_x_y: torch.Tensor,
                     reg_point_distribute_tensor: torch.Tensor):
        """

        :param scale_anchor_points_x_y:
        :param reg_point_distribute_tensor:
        :return:
        """
        x = reg_point_distribute_tensor[:, :, 0::2] + scale_anchor_points_x_y[:, 0:1]
        y = reg_point_distribute_tensor[:, :, 1::2] + scale_anchor_points_x_y[:, 1:2]
        x = torch.unsqueeze(x, dim=-1)
        y = torch.unsqueeze(y, dim=-1)
        xy = torch.cat((x, y), dim=-1)
        return xy.view(xy.size(0), xy.size(1), xy.size(2) * xy.size(3))

    def forward(self, x: List[torch.Tensor]):
        cls_score_list = []
        reg_distribute_list = []
        reg_point_distribute_list = []
        for i in range(self.num_layers):
            _, _, h, w = x[i].size()
            x[i] = self.stems[i](x[i])
            cls_x = x[i]
            reg_x = x[i]
            point_x = x[i]
            cls_feat = self.cls_conv_list[i](cls_x)
            # [b, class_num, h, w]
            cls_output = self.cls_predict_list[i](cls_feat)

            reg_feat = self.reg_conv_list[i](reg_x)
            # [b, 4 * (reg_max + 1) + 2 * point_num, h, w]
            reg_output = self.reg_predict_list[i](reg_feat)

            reg_point_feat = self.reg_point_conv_list[i](point_x)
            reg_point_output = self.reg_point_predict_list[i](reg_point_feat)

            # 类别预测 [batch, class_num, w, h]
            cls_output = torch.sigmoid(cls_output)

            # [batch, w*h, class_num]
            cls_score_list.append(cls_output.view(cls_output.size(0), self.num_classes, h * w)
                                  .permute(0, 2, 1))
            #  [b, 4 * (reg_max + 1) , h, w] ->  [b, 4 * (reg_max + 1), h*w]-> [batch, h*w, 4 * (reg_max + 1)]
            if self.use_dfl:
                reg_bbox = reg_output.view(reg_output.size(0), 4 * (self.reg_max + 1), h * w).permute(0, 2, 1)
            else:
                reg_bbox = reg_output.view(reg_output.size(0), 4, h * w).permute(0, 2, 1)

            #  [b, 2 * point_num , h, w] ->  [b, 2 * point_num, h*w]-> [batch, h*w, 2 * point_num]
            reg_point = reg_point_output.view(reg_point_output.size(0), 2 * self.point_num, h * w).permute(0, 2, 1)

            reg_distribute_list.append(reg_bbox)

            reg_point_distribute_list.append(reg_point)

        anchors_xx_yy, anchor_points_xy, num_anchors_list, stride_tensor = \
            self.__generate_train_anchors__(features=x)

        cls_score_tensor = torch.cat(cls_score_list, dim=1)
        reg_distribute_tensor = torch.cat(reg_distribute_list, dim=1)
        reg_point_distribute_tensor = torch.cat(reg_point_distribute_list, dim=1)

        scale_anchor_points_x_y = anchor_points_xy / stride_tensor
        scale_predict_xy_xy = self.bbox_decode(scale_anchor_points_x_y=scale_anchor_points_x_y,
                                               reg_distribute_tensor=reg_distribute_tensor)

        scale_point_predict_xy = self.point_decode(scale_anchor_points_x_y=scale_anchor_points_x_y,
                                                   reg_point_distribute_tensor=reg_point_distribute_tensor)
        point_predict_xy = scale_point_predict_xy * stride_tensor
        predict_xy_xy = scale_predict_xy_xy * stride_tensor
        return anchors_xx_yy, anchor_points_xy, num_anchors_list, \
            stride_tensor, cls_score_tensor, predict_xy_xy, point_predict_xy, reg_distribute_tensor


class YoloV6FaceBody(nn.Module):

    def __init__(self, param: Yolo6FaceParam):
        super(YoloV6FaceBody, self).__init__()
        self.param = param
        back_bone_network, neck_network, head_network, reg_max = build_network(m_type=param.m_type,
                                                                               point_num=param.point_num,
                                                                               num_classes=param.class_num,
                                                                               num_layers=param.num_layers)
        self.back_bone_network = back_bone_network
        self.neck_network = neck_network

        self.head = YoloV6FaceHead(num_classes=param.class_num,
                                   num_layers=param.num_layers,
                                   head_layers=head_network,
                                   point_num=param.point_num,
                                   use_dfl=param.m_type in ["m", "l"])

        # 初始化权重
        self._initialize_weight()

    def _initialize_weight(self):
        for m in self.modules():
            _type = type(m)
            if _type is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif _type in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """

        :param x:
        :return:
        """
        bone_features = self.back_bone_network(x)
        neck_features = self.neck_network(bone_features)
        return neck_features
