import torch
import numpy as np
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
    def __init__(self, c1: int,
                 c2: int,
                 k: int = 1,
                 s: int = 1,
                 p: Union[int, None] = None,
                 g: int = 1,
                 act=SiLU()):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=c1,
                              out_channels=c2,
                              kernel_size=(k, k),
                              stride=(s, s),
                              padding=_auto_pad(k, p),
                              groups=g,
                              bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def fuse_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class MultiConcatBlock(nn.Module):
    def __init__(self, c1: int, c2: int, c3: int, n: int = 4, e: int = 1, ids=None):
        super(MultiConcatBlock, self).__init__()
        if ids is None:
            ids = [0]
        c_ = int(c2 * e)

        self.ids = ids
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = nn.ModuleList([Conv(c_ if i == 0 else c2, c2, 3, 1) for i in range(n)])
        self.cv4 = Conv(c_ * 2 + c2 * (len(ids) - 2), c3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = self.cv1(x)
        x_2 = self.cv2(x)

        x_all = [x_1, x_2]
        for i in range(len(self.cv3)):
            x_2 = self.cv3[i](x_2)
            x_all.append(x_2)
        out = self.cv4(torch.cat([x_all[_id] for _id in self.ids], dim=1))
        return out


class MP(nn.Module):
    def __init__(self, k: int = 2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)


class TransitionBlock(nn.Module):
    def __init__(self, c1: int, c2: int):
        super(TransitionBlock, self).__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.cv3 = Conv(c2, c2, 3, 2)
        self.mp = MP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = self.mp(x)
        x_1 = self.cv1(x_1)

        x_2 = self.cv2(x)
        x_2 = self.cv3(x_2)

        return torch.cat([x_2, x_1], 1)


class Backbone(nn.Module):
    def __init__(self, transition_channels: int,
                 block_channels: int,
                 n: int,
                 phi: str):
        super().__init__()
        ids = {'l': [-1, -3, -5, -6],
               'x': [-1, -3, -5, -7, -8]}[phi]

        self.stem = nn.Sequential(Conv(3, transition_channels, 3, 1),
                                  Conv(transition_channels, transition_channels * 2, 3, 2),
                                  Conv(transition_channels * 2, transition_channels * 2, 3, 1))

        self.dark2 = nn.Sequential(Conv(transition_channels * 2, transition_channels * 4, 3, 2),
                                   MultiConcatBlock(transition_channels * 4, block_channels * 2,
                                                    transition_channels * 8, n=n, ids=ids))
        self.dark3 = nn.Sequential(
            TransitionBlock(transition_channels * 8, transition_channels * 4),
            MultiConcatBlock(transition_channels * 8, block_channels * 4, transition_channels * 16, n=n, ids=ids))

        self.dark4 = nn.Sequential(
            TransitionBlock(transition_channels * 16, transition_channels * 8),
            MultiConcatBlock(transition_channels * 16, block_channels * 8, transition_channels * 32, n=n, ids=ids))

        self.dark5 = nn.Sequential(
            TransitionBlock(transition_channels * 32, transition_channels * 16),
            MultiConcatBlock(transition_channels * 32, block_channels * 8, transition_channels * 32, n=n, ids=ids))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param x:
        :return:
        """
        x = self.stem(x)
        x = self.dark2(x)
        # 8倍下采样特征
        x = self.dark3(x)
        feat1 = x
        # 16倍下采样特征
        x = self.dark4(x)
        feat2 = x
        # 32倍下采样特征
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3


def create_body(transition_channels: int,
                block_channels: int,
                n: int,
                phi: str,
                pretrained_path: str):
    """

    :param transition_channels:
    :param block_channels:
    :param n:
    :param phi
    :param pretrained_path
    :return:
    """
    net = Backbone(transition_channels=transition_channels,
                   block_channels=block_channels,
                   n=n,
                   phi=phi)
    if pretrained_path is not None and file_utils.file_is_exists(path=pretrained_path):
        print("pretrained_path{0}".format(pretrained_path))
        net.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
    return net


class SppCspC(nn.Module):
    def __init__(self, c1: int, c2: int, e: float = 0.5, k: Tuple[int, int, int] = (5, 9, 13)):
        super(SppCspC, self).__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class RepConv(nn.Module):
    def __init__(self, c1: int,
                 c2: int,
                 k: int = 3,
                 s: int = 1,
                 p=None,
                 g: int = 1,
                 act=SiLU(),
                 deploy: bool = False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert _auto_pad(k, p) == 1

        padding_11 = _auto_pad(k, p) - k // 2
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else \
            (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=c1,
                                         out_channels=c2,
                                         kernel_size=(k, k),
                                         stride=(s, s),
                                         padding=_auto_pad(k, p),
                                         groups=g,
                                         bias=True)
        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=c1, eps=0.001, momentum=0.03) if c2 == c1 and s == 1 else None)
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_channels=c1,
                          out_channels=c2,
                          kernel_size=(k, k),
                          stride=(s, s),
                          padding=_auto_pad(k, p),
                          groups=g,
                          bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03))
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_channels=c1,
                          out_channels=c2,
                          kernel_size=(1, 1),
                          stride=(s, s),
                          padding=padding_11,
                          groups=g,
                          bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel_id, bias_id = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernel_id,
            bias3x3 + bias1x1 + bias_id,
        )

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
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

    @staticmethod
    def fuse_conv_bn(conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        conv = nn.Conv2d(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=True,
                         padding_mode=conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_rep_vgg_block(self):
        """
        重参化处理: 训练阶段  1*1 3*3 bn 三部分相加
                  预测阶段 合成一个 3*3 的卷积
        :return:
        """
        if self.deploy:
            return
        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])

        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or
                isinstance(self.rbr_identity,
                           nn.modules.batchnorm.SyncBatchNorm)):
            identity_conv_1x1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0,
                groups=self.groups,
                bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
            weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))

        self.rbr_dense.weight = torch.nn.Parameter(
            self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)

        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


def _fuse_conv_and_bn(conv: nn.Module, bn: nn.Module):
    fused_conv = nn.Conv2d(conv.in_channels,
                           conv.out_channels,
                           kernel_size=conv.kernel_size,
                           stride=conv.stride,
                           padding=conv.padding,
                           groups=conv.groups,
                           bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused_conv.weight.copy_(torch.mm(w_bn, w_conv).view(fused_conv.weight.shape))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_conv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fused_conv


class YoloV7FaceBody(nn.Module):
    def __init__(self, anchor_num: int,
                 num_classes: int,
                 point_num: int,
                 phi: str,
                 pretrained_path: str):
        super(YoloV7FaceBody, self).__init__()
        self.point_num = point_num
        transition_channels = {'l': 32, 'x': 40}[phi]
        block_channels = 32
        planet_channels = {'l': 32, 'x': 64}[phi]
        e = {'l': 2, 'x': 1}[phi]
        n = {'l': 4, 'x': 6}[phi]
        ids = {'l': [-1, -2, -3, -4, -5, -6], 'x': [-1, -3, -5, -7, -8]}[phi]
        conv = {'l': RepConv, 'x': Conv}[phi]

        self.backbone = create_body(transition_channels=transition_channels,
                                    block_channels=block_channels,
                                    n=n,
                                    phi=phi,
                                    pretrained_path=pretrained_path)

        self.up_sample = nn.Upsample(scale_factor=2, mode="nearest")

        self.spp_csp_c = SppCspC(transition_channels * 32, transition_channels * 16)
        self.conv_for_P5 = Conv(transition_channels * 16, transition_channels * 8)
        self.conv_for_feat2 = Conv(transition_channels * 32, transition_channels * 8)
        self.conv3_for_up_sample1 = MultiConcatBlock(transition_channels * 16, planet_channels * 4,
                                                     transition_channels * 8, e=e, n=n, ids=ids)

        self.conv_for_P4 = Conv(transition_channels * 8, transition_channels * 4)
        self.conv_for_feat1 = Conv(transition_channels * 16, transition_channels * 4)
        self.conv3_for_up_sample2 = MultiConcatBlock(transition_channels * 8, planet_channels * 2,
                                                     transition_channels * 4, e=e, n=n, ids=ids)

        self.down_sample1 = TransitionBlock(transition_channels * 4, transition_channels * 4)
        self.conv3_for_down_sample1 = MultiConcatBlock(transition_channels * 16, planet_channels * 4,
                                                       transition_channels * 8, e=e, n=n, ids=ids)

        self.down_sample2 = TransitionBlock(transition_channels * 8, transition_channels * 8)
        self.conv3_for_down_sample2 = MultiConcatBlock(transition_channels * 32, planet_channels * 8,
                                                       transition_channels * 16, e=e, n=n, ids=ids)

        self.rep_conv_1 = conv(transition_channels * 4, transition_channels * 8, 3, 1)
        self.rep_conv_2 = conv(transition_channels * 8, transition_channels * 16, 3, 1)
        self.rep_conv_3 = conv(transition_channels * 16, transition_channels * 32, 3, 1)

        output_size = 5 + point_num * 2 + num_classes

        self.yolo_head_P3 = nn.Conv2d(in_channels=transition_channels * 8,
                                      out_channels=output_size * anchor_num,
                                      kernel_size=(1, 1))
        self.yolo_head_P4 = nn.Conv2d(in_channels=transition_channels * 16,
                                      out_channels=output_size * anchor_num,
                                      kernel_size=(1, 1))
        self.yolo_head_P5 = nn.Conv2d(in_channels=transition_channels * 32,
                                      out_channels=output_size * anchor_num,
                                      kernel_size=(1, 1))

    def fuse(self):
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse_rep_vgg_block()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = _fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuse_forward
        return self

    def re_eval(self) -> nn.Module:
        """
        重写 eval 方法 保证先重惨化 由后续过程 显示调用
        :return:
        """
        self.fuse()
        return self.eval()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat1, feat2, feat3 = self.backbone.forward(x)

        p5 = self.spp_csp_c(feat3)
        p5_conv = self.conv_for_P5(p5)
        p5_up_sample = self.up_sample(p5_conv)
        p4 = torch.cat([self.conv_for_feat2(feat2), p5_up_sample], 1)
        p4 = self.conv3_for_up_sample1(p4)

        p4_conv = self.conv_for_P4(p4)
        p4_up_sample = self.up_sample(p4_conv)
        p3 = torch.cat([self.conv_for_feat1(feat1), p4_up_sample], 1)
        p3 = self.conv3_for_up_sample2(p3)

        p3_down_sample = self.down_sample1(p3)
        p4 = torch.cat([p3_down_sample, p4], 1)
        p4 = self.conv3_for_down_sample1(p4)

        p4_down_sample = self.down_sample2(p4)
        p5 = torch.cat([p4_down_sample, p5], 1)
        p5 = self.conv3_for_down_sample2(p5)

        p3 = self.rep_conv_1(p3)
        p4 = self.rep_conv_2(p4)
        p5 = self.rep_conv_3(p5)

        out2 = self.yolo_head_P3(p3)
        out1 = self.yolo_head_P4(p4)
        out0 = self.yolo_head_P5(p5)

        return out0, out1, out2
