import torch
import torch.nn as nn
from abc import ABC


class Mish(nn.Module, ABC):
    """
    mish 激活函数
    """

    def __init__(self):
        super(Mish, self).__init__()
        self.tanh = nn.Tanh()
        self.soft_plus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x:
        :return:
        """
        x = x * (self.tanh(self.soft_plus(x)))
        return x


class Silu(nn.Module, ABC):
    """
    swish 激活函数
    """

    def __init__(self):
        super(Silu, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x:
        :return:
        """
        x = x * self.sigmoid(x)
        return x


def get_activate(activate: str) -> nn.Module:
    if activate == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    elif activate == "mish":
        return Mish()
    elif activate == "silu":
        return Silu()
    else:
        raise NotImplemented
