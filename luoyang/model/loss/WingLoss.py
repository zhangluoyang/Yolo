import torch
import math
import torch.nn as nn


class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)

    def forward(self, predict, target):
        y = target
        y_hat = predict
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        loss2 = delta_y2 - self.C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2) + 1e-3)
