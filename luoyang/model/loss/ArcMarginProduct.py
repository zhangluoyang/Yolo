import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int,
                 label_size: int,
                 scale: float = 30.0,
                 margin: float = 0.50,
                 easy_margin: bool = False,
                 feature_norm: bool = False,
                 ls_eps: float = 0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.label_size = label_size
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps
        self.feature_norm = feature_norm
        self.weight = nn.Parameter(torch.FloatTensor(label_size, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.loss = nn.CrossEntropyLoss()

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, feature: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if self.feature_norm:
            feature = F.normalize(feature)
        cosine = F.linear(feature, F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=feature.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return self.loss(output, label)
