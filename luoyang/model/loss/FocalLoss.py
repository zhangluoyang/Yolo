import torch
from typing import *
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 0.75,
                 gamma: float = 2.0,
                 reduction: str = "none"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred_score: torch.Tensor,
                gt_score: torch.Tensor,
                label: torch.Tensor):
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy(pred_score.float(), gt_score.float(), reduction='none') * weight)
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        else:
            raise NotImplemented

        return loss
