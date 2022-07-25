import torch
import torch.nn as nn
import torchvision
from typing import *


class GIouLoss(nn.Module):

    def __init__(self, wh_to_xy: bool = True,
                 return_g_iou: bool = False):
        """

        :param wh_to_xy:
        :param return_g_iou
        """
        super(GIouLoss, self).__init__()
        self.eps = 1e-7
        self.wh_to_xy = wh_to_xy
        self.return_g_iou = return_g_iou

    def forward(self, predict: torch.Tensor,
                target: torch.Tensor) -> Union[torch.Tensor,
                                               Tuple[torch.Tensor, torch.Tensor]]:
        """

        :param predict:
        :param target:
        :return:
        """
        if self.wh_to_xy:
            predict = torchvision.ops.box_convert(boxes=predict,
                                                  in_fmt="cxcywh",
                                                  out_fmt="xyxy")
            target = torchvision.ops.box_convert(boxes=target,
                                                 in_fmt="cxcywh",
                                                 out_fmt="xyxy")

        ix1 = torch.max(predict[..., 0], target[..., 0])
        iy1 = torch.max(predict[..., 1], target[..., 1])
        ix2 = torch.min(predict[..., 2], target[..., 2])
        iy2 = torch.min(predict[..., 3], target[..., 3])

        iw = (ix2 - ix1 + 1.0).clamp(0.)
        ih = (iy2 - iy1 + 1.0).clamp(0.)

        # overlap
        inters = iw * ih
        # union
        uni = (predict[..., 2] - predict[..., 0] + 1.0) * (predict[..., 3] - predict[..., 1] + 1.0) + (
                target[..., 2] - target[..., 0] + 1.0) * (
                      target[..., 3] - target[..., 1] + 1.0) - inters + self.eps
        # ious
        ious = inters / uni

        ex1 = torch.min(predict[..., 0], target[..., 0])
        ey1 = torch.min(predict[..., 1], target[..., 1])
        ex2 = torch.max(predict[..., 2], target[..., 2])
        ey2 = torch.max(predict[..., 3], target[..., 3])
        ew = (ex2 - ex1 + 1.0).clamp(min=0.)
        eh = (ey2 - ey1 + 1.0).clamp(min=0.)

        enclose = ew * eh + self.eps
        giou = ious - (enclose - uni) / enclose
        loss = 1 - giou
        if self.return_g_iou:
            return loss, giou
        else:
            return torch.mean(loss)
