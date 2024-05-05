import torch
import torch.nn as nn
import torchvision
from typing import *


class DIouLoss(nn.Module):

    def __init__(self,
                 reduction: str = 'none',
                 wh_to_xy: bool = True,
                 return_d_iou: bool = False):
        super(DIouLoss, self).__init__()
        self.eps = 1e-7
        self.reduction = reduction
        self.wh_to_xy = wh_to_xy
        self.return_d_iou = return_d_iou

    def forward(self, predict: torch.Tensor,
                target: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """

        :param predict: [[min_x, min_y, max_x, max_y],......]
        :param target: [[min_x, min_y, max_x, max_y],......]
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

        iw = (ix2 - ix1).clamp(min=0.)
        ih = (iy2 - iy1).clamp(min=0.)

        # overlaps
        inters = iw * ih

        # union
        uni = (predict[..., 2] - predict[..., 0]) * (predict[..., 3] - predict[..., 1]) + \
              (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1]) - inters

        # iou
        iou = inters / (uni + self.eps)

        # inter_diag
        cx_predict = (predict[..., 2] + predict[..., 0]) / 2
        cy_predict = (predict[..., 3] + predict[..., 1]) / 2

        cx_target = (target[..., 2] + target[..., 0]) / 2
        cy_target = (target[..., 3] + target[..., 1]) / 2

        inter_diag = (cx_target - cx_predict) ** 2 + (cy_target - cy_predict) ** 2

        # outer_diag
        ox1 = torch.min(predict[..., 0], target[..., 0])
        oy1 = torch.min(predict[..., 1], target[..., 1])
        ox2 = torch.max(predict[..., 2], target[..., 2])
        oy2 = torch.max(predict[..., 3], target[..., 3])

        outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2

        d_iou = iou - inter_diag / outer_diag
        d_iou = torch.clamp(d_iou, min=-1.0, max=1.0)

        d_iou_loss = 1 - d_iou

        loss = torch.mean(d_iou_loss, dim=-1)

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        if self.return_d_iou:
            return loss, d_iou
        else:
            return loss
