import math
import torch
import torch.nn as nn
import torchvision


class CIouLoss(nn.Module):

    def __init__(self, wh_to_xy: bool = True):
        super(CIouLoss, self).__init__()
        self.eps = 1e-7
        self.wh_to_xy = wh_to_xy

    def forward(self, predict: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """

        :param predict:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
        :param target:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
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

        iw = (ix2 - ix1 + 1.0).clamp(min=0.)
        ih = (iy2 - iy1 + 1.0).clamp(min=0.)

        # overlaps
        inters = iw * ih

        # union
        uni = (predict[..., 2] - predict[..., 0] + 1.0) * (predict[..., 3] - predict[..., 1] + 1.0) + (
                target[..., 2] - target[..., 0] + 1.0) * (
                      target[..., 3] - target[..., 1] + 1.0) - inters

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

        # calculate v,alpha
        w_target = target[..., 2] - target[..., 0] + 1.0
        h_target = target[..., 3] - target[..., 1] + 1.0
        w_predict = predict[..., 2] - predict[..., 0] + 1.0
        h_predict = predict[..., 3] - predict[..., 1] + 1.0
        v = torch.pow((torch.atan(w_target / h_target) - torch.atan(w_predict / h_predict)), 2) * (4 / (math.pi ** 2))
        alpha = v / (1 - iou + v)
        c_iou = d_iou - alpha * v
        c_iou = torch.clamp(c_iou, min=-1.0, max=1.0)

        c_iou_loss = 1 - c_iou

        return torch.mean(c_iou_loss)
