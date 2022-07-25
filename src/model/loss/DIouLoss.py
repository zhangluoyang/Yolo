import torch
import torch.nn as nn
import torchvision


class DIouLoss(nn.Module):

    def __init__(self, wh_to_xy: bool = True):
        super(DIouLoss, self).__init__()
        self.eps = 1e-7
        self.wh_to_xy = wh_to_xy

    def forward(self, predict: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
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

        iw = (ix2 - ix1 + 1.0).clamp(min=0.)
        ih = (iy2 - iy1 + 1.0).clamp(min=0.)

        # overlaps
        inters = iw * ih

        # union
        uni = (predict[..., 2] - predict[..., 0] + 1.0) * (predict[..., 3] - predict[..., 1] + 1.0) + \
              (target[..., 2] - target[..., 0] + 1.0) * (target[..., 3] - target[..., 1] + 1.0) - inters

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

        # [batch_size, ]
        loss = torch.mean(d_iou_loss, dim=-1)
        return loss
