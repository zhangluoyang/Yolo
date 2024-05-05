import torch
import numpy as np
from typing import *
import torch.nn as nn


class YoloCandidateTarget(nn.Module):
    """
    yolo 中 找出候选的 target yolo v5 pose
    """

    def __init__(self,
                 anchors: List[Tuple[int, int]],
                 stride: int,
                 anchor_num: int = 3,
                 point_num: int = 17):
        super(YoloCandidateTarget, self).__init__()

        self.anchors = np.array(anchors, dtype=np.float32)
        self.stride = stride
        self.anchor_num = anchor_num
        self.point_num = point_num
        self.threshold: float = 4.0

    def forward(self, prediction: torch.Tensor,
                targets: torch.Tensor):
        """
        :param prediction [b, anchor, height, width, 4 + 1 + 2 * point_num + point_num + class_num]
        :param targets: shape=(?, 16)  [[image_id, c, x, y, w, h, point1_x, point1_y, ....]]
        :return:
        """
        num_gt = targets.shape[0]
        if num_gt:
            # 41 = im_id, class_id, x, y, w, h, 2 * 17, anchor_id
            gain = torch.ones(6 + 2 * self.point_num + 1, device=targets.device)
            ai = torch.arange(self.anchor_num, device=targets.device).float() \
                .view(self.anchor_num, 1).repeat(1, num_gt)
            # [3, g_num, 1 + 1 + 4 + point_num * 2 + 1] 最后的1表示anchor_id
            targets = torch.cat((targets.repeat(self.anchor_num, 1, 1), ai[:, :, None]), 2)

            g = 0.5
            off = torch.tensor([[0, 0],  # 中间
                                [1, 0],  # 右边
                                [0, 1],  # 下边
                                [-1, 0],  # 左边
                                [0, -1]],  # 上边
                               device=targets.device).float() * g

            # anchor 位于当前特征层的尺寸
            anchors_i = torch.from_numpy(self.anchors / self.stride).type_as(prediction)
            shape = prediction.shape
            # ground_true [w, h, w, h]
            gain[2:6] = torch.tensor(prediction.shape)[[3, 2, 3, 2]]
            # key point
            gain[6:6 + 2 * self.point_num] = torch.tensor(prediction.shape)[[3, 2] * self.point_num]
            # 最终目标在坐在特征层中坐标
            target_in_feature_layer = targets * gain

            r = target_in_feature_layer[:, :, 4:6] / anchors_i[:, None]
            j = torch.max(r, 1. / r).max(2)[0] < self.threshold
            target_in_feature_layer = target_in_feature_layer[j]
            # ground_true x y 坐标
            gxy = target_in_feature_layer[:, 2:4]
            # 计算相对于 左上角的偏移量
            gxi = gain[[2, 3]] - gxy
            # 判断是第几象限 并找到符合条件
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            target_in_feature_layer = target_in_feature_layer.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            b, c = target_in_feature_layer[:, :2].long().T
            gxy = target_in_feature_layer[:, 2:4]
            gwh = target_in_feature_layer[:, 4:6]
            g_points = target_in_feature_layer[:, 6: -1]
            gij = (gxy - offsets).long()
            gi, gj = gij.T
            a = target_in_feature_layer[:, -1].long()
            # [(image_id, anchor_id, y_id, x_id)]
            g_bbox = torch.cat(tensors=(gxy, gwh), dim=-1)
            return b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1), g_points, g_bbox, c
