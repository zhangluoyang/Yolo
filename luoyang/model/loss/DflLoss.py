from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from luoyang.model.loss.GIouLoss import GIouLoss


class DflLoss(nn.Module):

    def __init__(self, num_classes: int,
                 reg_max: int,
                 wh_to_xy: bool,
                 use_dfl: bool):
        super(DflLoss, self).__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        self.iou_loss = GIouLoss(return_g_iou=False,
                                 wh_to_xy=wh_to_xy,
                                 reduction="none")

        self.use_dfl = use_dfl

    @staticmethod
    def bbox2dist(anchor_points: torch.Tensor, xy_xy: torch.Tensor, reg_max: int):
        x1y1, x2y2 = torch.split(xy_xy, 2, -1)
        # left top 距离
        lt = anchor_points - x1y1
        # right bottom 距离
        rb = x2y2 - anchor_points
        dist = torch.cat([lt, rb], -1).clip(0, reg_max - 0.01)
        return dist

    def forward(self, reg_distribute: torch.Tensor,
                scale_predict_xy_xy: torch.Tensor,
                scale_anchor_points_x_y: torch.Tensor,
                scale_ground_true_xx_yy: torch.Tensor,
                target_scores: torch.Tensor,
                target_scores_sum: torch.Tensor,
                fg_mask: torch.Tensor):
        num_pos = fg_mask.sum()
        if num_pos > 0:
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
            predict_xy_xy_pos = torch.masked_select(scale_predict_xy_xy,
                                                    bbox_mask).reshape([-1, 4])
            target_xy_xy_pos = torch.masked_select(
                scale_ground_true_xx_yy, bbox_mask).reshape([-1, 4])

            bbox_weight = torch.masked_select(
                target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_iou = self.iou_loss(predict_xy_xy_pos,
                                     target_xy_xy_pos)
            # [anchor_num, ] -> [anchor_num, 1]
            loss_iou = loss_iou.unsqueeze(-1) * bbox_weight

            if target_scores_sum > 1:
                loss_iou = loss_iou.sum() / target_scores_sum
            else:
                loss_iou = loss_iou.sum()

            if self.use_dfl:
                dist_mask = fg_mask.unsqueeze(-1).repeat([1, 1, (self.reg_max + 1) * 4])
                predict_dist_pos = torch.masked_select(reg_distribute, dist_mask).reshape([-1, 4, self.reg_max + 1])

                target_l_t_r_b = self.bbox2dist(scale_anchor_points_x_y, scale_ground_true_xx_yy, self.reg_max)
                target_l_t_r_b_pos = torch.masked_select(target_l_t_r_b, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(predict_dist_pos, target_l_t_r_b_pos) * bbox_weight

                if target_scores_sum > 1:
                    loss_dfl = loss_dfl.sum() / target_scores_sum
                else:
                    loss_dfl = loss_dfl.sum()
            else:
                loss_dfl = reg_distribute.sum() * 0
        else:
            loss_iou = reg_distribute.sum() * 0.
            loss_dfl = reg_distribute.sum() * 0.
        return loss_iou, loss_dfl

    def _df_loss(self, pred_dist: torch.Tensor, target: torch.Tensor):
        # 向下取整 作为目标的左侧整数
        target_left = target.to(torch.long)
        # 向上取整 作为目标的右侧整数
        target_right = target_left + 1
        # 左侧权重 距离右侧越远 权重越大
        weight_left = target_right.to(torch.float) - target
        # 右侧权重 距离右侧越近 权重越大
        weight_right = 1 - weight_left
        # 左侧目标函数
        loss_left = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction='none').view(
            target_left.shape) * weight_left
        # 右侧目标函数
        loss_right = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction='none').view(
            target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)
