"""
yolo v8 的损失函数
"""
import torch
from typing import *
import torch.nn.functional as F
from luoyang.model.Layer import LossLayer
from luoyang.model.loss.DflLoss import DflLoss
from luoyang.model.loss.FocalLoss import FocalLoss


class YoloV8Loss(LossLayer):

    def __init__(self,
                 num_classes: int,
                 loss_name: str,
                 reg_max: int = 16):
        super(YoloV8Loss, self).__init__()
        self.num_classes = num_classes
        self.loss_name = loss_name

        self.focal_loss = FocalLoss(reduction="sum")

        self.dfl_loss = DflLoss(num_classes=num_classes,
                                reg_max=reg_max - 1,
                                wh_to_xy=False,
                                use_dfl=True)

        self.class_loss_weight = 0.5
        self.iou_loss_weight = 7.5
        self.dfl_loss_weight = 1.5

    def loss_feed_forward(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]) -> \
            Dict[str, torch.Tensor]:
        stride_tensor = tensor_dict["stride_tensor"]
        ground_true_xx_yy = tensor_dict["target_bbox_es"]

        scale_ground_true_xx_yy = ground_true_xx_yy / stride_tensor

        predict_scaled_bounding_boxes = tensor_dict["predict_scaled_bounding_boxes"]
        scale_predict_xy_xy = predict_scaled_bounding_boxes

        scale_anchor_points_x_y = tensor_dict["anchor_points"]
        target_labels = tensor_dict["target_labels"]
        fg_mask = tensor_dict["fg_mask"]

        target_scores = tensor_dict["target_scores"]
        cls_score_tensor = tensor_dict["predict_scores"]

        # one-hot 并 去掉 padding的影响
        target_labels = torch.where(fg_mask, target_labels, torch.full_like(target_labels, self.num_classes))
        one_hot_label = F.one_hot(target_labels.long(), self.num_classes + 1)[..., :-1]
        # 类别损失
        loss_cls = self.focal_loss(cls_score_tensor, target_scores, one_hot_label)
        target_scores_sum = target_scores.sum()
        if target_scores_sum > 1:
            loss_cls /= target_scores_sum

        # dfl 预测结果 (没有加权求和之前)
        predict_distribute = tensor_dict["predict_distribute"]

        loss_iou, loss_dfl = self.dfl_loss(reg_distribute=predict_distribute,
                                           scale_predict_xy_xy=scale_predict_xy_xy,
                                           scale_anchor_points_x_y=scale_anchor_points_x_y,
                                           scale_ground_true_xx_yy=scale_ground_true_xx_yy,
                                           target_scores=target_scores,
                                           target_scores_sum=target_scores_sum,
                                           fg_mask=fg_mask)

        loss = self.class_loss_weight * loss_cls + self.iou_loss_weight * loss_iou + self.dfl_loss_weight * loss_dfl

        return {"{0}_loss".format(self.loss_name): loss}
