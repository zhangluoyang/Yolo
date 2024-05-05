"""
yolo v6 的损失函数
"""
from luoyang.model.Layer import LossLayer
from luoyang.model.loss.DflLoss import DflLoss
from luoyang.model.loss.FocalLoss import FocalLoss
from luoyang.yolov6_face.YoloV6FaceTarget import *
from luoyang.model.loss.BceSmoothLoss import BceSmoothLoss
from luoyang.model.loss.WingLoss import WingLoss


class YoloV6FaceLoss(LossLayer):

    def __init__(self,
                 target_name: str,
                 anchors_xy_xy_name: str,
                 anchor_points_xy_name: str,
                 num_anchors_name: str,
                 predict_xy_xy_name: str,
                 point_predict_xy_name: str,
                 predict_score_name: str,
                 stride_name: str,
                 num_classes: int,
                 point_num: int,
                 img_size: int,
                 use_dfl: bool,
                 reg_distribute_name: str,
                 loss_name: str,
                 reg_max: int = 16,
                 warmup_epoch: int = 4):
        super(YoloV6FaceLoss, self).__init__()
        self.img_size = img_size
        self.target_name = target_name
        self.anchors_xy_xy_name = anchors_xy_xy_name
        self.num_anchors_name = num_anchors_name
        self.predict_xy_xy_name = predict_xy_xy_name
        self.point_predict_xy_name = point_predict_xy_name
        self.predict_score_name = predict_score_name
        self.anchor_points_xy_name = anchor_points_xy_name
        self.warmup_epoch = warmup_epoch
        self.num_classes = num_classes
        self.stride_name = stride_name
        self.reg_distribute_name = reg_distribute_name
        self.loss_name = loss_name
        self.point_num = point_num

        self.use_dfl = use_dfl

        self.focal_loss = FocalLoss(reduction="sum")

        self.point_loss = WingLoss()

        self.dfl_loss = DflLoss(num_classes=num_classes,
                                reg_max=reg_max,
                                wh_to_xy=False,
                                use_dfl=self.use_dfl)

        self.formal_assigner = TaskAlignedAssigner(num_classes=num_classes,
                                                   point_num=point_num,
                                                   alpha=1.0,
                                                   beta=6.0)

        self.point_conf_loss = BceSmoothLoss(reduce="mean")

        self.class_loss_weight = 1.0
        self.iou_loss_weight = 2.5
        self.dfl_loss_weight = 0.5
        self.point_weight = 1.25

    def loss_feed_forward(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]) -> \
            Dict[str, torch.Tensor]:
        predict_xy_xy = tensor_dict[self.predict_xy_xy_name]
        anchor_points_xy = tensor_dict[self.anchor_points_xy_name]
        device = predict_xy_xy.device
        # [bs, anchor_num, 2 * point_num]
        point_predict_xy = tensor_dict[self.point_predict_xy_name]

        ground_true = tensor_dict[self.target_name]
        # [batch, max_ground_true_num, 1]
        ground_true_labels = ground_true[:, :, -1:]
        # [batch, max_ground_true_num, 4]
        ground_true_xy_xy = ground_true[:, :, :4]
        # [batch, 2 * max_ground_true_num]
        ground_true_point = ground_true[:, :, 4:-1]

        mask_ground_true = (ground_true_xy_xy.sum(-1, keepdim=True) > 0).float()
        # anchors_xx_yy = tensor_dict[self.anchors_xy_xy_name]
        # num_anchors_list = tensor_dict[self.num_anchors_name]
        cls_score_tensor = tensor_dict[self.predict_score_name]

        stride_tensor = tensor_dict[self.stride_name]

        reg_distribute_tensor = tensor_dict[self.reg_distribute_name]
        # 由特征层 转换为输出层
        try:
            target_labels, ground_true_xx_yy, ground_true_point_xy, target_scores, fg_mask = \
                self.formal_assigner(predict_xy_xy=predict_xy_xy,
                                     predict_score=cls_score_tensor.detach(),
                                     anchors_xx=anchor_points_xy,
                                     ground_true_labels=ground_true_labels,
                                     ground_true_xx_yy=ground_true_xy_xy,
                                     ground_true_point=ground_true_point,
                                     mask_ground_true=mask_ground_true)
        except RuntimeError as e:
            target_labels, ground_true_xx_yy, ground_true_point_xy, target_scores, fg_mask = \
                self.formal_assigner(predict_xy_xy=predict_xy_xy.float().cpu(),
                                     predict_score=cls_score_tensor.float().cpu().detach(),
                                     anchors_xx=anchor_points_xy.float().cpu(),
                                     ground_true_labels=ground_true_labels.float().cpu(),
                                     ground_true_xx_yy=ground_true_xy_xy.float().cpu(),
                                     ground_true_point=ground_true_point.float().cpu(),
                                     mask_ground_true=mask_ground_true.float().cpu())
            target_labels = target_labels.to(device)
            ground_true_xx_yy = ground_true_xx_yy.to(device)
            target_scores = target_scores.to(device)
            ground_true_point_xy = ground_true_point_xy.to(device)
            fg_mask = fg_mask.to(device)

        scale_ground_true_xx_yy = ground_true_xx_yy / stride_tensor
        scale_predict_xy_xy = predict_xy_xy / stride_tensor
        scale_anchor_points_x_y = anchor_points_xy / stride_tensor
        scale_ground_true_point_xy = ground_true_point_xy / stride_tensor
        scale_point_predict_xy = point_predict_xy / stride_tensor
        # one-hot 并 去掉 padding的影响
        target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.num_classes))
        one_hot_label = F.one_hot(target_labels.long(), self.num_classes + 1)[..., :-1]
        # 类别损失
        loss_cls = self.focal_loss(cls_score_tensor, target_scores, one_hot_label)
        target_scores_sum = target_scores.sum()
        if target_scores_sum > 1:
            loss_cls /= target_scores_sum

        loss_iou, loss_dfl = self.dfl_loss(reg_distribute=reg_distribute_tensor,
                                           scale_predict_xy_xy=scale_predict_xy_xy,
                                           scale_anchor_points_x_y=scale_anchor_points_x_y,
                                           scale_ground_true_xx_yy=scale_ground_true_xx_yy,
                                           target_scores=target_scores,
                                           target_scores_sum=target_scores_sum,
                                           fg_mask=fg_mask)

        num_pos = fg_mask.sum()
        if num_pos > 0:
            # [num_pos, point_num * 2]
            _point_predict_xy = scale_point_predict_xy[fg_mask]
            # [num_pos, point_num * 2]
            _ground_true_point_xy = scale_ground_true_point_xy[fg_mask]
            # [num_pos, point_num * 2]
            point_mask = _ground_true_point_xy > 0

            point_loss = self.point_loss(_point_predict_xy[point_mask],
                                         _ground_true_point_xy[point_mask]) * self.point_weight
        else:
            point_loss = 0
        loss = self.class_loss_weight * loss_cls + self.iou_loss_weight * loss_iou + self.dfl_loss_weight * loss_dfl
        loss += point_loss
        return {"{0}_loss".format(self.loss_name): loss}
