import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *
import torchvision


class YoloCandidateTarget(nn.Module):

    def __init__(self,
                 anchors: List[List[Tuple[int, int]]],
                 strides: List[int],
                 anchor_num: int = 3):
        super(YoloCandidateTarget, self).__init__()

        self.anchors = torch.tensor(np.array(anchors, dtype=np.float32))
        self.strides = strides
        self.anchor_num = anchor_num
        self.threshold: float = 4.0

    def forward(self, predictions: List[torch.Tensor],
                targets: torch.Tensor) -> Tuple[List[Tuple[torch.Tensor,
                                                           torch.Tensor,
                                                           torch.Tensor,
                                                           torch.Tensor]],
                                                List[torch.Tensor]]:
        """
        :param predictions [b, anchor, height, width,1 + 4 + class_num]
        :param targets: shape=(?, 6)  [[image_id, x, y, w, h, c]]
        :return:
        """
        num_gt = targets.shape[0]
        indices, anchors = [], []

        gain = torch.ones(7, device=targets.device)

        ai = torch.arange(self.anchor_num, device=targets.device).float() \
            .view(self.anchor_num, 1).repeat(1, num_gt)
        targets = torch.cat((targets.repeat(self.anchor_num, 1, 1), ai[:, :, None]), 2)

        g = 0.5
        off = torch.tensor([[0, 0],  # 中间
                            [1, 0],  # 右边
                            [0, 1],  # 下边
                            [-1, 0],  # 左边
                            [0, -1]],  # 上边
                           device=targets.device).float() * g

        for i in range(len(predictions)):
            # anchor 位于当前特征层的尺寸
            anchors_i = (self.anchors[i] / self.strides[i]).type_as(predictions[i])
            shape = predictions[i].shape
            # [w, h, w, h]
            gain[2:6] = torch.tensor(predictions[i].shape)[[3, 2, 3, 2]]
            # 最终目标在坐在特征层中坐标
            target_in_feature_layer = targets * gain
            if num_gt:

                r = target_in_feature_layer[:, :, 4:6] / anchors_i[:, None]
                j = torch.max(r, 1. / r).max(2)[0] < self.threshold
                target_in_feature_layer = target_in_feature_layer[j]

                gxy = target_in_feature_layer[:, 2:4]
                # 计算相对于 左上角的偏移量
                gxi = gain[[2, 3]] - gxy
                # 判断是第几象限 并找到符合条件
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))

                target_in_feature_layer = target_in_feature_layer.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                target_in_feature_layer = targets[0]
                offsets = 0

            b, c = target_in_feature_layer[:, :2].long().T
            gxy = target_in_feature_layer[:, 2:4]
            gwh = target_in_feature_layer[:, 4:6]
            gij = (gxy - offsets).long()
            gi, gj = gij.T

            a = target_in_feature_layer[:, -1].long()
            # [(image_id, anchor_id, y_id, x_id)]
            indices.append((b,
                            a,
                            gj.clamp_(0, shape[2] - 1),
                            gi.clamp_(0, shape[3] - 1)))
            # [(anchor_w, anchor_h)]
            anchors.append(anchors_i[a])

        return indices, anchors


class YoloSimpleOTATarget(nn.Module):

    """
    步骤1: 根据 ground_true 确定 候选样本
        3个特征层 20 40 80 (按照输入的尺寸 640)
        1.1 计算 ground_true 与 anchor 的长宽 比例
        1.2 保留 长宽比在 1/4 ~ 4
        1.3 根据ground_true中心点 确定 anchor中心点，再根据中心点所在的偏移坐标，扩展两个中心点
        1.4 如果一个 anchor匹配到了两个ground_true则 选择iou最大的
    步骤2: 确定正负样本
        2.1 计算候选anchor与ground_true的iou值，计算sum()值，确定候选的样本数目，保证候选样本为1 dynamic_k
        2.2 根据候选的anchor和ground_true计算 loss: 3*iou_loss + cls_loss
        2.3 选择 loss 最小的 dynamic_k 作为正样本 其余作为负样本


    """

    def __init__(self, image_size: Tuple[int, int],
                 strides: List[int],
                 num_classes: int):
        super(YoloSimpleOTATarget, self).__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.strides = strides
        self.max_top_k: int = 20

    def forward(self, predictions: List[torch.Tensor],
                targets: List[torch.Tensor],
                indices: List[Tuple[torch.Tensor,
                                    torch.Tensor,
                                    torch.Tensor,
                                    torch.Tensor]],
                anchors: List[torch.Tensor]):
        """

        :param predictions: 3层预测值 (分别对应8,16,32倍下采样的结果)
        :param targets: [[image_id, x, y, w, h, c]]
        :param indices: 候选的索引
        :param anchors: 候选的索引对应的anchor 位于当前特征层的尺寸
        :return:
        """
        # 匹配的 image_id (每一个匹次中第几张图片)
        matching_bs: List[List[torch.Tensor]] = [[] for _ in predictions]
        # 匹配的 anchor_id
        matching_as: List[List[torch.Tensor]] = [[] for _ in predictions]
        # 匹配的 y_id
        matching_gjs: List[List[torch.Tensor]] = [[] for _ in predictions]
        # 匹配的 x_id
        matching_gis: List[List[torch.Tensor]] = [[] for _ in predictions]
        # 匹配的目标
        matching_targets: List[List[torch.Tensor]] = [[] for _ in predictions]
        # 匹配的 anchor
        matching_anchors: List[List[torch.Tensor]] = [[] for _ in predictions]
        # 层数 (3层)
        num_layer = len(predictions)

        batch_size = predictions[0].shape[0]

        device = targets[0].device

        for batch_idx in range(batch_size):
            # 属于 batch_idx 的 target
            b_idx = targets[:, 0] == batch_idx
            this_target = targets[b_idx]
            # 当前图片的 gt
            num_gt = this_target.shape[0]
            if num_gt == 0:
                continue
            # 真是图片坐标系的坐标
            x_y_w_h_in_image = this_target[:, 2:6] * self.image_size[0]

            x_y_x_y_in_image = torchvision.ops.box_convert(boxes=x_y_w_h_in_image,
                                                           in_fmt="cxcywh",
                                                           out_fmt="xyxy")
            # 记录预测值
            _predict_x_y_x_y_in_image = []

            _p_obj = []
            _p_cls = []

            _layer_mask = []
            # 所有的 image_id
            _all_b = []
            # 所有的 anchor_id
            _all_a = []
            # 所有的 y坐标
            _all_gj = []
            # 所有的 x坐标
            _all_gi = []
            # 所有的 anchor尺寸
            _all_anchors = []

            for layer_id, prediction in enumerate(predictions):
                # 找出属于该  image_id 的 anchor_id, y_id, x_id 和 anchor
                b, a, gj, gi = indices[layer_id]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]

                _all_b.append(b)
                _all_a.append(a)
                _all_gj.append(gj)
                _all_gi.append(gi)
                _all_anchors.append(anchors[layer_id][idx])

                _layer_mask.append(torch.ones(size=(len(b),)).to(device) * layer_id)
                # 对应候选位置 的预测结果 (此时的坐标系是特征层的坐标系)
                _candidate_prediction = prediction[b, a, gj, gi]
                # 预测的置信度
                _p_obj.append(_candidate_prediction[:, 4:5])
                # 预测的类别
                _p_cls.append(_candidate_prediction[:, 5:])
                # 图像坐标系的坐标
                pxy_in_image = _candidate_prediction[:, :2] * self.strides[layer_id]
                pwh_in_image = _candidate_prediction[:, 2:4] * self.strides[layer_id]
                p_xy_wh_in_image = torch.cat([pxy_in_image, pwh_in_image], dim=-1)
                p_xy_xy_in_image = torchvision.ops.box_convert(boxes=p_xy_wh_in_image,
                                                               in_fmt="cxcywh",
                                                               out_fmt="xyxy")
                _predict_x_y_x_y_in_image.append(p_xy_xy_in_image)

            predict_x_y_x_y_in_image = torch.cat(_predict_x_y_x_y_in_image, dim=0)
            # ? 一定存在预测框吧？
            if predict_x_y_x_y_in_image.shape[0] == 0:
                continue

            p_obj = torch.cat(_p_obj, dim=0)
            p_cls = torch.cat(_p_cls, dim=0)
            layer_mask = torch.cat(_layer_mask, dim=0)

            all_b = torch.cat(_all_b, dim=0)
            all_a = torch.cat(_all_a, dim=0)
            all_gj = torch.cat(_all_gj, dim=0)
            all_gi = torch.cat(_all_gi, dim=0)
            all_anchors = torch.cat(_all_anchors, dim=0)

            """
            计算预测框与真实框的重合程度
            """
            pair_iou = torchvision.ops.box_iou(boxes1=x_y_x_y_in_image,
                                               boxes2=predict_x_y_x_y_in_image)
            pair_iou_loss = - torch.log(pair_iou + 1e-5)

            top_k, _ = torch.topk(pair_iou, min(self.max_top_k, pair_iou.shape[1]), dim=1)
            # 正样本 (至少为1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)
            # 类别损失
            gt_cls_per_image = F.one_hot(this_target[:, 1].to(torch.int64),
                                         self.num_classes) \
                .float().unsqueeze(1).repeat(1, predict_x_y_x_y_in_image.shape[0], 1)

            cls_conf_predict = \
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1) * p_obj.unsqueeze(0).repeat(num_gt, 1, 1)

            y = cls_conf_predict.sqrt_()
            pair_cls_conf_loss = F.binary_cross_entropy_with_logits(torch.log(y / (1 - y)), gt_cls_per_image,
                                                                    reduction="none").sum(-1)
            # 代价 shape=(predict_num(candidate_num), ground_true_num)
            cost = pair_cls_conf_loss + 3.0 * pair_iou_loss
            # 1 表示选中 0 表示未选中
            matching_matrix = torch.zeros_like(cost)
            for gt_idx in range(num_gt):
                # 选择k个代价函数最小的
                _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
                matching_matrix[gt_idx][pos_idx] = 1.0
            # 满足条件的数目
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                # 出现一个预测框 对应 多个 ground_true
                # 1 先找出最小的索引
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                # 2 先全部置0
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                # 3 将对应多个部分的最小代价值下标索引置为1
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            # 有效的匹配数目
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            # 找出最佳匹配 (其实每一行仅有一个下标对应的数值为1)
            matched_gt_ids = matching_matrix[:, fg_mask_inboxes].argmax(0)

            layer_mask = layer_mask[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anchors = all_anchors[fg_mask_inboxes]
            this_target = this_target[matched_gt_ids]

            for layer_id in range(num_layer):
                layer_idx = layer_mask == layer_id
                matching_bs[layer_id].append(all_b[layer_idx])
                matching_as[layer_id].append(all_a[layer_idx])
                matching_gjs[layer_id].append(all_gj[layer_idx])
                matching_gis[layer_id].append(all_gi[layer_idx])
                matching_targets[layer_id].append(this_target[layer_idx])
                matching_anchors[layer_id].append(all_anchors[layer_idx])

        for layer_id in range(num_layer):
            matching_bs[layer_id] = torch.cat(matching_bs[layer_id], dim=0) \
                if len(matching_bs[layer_id]) != 0 else torch.Tensor(matching_bs[layer_id])
            matching_as[layer_id] = torch.cat(matching_as[layer_id], dim=0) if len(
                matching_as[layer_id]) != 0 else torch.Tensor(
                matching_as[layer_id])
            matching_gjs[layer_id] = torch.cat(matching_gjs[layer_id], dim=0) if len(
                matching_gjs[layer_id]) != 0 else torch.Tensor(
                matching_gjs[layer_id])
            matching_gis[layer_id] = torch.cat(matching_gis[layer_id], dim=0) if len(
                matching_gis[layer_id]) != 0 else torch.Tensor(
                matching_gis[layer_id])
            matching_targets[layer_id] = torch.cat(matching_targets[layer_id], dim=0) if len(
                matching_targets[layer_id]) != 0 else torch.Tensor(matching_targets[layer_id])
            matching_anchors[layer_id] = torch.cat(matching_anchors[layer_id], dim=0) if len(
                matching_anchors[layer_id]) != 0 else torch.Tensor(
                matching_anchors[layer_id])

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchors
