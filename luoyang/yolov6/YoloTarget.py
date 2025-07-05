import torch
from typing import *
import torch.nn as nn
import torch.nn.functional as F
import luoyang.utils.torch_utils as torch_utils


def select_candidates_in_gts(xy_centers: torch.Tensor,
                             ground_true_xx_yy: torch.Tensor,
                             eps: float = 1e-9):
    """
    保证 选择的 anchor 中心点 在 ground_true 的 box 内
    :param xy_centers:
    :param ground_true_xx_yy:
    :param eps:
    :return:
    """
    n_anchors = xy_centers.size(0)
    bs, n_max_boxes, _ = ground_true_xx_yy.size()
    _ground_true_xx_yy = ground_true_xx_yy.reshape([-1, 4])
    xy_centers = xy_centers.unsqueeze(0).repeat(bs * n_max_boxes, 1, 1)
    gt_xx_yy_lt = _ground_true_xx_yy[:, 0:2].unsqueeze(1).repeat(1, n_anchors, 1)
    gt_xx_yy_rb = _ground_true_xx_yy[:, 2:4].unsqueeze(1).repeat(1, n_anchors, 1)
    # 计算 ground_true
    b_lt = xy_centers - gt_xx_yy_lt
    b_rb = gt_xx_yy_rb - xy_centers
    bbox_deltas = torch.cat([b_lt, b_rb], dim=-1)
    bbox_deltas = bbox_deltas.reshape([bs, n_max_boxes, n_anchors, -1])
    return (bbox_deltas.min(axis=-1)[0] > eps).to(ground_true_xx_yy.dtype)


def select_highest_overlaps(mask_pos: torch.Tensor,
                            overlaps: torch.Tensor,
                            n_max_boxes: int):
    """

    :param mask_pos:
    :param overlaps:
    :param n_max_boxes:
    :return:
    """
    fg_mask = mask_pos.sum(dim=-2)
    if fg_mask.max() > 1:
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])
        max_overlaps_idx = overlaps.argmax(dim=1)
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
        fg_mask = mask_pos.sum(dim=-2)
    target_gt_idx = mask_pos.argmax(dim=-2)
    return target_gt_idx, fg_mask, mask_pos


class ATAssigner(nn.Module):
    """
    Adaptive Training Sample Selection Assigner
    1. 计算 ground_true 与 每一层特征中心点的 L2 距离， 距离最近的 k 个点作为候选点， 一共L层，就会生成 k*L 个候选特征点
    2. 计算候选点与 ground_true的 IOU值
    3. 计算 IOU值的 均值与标准差
    4. iou > 均值 + 标准差 则作为正样本 其余作为负样本
    5. 出现一个 anchor 对应多个 ground_true的情况 取iou最大的类别
    """

    def __init__(self,
                 num_classes: int,
                 top_k: int = 9):
        super(ATAssigner, self).__init__()
        self.num_classes = num_classes
        self.top_k = top_k
        self.bg_idx = num_classes
        # anchor 数目
        self.n_anchors = NotImplemented
        self.bs = NotImplemented
        # 本次batch中 最大的 ground_true 数目
        self.n_max_boxes = NotImplemented

    @torch.no_grad()
    def forward(self, anchors_xx_yy: torch.Tensor,
                num_anchors_list: List[int],
                ground_true_labels: torch.Tensor,
                ground_true_xx_yy: torch.Tensor,
                mask_ground_true: torch.Tensor,
                predict_xy_xy: torch.Tensor):
        """

        :param anchors_xx_yy: [[l, t, r, d], ....]  每一个anchor对应的框坐标
        :param num_anchors_list: 每一层对应的anchor数目
        :param ground_true_labels: 类别
        :param ground_true_xx_yy: 目标框
        :param mask_ground_true: 0 表示 padding, 1 表示正常的样本
        :param predict_xy_xy: 预测的目标框
        :return:
        """
        self.n_anchors = anchors_xx_yy.size(0)
        self.bs = ground_true_xx_yy.size(0)
        self.n_max_boxes = ground_true_xx_yy.size(1)

        if self.n_max_boxes == 0:
            # 无目标框的情况
            device = ground_true_xx_yy.device
            return torch.full([self.bs, self.n_anchors], self.bg_idx).to(device), \
                   torch.zeros([self.bs, self.n_anchors, 4]).to(device), \
                   torch.zeros([self.bs, self.n_anchors, self.num_classes]).to(device), \
                   torch.zeros([self.bs, self.n_anchors]).to(device)
        # 计算 ground_true 与 anchor 的 iou 值
        overlaps = torch_utils.cal_iou(ground_true_xx_yy.reshape([-1, 4]),
                                       anchors_xx_yy)
        overlaps = overlaps.reshape([self.bs, -1, self.n_anchors])
        # 计算 ground_true 与 anchor 的 中心点距离
        distances, ac_points = torch_utils.cal_center_distance(ground_true_xx_yy.reshape([-1, 4]),
                                                               anchors_xx_yy)
        distances = distances.reshape([self.bs, -1, self.n_anchors])

        is_in_candidate_tensor, candidate_idx_tensor = self.select_top_k_candidates(distances=distances,
                                                                                    num_anchors_list=num_anchors_list,
                                                                                    mask_gt=mask_ground_true)
        # 计算 判断的阈值 (均值+方差)
        overlaps_thr_per_gt, iou_candidates = self.threshold_calculator(is_in_candidate_tensor=is_in_candidate_tensor,
                                                                        candidate_idx_tensor=candidate_idx_tensor,
                                                                        overlaps=overlaps)
        # 正样本
        is_pos = torch.where(iou_candidates > overlaps_thr_per_gt.repeat([1, 1, self.n_anchors]),
                             is_in_candidate_tensor, torch.zeros_like(is_in_candidate_tensor))

        is_in_gts = select_candidates_in_gts(xy_centers=ac_points,
                                             ground_true_xx_yy=ground_true_xx_yy)
        # 同时 满足 3个条件 才可以 当作正样本
        # 1 iou值阈值 2 中心点在 ground_true 内 3 非 padding
        # [batch, 1, anchor_num]
        mask_pos = is_pos * is_in_gts * mask_ground_true
        # [batch, anchor_num]
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos,
                                                                   overlaps,
                                                                   self.n_max_boxes)
        # [batch, anchor_num] [batch, anchor_num, 4] [batch, anchor_num, class_num]
        target_labels, target_xy_xy, target_scores = self.get_targets(ground_true_labels=ground_true_labels,
                                                                      ground_true_xx_yy=ground_true_xx_yy,
                                                                      target_gt_idx=target_gt_idx,
                                                                      fg_mask=fg_mask)
        if predict_xy_xy is not None:
            predict_iou = torch_utils.cal_m_iou(bbox_xx_yy_01=ground_true_xx_yy,
                                                bbox_xx_yy_02=predict_xy_xy) * mask_pos
            # [batch, anchor_num, 1]
            iou_max = predict_iou.max(dim=-2)[0].unsqueeze(-1)
            # [batch, anchor_num, class_num]
            target_scores *= iou_max

        return target_labels.long(), target_xy_xy, target_scores, fg_mask.bool()

    def select_top_k_candidates(self,
                                distances: torch.Tensor,
                                num_anchors_list: List[int],
                                mask_gt: torch.Tensor):
        mask_gt = mask_gt.repeat(1, 1, self.top_k).bool()
        # 按照 每一层的数目 进行切分
        level_distances = torch.split(distances, num_anchors_list, dim=-1)
        is_in_candidate_list = []
        candidate_idx_list = []
        start_idx = 0
        for per_level_distances, num_anchors in zip(level_distances, num_anchors_list):
            selected_k = min(self.top_k, num_anchors)
            _, per_level_top_k_idx_list = per_level_distances.topk(selected_k, dim=-1, largest=False)
            candidate_idx_list.append(per_level_top_k_idx_list + start_idx)
            # 保证 选取的候选 anchor 不是 padding的
            per_level_top_k_idx_list = torch.where(mask_gt,
                                                   per_level_top_k_idx_list,
                                                   torch.zeros_like(per_level_top_k_idx_list))

            is_in_candidate = F.one_hot(per_level_top_k_idx_list, num_anchors).sum(dim=-2)
            is_in_candidate = torch.where(is_in_candidate > 1,
                                          torch.zeros_like(is_in_candidate), is_in_candidate)
            is_in_candidate_list.append(is_in_candidate.to(distances.dtype))
            start_idx += num_anchors

        is_in_candidate_tensor = torch.cat(is_in_candidate_list, dim=-1)
        candidate_idx_tensor = torch.cat(candidate_idx_list, dim=-1)
        return is_in_candidate_tensor, candidate_idx_tensor

    def threshold_calculator(self,
                             is_in_candidate_tensor: torch.Tensor,
                             candidate_idx_tensor: torch.Tensor,
                             overlaps: torch.Tensor):
        n_bs_max_boxes = self.bs * self.n_max_boxes

        _candidate_overlaps = torch.where(is_in_candidate_tensor > 0,
                                          overlaps, torch.zeros_like(overlaps))

        candidate_idx_tensor = candidate_idx_tensor.reshape([n_bs_max_boxes, -1])

        assist_idx_tensor = self.n_anchors * torch.arange(n_bs_max_boxes,
                                                          device=candidate_idx_tensor.device)

        assist_idx_tensor = assist_idx_tensor[:, None]
        flatten_idx_tensor = candidate_idx_tensor + assist_idx_tensor
        candidate_overlaps = _candidate_overlaps.reshape(-1)[flatten_idx_tensor]
        candidate_overlaps = candidate_overlaps.reshape([self.bs, self.n_max_boxes, -1])
        # 计算均值与方差
        overlaps_mean_per_gt = candidate_overlaps.mean(axis=-1, keepdim=True)
        overlaps_std_per_gt = candidate_overlaps.std(axis=-1, keepdim=True)
        # 均值 + 方差 作为判断的阈值
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        return overlaps_thr_per_gt, _candidate_overlaps

    def get_targets(self,
                    ground_true_labels: torch.Tensor,
                    ground_true_xx_yy: torch.Tensor,
                    target_gt_idx: torch.Tensor,
                    fg_mask: torch.Tensor):
        """

        :param ground_true_labels:
        :param ground_true_xx_yy:
        :param target_gt_idx:
        :param fg_mask:
        :return:
        """
        batch_idx = torch.arange(self.bs, dtype=ground_true_labels.dtype, device=ground_true_labels.device)
        batch_idx = batch_idx[..., None]
        target_gt_idx = (target_gt_idx + batch_idx * self.n_max_boxes).long()
        target_labels = ground_true_labels.flatten()[target_gt_idx.flatten()]
        target_labels = target_labels.reshape([self.bs, self.n_anchors])
        # 如果是 padding 则 使用 bg_idx 代替
        target_labels = torch.where(fg_mask > 0,
                                    target_labels, torch.full_like(target_labels, self.bg_idx))

        # assigned target boxes
        target_xy_xy = ground_true_xx_yy.reshape([-1, 4])[target_gt_idx.flatten()]
        target_xy_xy = target_xy_xy.reshape([self.bs, self.n_anchors, 4])

        # assigned target scores
        target_scores = F.one_hot(target_labels.long(), self.num_classes + 1).float()
        # 去掉 padding 的影响 如果是 padding 则 全为 0
        target_scores = target_scores[:, :, :self.num_classes]

        return target_labels, target_xy_xy, target_scores


class TaskAlignedAssigner(nn.Module):
    """
    动态选择 正样本数目
    1. 计算交并比 IOU
    2. 计算评分值 iou^_{alpha} * score^_{beta}
    3. 计算top_k评分值
    4. 过滤掉不在 ground_true box 当中的数据
    5. 若一个 anchor 匹配多个 ground_true 则选择 iou 值最大的
    6. 生成匹配的 target
    """

    def __init__(self,
                 num_classes: int,
                 top_k: int = 13,
                 alpha: float = 1.0,
                 beta: float = 6.0,
                 eps: float = 1e-9):
        super(TaskAlignedAssigner, self).__init__()
        self.bg_idx = num_classes
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        # anchor 数目
        self.n_anchors = NotImplemented
        self.bs = NotImplemented
        # 本次batch中 最大的 ground_true 数目
        self.n_max_boxes = NotImplemented

    @torch.no_grad()
    def forward(self, predict_xy_xy: torch.Tensor,
                predict_score: torch.Tensor,
                anchors_xx: torch.Tensor,
                ground_true_labels: torch.Tensor,
                ground_true_xx_yy: torch.Tensor,
                mask_ground_true: torch.Tensor):
        """

        :param predict_xy_xy: [batch_size, anchor_nums, 4]
        :param predict_score: [batch_size, anchor_nums, num_class]
        :param anchors_xx: [anchor_num, 2]
        :param ground_true_labels: [batch_size, n_max_boxes, 1]
        :param ground_true_xx_yy: [batch_size, n_max_boxes, 4]
        :param mask_ground_true: [batch_size, n_max_boxes, 1]
        :return:
                target_labels [batch_size, anchor_nums]
                ground_true_xx_yy [batch_size, anchor_nums, 4]
                target_scores [batch_size, anchor_nums, num_class]
                fg_mask [batch_size, anchor_nums]
        """
        self.bs = predict_score.size(0)
        self.n_max_boxes = ground_true_xx_yy.size(1)

        if self.n_max_boxes == 0:
            device = predict_xy_xy.device
            return torch.full_like(predict_score[..., 0], self.bg_idx).to(device), \
                   torch.zeros_like(predict_xy_xy).to(device), \
                   torch.zeros_like(predict_score).to(device), \
                   torch.zeros_like(predict_score[..., 0]).to(device)

        cycle, step, self.bs = (1, self.bs, self.bs) if self.n_max_boxes <= 100 else (self.bs, 1, 1)
        target_labels_lst, target_xy_xy_lst, target_scores_lst, fg_mask_lst = [], [], [], []

        for i in range(cycle):
            # 起始下标
            start, end = i * step, (i + 1) * step
            pd_scores_ = predict_score[start:end, ...]
            pd_xy_xy_ = predict_xy_xy[start:end, ...]
            gt_labels_ = ground_true_labels[start:end, ...]
            gt_xy_xy_ = ground_true_xx_yy[start:end, ...]
            mask_gt_ = mask_ground_true[start:end, ...]
            # [batch, max_ground_true_num, anchor_num]
            mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores_, pd_xy_xy_, gt_labels_,
                                                                 gt_xy_xy_, anchors_xx, mask_gt_)

            target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(
                mask_pos, overlaps, self.n_max_boxes)

            # assigned target
            target_labels, target_xy_xy, target_scores = self.get_targets(
                gt_labels_, gt_xy_xy_, target_gt_idx, fg_mask)

            # normalize
            align_metric *= mask_pos
            pos_align_metrics = align_metric.max(axis=-1, keepdim=True)[0]
            pos_overlaps = (overlaps * mask_pos).max(axis=-1, keepdim=True)[0]
            norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).max(-2)[0].unsqueeze(-1)
            target_scores = target_scores * norm_align_metric

            # append
            target_labels_lst.append(target_labels)
            target_xy_xy_lst.append(target_xy_xy)
            target_scores_lst.append(target_scores)
            fg_mask_lst.append(fg_mask)

        target_labels = torch.cat(target_labels_lst, 0)
        target_xy_xy = torch.cat(target_xy_xy_lst, 0)
        target_scores = torch.cat(target_scores_lst, 0)
        fg_mask = torch.cat(fg_mask_lst, 0)

        return target_labels, target_xy_xy, target_scores, fg_mask.bool()

    def get_pos_mask(self, predict_scores: torch.Tensor,
                     predict_xy_xy: torch.Tensor,
                     ground_true_labels: torch.Tensor,
                     ground_true_xy_xy: torch.Tensor,
                     anc_points: torch.Tensor,
                     mask_gt: torch.Tensor):
        """
        正样本3个条件
        1. 非 padding
        2. anchor 在 ground_true 中
        3. top_k 分值
        :param predict_scores:
        :param predict_xy_xy:
        :param ground_true_labels:
        :param ground_true_xy_xy:
        :param anc_points:
        :param mask_gt:
        :return:
        """
        align_metric, overlaps = self.get_box_metrics(predict_scores=predict_scores,
                                                      predict_xy_xy=predict_xy_xy,
                                                      ground_true_labels=ground_true_labels,
                                                      ground_true_xy_xy=ground_true_xy_xy)

        mask_in_gts = select_candidates_in_gts(xy_centers=anc_points, ground_true_xx_yy=ground_true_xy_xy)
        # [batch, 1, anchor_num]
        mask_top_k = self.select_top_k_candidates(align_metric * mask_in_gts,
                                                  top_k_mask=mask_gt.repeat([1, 1, self.top_k]).bool())
        mask_pos = mask_top_k * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(self,
                        predict_scores: torch.Tensor,
                        predict_xy_xy: torch.Tensor,
                        ground_true_labels: torch.Tensor,
                        ground_true_xy_xy: torch.Tensor):
        """

        :param predict_scores:
        :param predict_xy_xy:
        :param ground_true_labels:
        :param ground_true_xy_xy:
        :return:
        """
        predict_scores = predict_scores.permute(0, 2, 1)
        ground_true_labels = ground_true_labels.to(torch.long)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        # batch_size
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)
        ind[1] = ground_true_labels.squeeze(-1)
        # 分值
        bbox_scores = predict_scores[ind[0], ind[1]]
        # 计算交并比
        overlaps = torch_utils.cal_m_iou(ground_true_xy_xy, predict_xy_xy)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_top_k_candidates(self,
                                metrics: torch.Tensor,
                                largest: bool = True,
                                top_k_mask: Union[torch.Tensor, None] = None):
        """
        获取 top_k 结果
        :param metrics:
        :param largest:
        :param top_k_mask:
        :return:
        """
        num_anchors = metrics.shape[-1]
        top_k_metrics, top_k_ids = torch.topk(metrics, self.top_k, dim=-1, largest=largest)
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.max(axis=-1, keepdim=True) > self.eps).tile([1, 1, self.top_k])
        top_k_ids = torch.where(top_k_mask, top_k_ids, torch.zeros_like(top_k_ids))
        is_in_top_k = F.one_hot(top_k_ids, num_anchors).sum(axis=-2)
        is_in_top_k = torch.where(is_in_top_k > 1, torch.zeros_like(is_in_top_k), is_in_top_k)
        return is_in_top_k.to(metrics.dtype)

    def get_targets(self,
                    ground_true_labels: torch.Tensor,
                    ground_true_xx_yy: torch.Tensor,
                    target_gt_idx: torch.Tensor,
                    fg_mask: torch.Tensor):
        """

        :param ground_true_labels:
        :param ground_true_xx_yy:
        :param target_gt_idx:
        :param fg_mask:
        :return:
        """
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=ground_true_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = ground_true_labels.long().flatten()[target_gt_idx]

        # assigned target boxes
        target_xx_yy = ground_true_xx_yy.reshape([-1, 4])[target_gt_idx]

        # assigned target scores
        target_labels[target_labels < 0] = 0
        target_scores = F.one_hot(target_labels, self.bg_idx)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.bg_idx)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, torch.full_like(target_scores, 0))

        return target_labels, target_xx_yy, target_scores
