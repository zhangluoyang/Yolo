import torch
from typing import *
import torch.nn as nn
import torch.nn.functional as F


def select_candidate_in_gts(xy_centers: torch.Tensor,
                            gt_bbox_es: torch.Tensor,
                            eps: float = 1e-9,
                            roll_out: bool = False) -> torch.Tensor:
    """

    :param xy_centers: anchor的中心坐标点
    :param gt_bbox_es:
    :param eps:
    :param roll_out: 是否逐个样本进行计算
    :return:
    """
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bbox_es.shape
    device = gt_bbox_es.device
    if roll_out:
        bbox_deltas = torch.empty((bs, n_boxes, n_anchors), device=device)
        for b in range(bs):
            lt, rb = gt_bbox_es[b].view(-1, 1, 4).chunk(2, 2)
            bbox_deltas[b] = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]),
                                       dim=2).view(n_boxes, n_anchors, -1).amin(2).gt_(eps)
        return bbox_deltas
    else:
        # 真实框的坐上右下left-top, right-bottom
        lt, rb = gt_bbox_es.view(-1, 1, 4).chunk(2, 2)
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps)


def select_highest_overlaps(mask_pos: torch.Tensor,
                            overlaps: torch.Tensor,
                            n_max_box_es: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    为每一个anchor选择最大的交并比
    :param mask_pos:
    :param overlaps:
    :param n_max_box_es: 每一个anchor最多的box
    :return:
    """
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1:
        # b, n_max_boxes, 8400
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_box_es, 1])
        # b, 8400
        max_overlaps_idx = overlaps.argmax(1)
        # b, 8400, n_max_boxes
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_box_es)
        # b, n_max_boxes, 8400
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
        # b, n_max_boxes, 8400
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
        fg_mask = mask_pos.sum(-2)
    # 最大交并比的id
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
    return target_gt_idx, fg_mask, mask_pos


def get_targets(gt_labels: torch.Tensor,
                gt_bbox_es: torch.Tensor,
                target_gt_idx: torch.Tensor,
                fg_mask: torch.Tensor,
                batch_size: int,
                n_max_boxes: int,
                num_classes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    格式转换
    :param gt_labels:  (b, max_num_obj, 1)
    :param gt_bbox_es: (b, max_num_obj, 4)
    :param target_gt_idx: (b, h*w)
    :param fg_mask: (b, h*w)
    :param batch_size:
    :param n_max_boxes: 每一个样本最多的box数目
    :param num_classes: 类别数目
    :return:
    """
    device = gt_labels.device
    batch_ind = torch.arange(end=batch_size, dtype=torch.int64, device=device)[..., None]
    # batch内的每一个样本每一个ground_true均有一个唯一的id
    target_gt_idx = target_gt_idx + batch_ind * n_max_boxes
    # 按照 target_id 重新对 label 和 bbox 排序
    target_labels = gt_labels.long().flatten()[target_gt_idx]
    target_bbox_es = gt_bbox_es.view(-1, 4)[target_gt_idx]

    target_labels.clamp(0)
    target_scores = F.one_hot(target_labels, num_classes)
    fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, num_classes)
    target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

    return target_labels, target_bbox_es, target_scores


def cal_iou(box_as: torch.Tensor, box_bs: torch.Tensor) -> torch.Tensor:
    """

    :param box_as:
    :param box_bs:
    :return:
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box_as.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box_bs.chunk(4, -1)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + 1e-9
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + 1e-9

    inter = (torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)).clamp(0) * \
            (torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)).clamp(0)

    union = w1 * h1 + w2 * h2 - inter + 1e-9
    return inter / union


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

    def __init__(self, top_k: int = 13,
                 num_classes: int = 80,
                 alpha: float = 1.0,
                 beta: float = 6.0,
                 eps: float = 1e-9,
                 roll_out_thr: int = 64):
        super(TaskAlignedAssigner, self).__init__()
        self.top_k = top_k
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.roll_out_thr = roll_out_thr

    def get_box_metrics(self, pd_scores: torch.Tensor,
                        pd_bbox_es: torch.Tensor,
                        gt_labels: torch.Tensor,
                        gt_bbox_es: torch.Tensor,
                        batch_size: int,
                        n_max_boxes: int,
                        roll_out: bool):
        """

        :param pd_scores: 预测的分值
        :param pd_bbox_es: 预测的框
        :param gt_labels:
        :param gt_bbox_es:
        :param batch_size:
        :param n_max_boxes: 最大的框数目
        :param roll_out:
        :return:
        """
        device = pd_scores.device
        if roll_out:
            align_metric = torch.empty((batch_size, n_max_boxes, pd_scores.shape[1]), device=device)
            overlaps = torch.empty((batch_size, n_max_boxes, pd_scores.shape[1]), device=device)
            ind_0 = torch.empty(n_max_boxes, dtype=torch.long)
            for b in range(batch_size):
                ind_0[:], ind_2 = b, gt_labels[b].squeeze(-1).long()
                # bs, max_num_obj, 8400
                bbox_scores = pd_scores[ind_0, :, ind_2]
                # bs, max_num_obj, 8400
                overlaps[b] = cal_iou(gt_bbox_es[b].unsqueeze(1),
                                      pd_bbox_es[b].unsqueeze(0)).squeeze(2).clamp(0)
                # 分值
                align_metric[b] = bbox_scores.pow(self.alpha) * overlaps[b].pow(self.beta)
        else:
            ind = torch.zeros([2, batch_size, n_max_boxes], dtype=torch.long)
            # batch当中的图片 id
            ind[0] = torch.arange(end=batch_size).view(-1, 1).repeat(1, n_max_boxes)
            # ground_true id
            ind[1] = gt_labels.long().squeeze(-1)

            bbox_scores = pd_scores[ind[0], :, ind[1]]

            overlaps = cal_iou(gt_bbox_es.unsqueeze(2),
                               pd_bbox_es.unsqueeze(1)).squeeze(3).clamp(0)
            align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def get_pos_mask(self,
                     pd_scores: torch.Tensor,
                     pd_bbox_es: torch.Tensor,
                     gt_labels: torch.Tensor,
                     gt_bbox_es: torch.Tensor,
                     anc_points: torch.Tensor,
                     mask_gt: torch.Tensor,
                     batch_size: int,
                     n_max_boxes: int,
                     roll_out: bool):
        # align_metric是一个算出来的代价值，某个先验点属于某个真实框的类的概率乘上某个先验点与真实框的重合程度
        # overlaps是某个先验点与真实框的重合程度
        # align_metric, overlaps    bs, max_num_obj, 8400
        align_metric, overlaps = self.get_box_metrics(pd_scores=pd_scores,
                                                      pd_bbox_es=pd_bbox_es,
                                                      gt_labels=gt_labels,
                                                      gt_bbox_es=gt_bbox_es,
                                                      batch_size=batch_size,
                                                      n_max_boxes=n_max_boxes,
                                                      roll_out=roll_out)

        # 正样本锚点需要同时满足：
        # 1、在真实框内
        # 2、是真实框top_k最重合的正样本
        # 3、满足mask_gt

        # get in_gts mask  b, max_num_obj, 8400
        # 判断先验点是否在真实框内
        mask_in_gts = select_candidate_in_gts(anc_points, gt_bbox_es, roll_out=roll_out)
        # get top_k_metric mask     b, max_num_obj, 8400
        # 判断锚点是否在真实框的top k中
        mask_top_k = self.select_top_k_candidates(align_metric * mask_in_gts,
                                                  roll_out=roll_out,
                                                  top_k_mask=mask_gt.repeat([1, 1, self.top_k]).bool())
        # merge all mask to a final mask, b, max_num_obj, h*w
        # 真实框存在，非padding
        mask_pos = mask_top_k * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def select_top_k_candidates(self, metrics, roll_out: bool, largest=True, top_k_mask=None):
        """

        :param metrics:
        :param roll_out:
        :param largest:
        :param top_k_mask:
        :return:
        """
        # 8400
        num_anchors = metrics.shape[-1]
        # b, max_num_obj, top_k
        top_k_metrics, top_k_idx_s = torch.topk(metrics, self.top_k, dim=-1, largest=largest)
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        # b, max_num_obj, top_k
        top_k_idx_s[~top_k_mask] = 0
        # b, max_num_obj, top_k, 8400 -> b, max_num_obj, 8400
        # 这一步得到的is_in_top_k为b, max_num_obj, 8400
        # 代表每个真实框对应的top k个先验点
        if roll_out:
            is_in_top_k = torch.empty(metrics.shape, dtype=torch.long, device=metrics.device)
            for b in range(len(top_k_idx_s)):
                is_in_top_k[b] = F.one_hot(top_k_idx_s[b], num_anchors).sum(-2)
        else:
            is_in_top_k = F.one_hot(top_k_idx_s, num_anchors).sum(-2)
        # 判断锚点是否在真实框的top_k中
        is_in_top_k = torch.where(is_in_top_k > 1, 0, is_in_top_k)
        return is_in_top_k.to(metrics.dtype)

    @torch.no_grad()
    def forward(self,
                pd_scores: torch.Tensor,
                pd_bbox_es: torch.Tensor,
                anc_points: torch.Tensor,
                gt_labels: torch.Tensor,
                gt_bbox_es: torch.Tensor,
                mask_gt: torch.Tensor):
        """

        :param pd_scores: 预测的置信度 (bs, num_total_anchors)
        :param pd_bbox_es: 预测的坐标 (bs, num_total_anchors, 4)
        :param anc_points: anchor参考点 (num_total_anchors, 2)
        :param gt_labels: ground_true label (bs, n_max_boxes, 1)
        :param gt_bbox_es: ground_true bbox (bs, n_max_boxes, 4)
        :param mask_gt: (bs, n_max_boxes, 1)
        :return:
        """
        n_max_boxes = gt_bbox_es.size(1)

        device = gt_bbox_es.device

        if n_max_boxes == 0:
            return (torch.full_like(pd_scores[..., 0], self.num_classes).to(device),
                    torch.zeros_like(pd_bbox_es).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        batch_size = pd_scores.size(0)
        # 每一个样本单独计算 还是 按照 batch 一起计算
        roll_out = n_max_boxes > self.roll_out_thr if self.roll_out_thr else False

        # mask_pos 是否满足三个条件 (1、ground_true内、top_k iou 匹配的正样本、mask_gt的描点)
        # align_metric 匹配的得分 (重合度*概率)
        # overlaps 重合度
        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores,
                                                             pd_bbox_es,
                                                             gt_labels,
                                                             gt_bbox_es,
                                                             anc_points,
                                                             mask_gt,
                                                             batch_size,
                                                             n_max_boxes=n_max_boxes,
                                                             roll_out=roll_out)
        # target_gt_idx b, 8400: 每一个anchor对应的ground_true
        # fg_mask b, 8400 每一个anchor是否有匹配的ground_true
        # mask_pos b, max_num_obj, 8400    one_hot后的target_gt_idx
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos=mask_pos,
                                                                   overlaps=overlaps,
                                                                   n_max_box_es=n_max_boxes)

        target_labels, target_bbox_es, target_scores = get_targets(gt_labels=gt_labels,
                                                                   gt_bbox_es=gt_bbox_es,
                                                                   target_gt_idx=target_gt_idx,
                                                                   fg_mask=fg_mask,
                                                                   batch_size=batch_size,
                                                                   n_max_boxes=n_max_boxes,
                                                                   num_classes=self.num_classes)
        # 去掉不符合条件的候选项
        align_metric *= mask_pos
        # 每一个ground_true的最大得分
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        # 每一个ground_true的最大重合度
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)
        # 归一化分值 (匹配得分*最大重合度) / 最大分值
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        # 最终分值
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bbox_es, target_scores, fg_mask.bool(), target_gt_idx
