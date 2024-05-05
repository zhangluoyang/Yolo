import math
import torch
import random
import numpy as np
import torchvision
import torchvision.ops
import torch.nn as nn
from functools import partial
from typing import *
from torch.optim.optimizer import Optimizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def to_device(data_dict: Dict[str, np.ndarray], device: str):
    """
    :param data_dict:
    :param device:
    :return:
    """
    for key in data_dict.keys():
        if isinstance(data_dict[key], np.ndarray):
            data_dict[key] = torch.from_numpy(data_dict[key]).to(device)
        elif isinstance(data_dict[key], torch.Tensor):
            data_dict[key] = data_dict[key].to(device)
        elif isinstance(data_dict[key], list):
            for v in data_dict[key]:
                if isinstance(v, dict):
                    to_device(data_dict=v, device=device)


def cal_iou(bbox_xx_yy_01: torch.Tensor,
            bbox_xx_yy_02: torch.Tensor,
            eps=1e-7):
    inter, union = _box_inter_union(bbox_xx_yy_01, bbox_xx_yy_02)
    iou = inter / (union + eps)
    return iou


def _box_inter_union(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    area1 = torchvision.ops.box_area(boxes1)
    area2 = torchvision.ops.box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union


def cal_m_iou(bbox_xx_yy_01: torch.Tensor,
              bbox_xx_yy_02: torch.Tensor,
              eps=1e-9):
    box1 = bbox_xx_yy_01.unsqueeze(2)
    box2 = bbox_xx_yy_02.unsqueeze(1)
    px1y1, px2y2 = box1[:, :, :, 0:2], box1[:, :, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, :, 0:2], box2[:, :, :, 2:4]
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps
    return overlap / union


def cal_center_distance(bbox_xx_yy_01: torch.Tensor,
                        bbox_xx_yy_02: torch.Tensor):
    """
    计算两个 bbox 中心点的距离
    :param bbox_xx_yy_01:
    :param bbox_xx_yy_02:
    :return:
    """
    gt_cx = (bbox_xx_yy_01[:, 0] + bbox_xx_yy_01[:, 2]) / 2.0
    gt_cy = (bbox_xx_yy_01[:, 1] + bbox_xx_yy_01[:, 3]) / 2.0
    gt_points = torch.stack([gt_cx, gt_cy], dim=1)
    ac_cx = (bbox_xx_yy_02[:, 0] + bbox_xx_yy_02[:, 2]) / 2.0
    ac_cy = (bbox_xx_yy_02[:, 1] + bbox_xx_yy_02[:, 3]) / 2.0
    ac_points = torch.stack([ac_cx, ac_cy], dim=1)
    distances = (gt_points[:, None, :] - ac_points[None, :, :]).pow(2).sum(-1).sqrt()
    return distances, ac_points


def _upcast(t: torch.Tensor) -> torch.Tensor:
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def non_max_suppression(prediction: torch.Tensor,
                        img_size: float,
                        in_fmt: str = "cxcywh",
                        conf_threshold: float = 0.5,
                        nms_threshold: float = 0.3) -> List[Union[None, torch.Tensor]]:
    """
    非极大值抑制算法
    :param prediction: [[center_x, center_y, width, height, conf, ...]] shape=(bs, ....)
    :param conf_threshold:
    :param img_size:
    :param in_fmt:
    :param nms_threshold:
    :return: 返回 [min_x, min_y, max_x, max_y, obj_conf, class_conf, class_id] 形式
    """
    prediction = torch.clone(prediction)
    batch_size = prediction.shape[0]
    if in_fmt == "cxcywh":
        prediction[..., :4] = torchvision.ops.box_convert(boxes=prediction[..., :4],
                                                          in_fmt=in_fmt,
                                                          out_fmt="xyxy")
    prediction[..., :4] = torch.clamp(prediction[..., :4], min=0, max=img_size - 1)
    # 保证单独同一类进行nms
    max_wh = 4096
    output: List[Union[None, torch.Tensor]] = [None for _ in range(batch_size)]

    for xi in range(batch_size):
        image_predict = prediction[xi]
        class_conf, class_id = torch.max(image_predict[:, 5:], 1, keepdim=True)
        conf_mask = (image_predict[:, 4] * class_conf[:, 0] >= conf_threshold)

        image_predict = image_predict[conf_mask]
        class_conf = class_conf[conf_mask]
        class_id = class_id[conf_mask]
        if image_predict.size(0):
            # [x1, y1, x2, y2, obj_conf, class_conf, class_id]
            detection = torch.cat(tensors=(image_predict[:, :5], class_conf, class_id.float()), dim=-1)
            c = detection[:, 6:7] * max_wh
            boxes, scores = detection[:, :4] + c, detection[:, 4] * detection[:, 5]
            i = torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=nms_threshold)
            output[xi] = detection[i]
    return output


def non_max_suppression_with_point(prediction: torch.Tensor,
                                   img_size: float,
                                   point_num: int,
                                   in_fmt: str = "cxcywh",
                                   conf_threshold: float = 0.5,
                                   nms_threshold: float = 0.3) -> List[Union[None, torch.Tensor]]:
    """
    非极大值抑制算法
    :param prediction: [[center_x, center_y, width, height, conf, point_1_x, point_1_y, ... class1, class2,...]]
    :param conf_threshold:
    :param img_size:
    :param point_num:
    :param in_fmt
    :param nms_threshold:
    :return: 返回 [min_x, min_y, max_x, max_y, obj_conf, class_conf, point_x, point_y,....class_id] 形式
    """
    prediction = torch.clone(prediction)
    batch_size = prediction.shape[0]
    if in_fmt == "cxcywh":
        prediction[..., :4] = torchvision.ops.box_convert(boxes=prediction[..., :4],
                                                          in_fmt="cxcywh",
                                                          out_fmt="xyxy")
    prediction[..., :4] = torch.clamp(prediction[..., :4], min=0, max=img_size - 1)

    max_wh = 4096
    # 允许检测的最长时间
    output: List[Union[None, torch.Tensor]] = [None for _ in range(batch_size)]

    for xi in range(batch_size):
        image_predict = prediction[xi]
        class_conf, class_id = torch.max(image_predict[:, 5 + point_num * 2:], 1, keepdim=True)
        conf_mask = (image_predict[:, 4] * class_conf[:, 0] >= conf_threshold)

        image_predict = image_predict[conf_mask]
        class_conf = class_conf[conf_mask]
        class_id = class_id[conf_mask]
        if image_predict.size(0):
            point = image_predict[:, 5: 5 + 2 * point_num]
            # [x1, y1, x2, t2, obj_conf, class_conf, 2*point,class_id]
            detection = torch.cat(tensors=(image_predict[:, :5],
                                           class_conf,
                                           point,
                                           class_id.float()), dim=-1)
            c = detection[:, -1:] * max_wh
            boxes, scores = detection[:, :4] + c, detection[:, 4] * detection[:, 5]
            i = torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=nms_threshold)
            output[xi] = detection[i]
    return output


def non_max_suppression_with_point_conf(prediction: torch.Tensor,
                                        img_size: float,
                                        point_num: int,
                                        in_fmt: str = "cxcywh",
                                        conf_threshold: float = 0.5,
                                        nms_threshold: float = 0.3) -> List[Union[None, torch.Tensor]]:
    """
    非极大值抑制算法
    :param prediction: [[center_x, center_y, width, height, conf, point_1_x, point_1_y,...,point_conf,... class1, class2,...]]
    :param conf_threshold:
    :param img_size:
    :param point_num:
    :param in_fmt
    :param nms_threshold:
    :return: 返回 [min_x, min_y, max_x, max_y, obj_conf, class_conf, point1_x, point1_y,...point1_conf,....class_id]
    """
    prediction = torch.clone(prediction)
    batch_size = prediction.shape[0]
    if in_fmt == "cxcywh":
        prediction[..., :4] = torchvision.ops.box_convert(boxes=prediction[..., :4],
                                                          in_fmt="cxcywh",
                                                          out_fmt="xyxy")
    prediction[..., :4] = torch.clamp(prediction[..., :4], min=0, max=img_size - 1)

    max_wh = 4096
    # 允许检测的最长时间
    output: List[Union[None, torch.Tensor]] = [None for _ in range(batch_size)]

    for xi in range(batch_size):
        image_predict = prediction[xi]
        class_conf, class_id = torch.max(image_predict[:, 5 + point_num * 3:], 1, keepdim=True)
        conf_mask = (image_predict[:, 4] * class_conf[:, 0] >= conf_threshold)

        image_predict = image_predict[conf_mask]
        class_conf = class_conf[conf_mask]
        class_id = class_id[conf_mask]
        if image_predict.size(0):
            point = image_predict[:, 5: 5 + 3 * point_num]
            # [x1, y1, x2, t2, obj_conf, class_conf, 3*point, class_id]
            detection = torch.cat(tensors=(image_predict[:, :5],
                                           class_conf,
                                           point,
                                           class_id.float()), dim=-1)
            c = detection[:, -1:] * max_wh
            boxes, scores = detection[:, :4] + c, detection[:, 4] * detection[:, 5]
            i = torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=nms_threshold)
            output[xi] = detection[i]
    return output


def empty_cache():
    torch.cuda.empty_cache()


def weight_init(net: nn.Module,
                init_gain: float = 2e-2):
    """

    :param net:
    :param init_gain:
    :return:
    """

    def init_func(m: nn.Module):
        class_name = m.__class__.__name__
        if hasattr(m, 'weight') and class_name.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
        elif class_name.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def is_parallel(model: Union[nn.Module, nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]):
    """
    判断模式是否是并行
    :param model:
    :return:
    """
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model: Union[nn.Module, nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]):
    """
    返回模型
    :param model:
    :return:
    """
    return model.module if is_parallel(model) else model


def get_lr_scheduler(lr_decay_type: str,
                     lr: float,
                     min_lr: float,
                     total_iter: int,
                     warmup_iter_ratio=0.1,
                     warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.3,
                     step_num=10):
    def warm_cos_lr(_lr: float,
                    _min_lr: float,
                    _total_iter: float,
                    _warmup_total_iter: float,
                    _warmup_lr_start: float,
                    _no_aug_iter: float,
                    _iter: int):
        if _iter <= _warmup_total_iter:
            _lr = (_lr - _warmup_lr_start) * pow(_iter / float(_warmup_total_iter), 2) + _warmup_lr_start
        elif _iter >= _total_iter - _no_aug_iter:
            _lr = _min_lr
        else:
            _lr = _min_lr + 0.5 * (_lr - _min_lr) * (1.0 + math.cos(
                math.pi * (_iter - _warmup_total_iter) / (_total_iter - _warmup_total_iter - _no_aug_iter)))
        return _lr

    def step_lr(_lr, _decay_rate, _step_size, _iter):
        if _step_size < 1:
            raise ValueError("step_size must above 1.")
        n = _iter // _step_size
        out_lr = _lr * _decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iter = min(max(warmup_iter_ratio * total_iter, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iter, 1), 15)
        func = partial(warm_cos_lr, lr, min_lr, total_iter, warmup_total_iter, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iter / step_num
        func = partial(step_lr, lr, decay_rate, step_size)
    return func


def set_optimizer_lr(optimizer: Optimizer, lr_scheduler_func, epoch: int):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr(optimizer: Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def intersect_dicts(da, db, exclude=()):
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def fuse_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    重参化 卷积层核匹归一化层
    :param conv:
    :param bn:
    :return:
    """
    assert isinstance(conv, nn.Conv2d)
    assert isinstance(bn, nn.BatchNorm2d)

    fused_conv = (nn.Conv2d(conv.in_channels,
                            conv.out_channels,
                            kernel_size=conv.kernel_size,
                            stride=conv.stride,
                            padding=conv.padding,
                            groups=conv.groups,
                            bias=True).requires_grad_(False).to(conv.weight.device))

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused_conv.weight.copy_(torch.mm(w_bn, w_conv).view(fused_conv.weight.shape))

    b_conv = (torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None
              else conv.bias)
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_conv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fused_conv


def dist2bbox(distance, anchor_points, output_format: str, dim=-1):
    """

    :param distance: 距离四个边界 (上下左右) 的距离
    :param anchor_points:
    :param output_format: 输出样式
    :param dim:
    :return:
    """
    assert output_format in ['xyxy', 'xywh', 'cxcywh']
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if output_format == "xyxy":
        return torch.cat((x1y1, x2y2), dim)
    elif output_format == "cxcywh":
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    else:
        wh = x2y2 - x1y1
        return torch.cat((x1y1, wh), dim)
