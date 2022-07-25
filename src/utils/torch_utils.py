import random
import torch
import torch.nn as nn
import numpy as np
import torchvision
from typing import Dict, List, Union
from torch.optim.lr_scheduler import LambdaLR
import math
from functools import partial
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


def non_max_suppression(prediction: torch.Tensor,
                        img_size: float,
                        conf_threshold: float = 0.5,
                        nms_threshold: float = 0.3) -> List[Union[None, torch.Tensor]]:
    """
    非极大值抑制算法
    :param prediction: [[center_x, center_y, width, height, conf, ...]] shape=(bs, ....)
    :param conf_threshold:
    :param img_size:
    :param nms_threshold:
    :return: 返回 [min_x, min_y, max_x, max_y] 形式
    """
    prediction = torch.clone(prediction)
    batch_size = prediction.shape[0]
    prediction[..., :4] = torchvision.ops.box_convert(boxes=prediction[..., :4],
                                                      in_fmt="cxcywh",
                                                      out_fmt="xyxy")
    prediction[..., :4] = torch.clamp(prediction[..., :4], min=0, max=img_size - 1)

    max_wh = 4096
    # 允许检测的最长时间
    output: List[Union[None, torch.Tensor]] = [None for _ in range(batch_size)]

    for xi in range(batch_size):
        image_predict = prediction[xi]
        class_conf, class_id = torch.max(image_predict[:, 5:], 1, keepdim=True)
        conf_mask = (image_predict[:, 4] * class_conf[:, 0] >= conf_threshold)

        image_predict = image_predict[conf_mask]
        class_conf = class_conf[conf_mask]
        class_id = class_id[conf_mask]
        if image_predict.size(0):
            # [x1, y1, x2, t2, obj_conf, class_conf, class_id]
            detection = torch.cat(tensors=(image_predict[:, :5], class_conf, class_id.float()), dim=-1)
            c = detection[:, 6:7] * max_wh
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
