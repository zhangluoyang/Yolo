import numpy as np
import torch
from typing import Union, Tuple


def check_anchor_order(m):
    a = m.anchors.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        m.anchors[:] = m.anchors.flip(0)


def mask_grid(height: int, width: int) -> torch.Tensor:
    """
    
    :param height: 
    :param width: 
    :return: 
    """
    yv, xv = torch.meshgrid([torch.arange(width), torch.arange(height)])
    return torch.stack((xv, yv), 2).view((1, 1, height, width, 2)).float()


def bbox_iou(bbox: np.ndarray,
             gt: np.ndarray):
    """
    计算 iou 值
    :param bbox: [n, 4]
    :param gt: [1, 4]
    :return: len(result) = n
    """
    # left_top (x, y)
    lt = np.maximum(bbox[:, None, :2], gt[:, :2])
    # right_bottom (x, y)
    rb = np.minimum(bbox[:, None, 2:], gt[:, 2:])
    # inter_area (w, h)
    wh = np.maximum(rb - lt + 1, 0)
    # shape: (n, m)
    inter_areas = wh[:, :, 0] * wh[:, :, 1]
    box_areas = (bbox[:, 2] - bbox[:, 0] + 1) * (bbox[:, 3] - bbox[:, 1] + 1)
    gt_areas = (gt[:, 2] - gt[:, 0] + 1) * (gt[:, 3] - gt[:, 1] + 1)
    iou = inter_areas / (box_areas[:, None] + gt_areas - inter_areas)
    return iou.reshape(-1, )


def box_to_rect(x: Union[float, int],
                y: Union[float, int],
                w: Union[float, int],
                h: Union[float, int]) \
        -> Tuple[Union[float, int], Union[float, int], Union[float, int], Union[float, int]]:
    """
    box 数据 转换为 rect 数据
    :param x:
    :param y:
    :param w:
    :param h:
    :return:
    """
    return x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h


def box_to_rect_np(x: np.ndarray,
                   y: np.ndarray,
                   w: np.ndarray,
                   h: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

    :param x:
    :param y:
    :param w:
    :param h:
    :return:
    """
    return x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h


def x_y_w_h_to_x_y_x_y(box: np.ndarray) -> np.ndarray:
    """
    将 (x, y, w, h) 形式转换为 (min_x, min_y, max_x, max_y) 形式
    :param box:
    :return:
    """
    return np.concatenate((box[..., :2] - 0.5 * box[..., 2:],
                           box[..., :2] + 0.5 * box[..., 2:]), axis=-1)
