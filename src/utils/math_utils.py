import math
import numpy as np
import copy
from typing import Dict


def cal_mean_on_line(on_line_dict: Dict[str, float],
                     one_dict: Dict[str, float],
                     step: int):
    """
    :param on_line_dict:
    :param one_dict:
    :param step:
    :return:
    """
    for key, value in one_dict.items():
        if key not in on_line_dict:
            on_line_dict[key] = value
        else:
            on_line_dict[key] = (max(step, 1) / (step + 1)) * on_line_dict[key] + one_dict[key] / (step + 1)


def rotate(points: np.ndarray, radian: float) -> np.ndarray:
    """
    绕原点旋转
    :param points 点 [None, 2]
    :param radian 旋转弧度
    """
    cos_rotate = math.cos(radian)
    sin_rotate = math.sin(radian)
    # 记录原有的
    ori_points: np.ndarray = copy.deepcopy(points)
    points[:, 0] = ori_points[:, 0] * cos_rotate - ori_points[:, 1] * sin_rotate
    points[:, 1] = ori_points[:, 0] * sin_rotate + ori_points[:, 1] * cos_rotate
    return points


def rotate_center(points: np.ndarray, radian: float, center: np.ndarray = None) -> np.ndarray:
    """
    根据中心点旋转
    :param points 点 [None, 2]
    :param radian 旋转弧度
    :param center 中心点坐标 默认值是原点
    """
    points: np.ndarray = copy.deepcopy(points)
    num, _ = np.shape(points)
    if center is None:
        center = np.array([[0, 0]] * num)
    cos_rotate = math.cos(radian)
    sin_rotate = math.sin(radian)
    # 记录原有的
    ori_points: np.ndarray = copy.deepcopy(points)
    points[:, 0] = (ori_points[:, 0] - center[:, 0]) * cos_rotate - (
            ori_points[:, 1] - center[:, 1]) * sin_rotate + center[:, 0]
    points[:, 1] = (ori_points[:, 0] - center[:, 0]) * sin_rotate + (
            ori_points[:, 1] - center[:, 1]) * cos_rotate + center[:, 1]
    return points
