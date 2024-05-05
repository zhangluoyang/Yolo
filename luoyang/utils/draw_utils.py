import cv2
import copy
import torch
import numpy as np
from typing import *
from typing import Dict, Union
from PIL.JpegImagePlugin import JpegImageFile


def draw_bbox_labels(img: Union[np.ndarray, JpegImageFile], boxes: np.ndarray, labels: List[str]):
    """
    可视化结果
    :param img:
    :param boxes:
    :param labels:
    :return:
    """
    img = copy.deepcopy(img)
    if not isinstance(img, np.ndarray):
        img = np.array(img, dtype=np.uint8)
    for label, bbox in zip(labels, boxes):
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.rectangle(img, (x1, y1 - t_size[1]), (int(x1 + t_size[0] * 0.5), y1), (0, 255, 255), -1)
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return img


def draw_points(img: Union[np.ndarray, JpegImageFile], points: np.ndarray):
    """

    :param img:
    :param points:
    :return:
    """
    img = copy.deepcopy(img)
    for point in points:
        point = (int(point[0]), int(point[1]))
        img = cv2.circle(img, point, 1, (255, 0, 255), -1)
    return img


def draw_object(data_dict: Dict[str, np.ndarray],
                class_names: List[str]) -> np.ndarray:
    """

    :param data_dict:
    :param class_names
    :return:
    """
    image: Union[np.ndarray, torch.Tensor] = copy.deepcopy(data_dict["image"]) * 255
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = image.astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    im_height, im_width, channel = np.shape(image)
    for target in data_dict["targets"]:
        center_norm_x = target[0]
        center_norm_y = target[1]
        norm_width = target[2]
        norm_height = target[3]

        label_name = class_names[int(target[4])]

        norm_min_x, norm_min_y = center_norm_x - 0.5 * norm_width, center_norm_y - 0.5 * norm_height
        norm_max_x, norm_max_y = center_norm_x + 0.5 * norm_width, center_norm_y + 0.5 * norm_height

        min_x, min_y = int(norm_min_x * im_width), int(norm_min_y * im_height)
        max_x, max_y = int(norm_max_x * im_width), int(norm_max_y * im_height)

        image = cv2.rectangle(img=image,
                              pt1=(min_x, min_y),
                              pt2=(max_x, max_y),
                              color=(0, 255, 255), thickness=5)
        cv2.putText(image, label_name, (int(min_x), int(min_y - 5)), 0, 0.5, (255, 255, 0), 2, lineType=cv2.LINE_AA)
    return image


def draw_pose(img: Union[np.ndarray, JpegImageFile], points: List[Tuple[float, float]]):
    """
    可视化结果
    :param img:
    :param points: 关节16个点
    :return:
    """
    img = copy.deepcopy(img)
    if not isinstance(img, np.ndarray):
        img = np.array(img, dtype=np.uint8)

    _int_points = []
    for p in points:
        _int_points.append((int(p[0]), int(p[1])))

    # (0 1) (1 2) (2 6) (6 3) (3 4) (4 5)
    img = cv2.line(img, pt1=_int_points[0], pt2=points[1], color=(0, 255, 255))
    img = cv2.line(img, pt1=_int_points[1], pt2=points[2], color=(0, 255, 255))
    img = cv2.line(img, pt1=_int_points[2], pt2=points[6], color=(0, 255, 255))
    img = cv2.line(img, pt1=_int_points[6], pt2=points[3], color=(0, 255, 255))
    img = cv2.line(img, pt1=_int_points[3], pt2=points[4], color=(0, 255, 255))
    img = cv2.line(img, pt1=_int_points[4], pt2=points[5], color=(0, 255, 255))
    # (6 7) (7 8) (8 9)
    img = cv2.line(img, pt1=_int_points[6], pt2=points[7], color=(0, 255, 255))
    img = cv2.line(img, pt1=_int_points[7], pt2=points[8], color=(0, 255, 255))
    img = cv2.line(img, pt1=_int_points[8], pt2=points[9], color=(0, 255, 255))
    # (10 11) (11 12) (12 8) (8 13) (13 14) (14 15)
    img = cv2.line(img, pt1=_int_points[10], pt2=points[11], color=(0, 255, 255))
    img = cv2.line(img, pt1=_int_points[11], pt2=points[12], color=(0, 255, 255))
    img = cv2.line(img, pt1=_int_points[12], pt2=points[8], color=(0, 255, 255))
    img = cv2.line(img, pt1=_int_points[8], pt2=points[13], color=(0, 255, 255))
    img = cv2.line(img, pt1=_int_points[13], pt2=points[14], color=(0, 255, 255))
    img = cv2.line(img, pt1=_int_points[14], pt2=points[15], color=(0, 255, 255))
    return img


def draw_skeleton(img: np.ndarray,
                  points: np.ndarray,
                  point_confs: np.ndarray,
                  confidence: float = 0.45):
    # coco 数据集格式
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
                [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    for p_ids in skeleton:
        p_id_1 = p_ids[0] - 1
        p_id_2 = p_ids[1] - 1
        pt1 = points[p_id_1]
        pt2 = points[p_id_2]
        if point_confs[p_id_1] > confidence and point_confs[p_id_2] > confidence:
            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = (int(pt2[0]), int(pt2[1]))
            cv2.line(img, pt1, pt2, (0, 0, 255), 4)
            img = cv2.circle(img, pt1, 2, (255, 0, 255), -1)
            img = cv2.circle(img, pt2, 2, (255, 0, 255), -1)
    return img
