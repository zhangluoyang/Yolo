import cv2
import copy
import numpy as np
from typing import List
from typing import Dict, Union

import torch
from PIL.JpegImagePlugin import JpegImageFile


def draw_bbox_labels(img: Union[np.ndarray, JpegImageFile], boxes: np.ndarray, labels: List[str]):
    """
    可视化结果
    :param img:
    :param boxes:
    :param labels:
    :return:
    """
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
                              color=(0, 255, 255), thickness=1)
        cv2.putText(image, label_name, (int(min_x), int(min_y - 5)), 0, 0.5, (255, 255, 0), 2, lineType=cv2.LINE_AA)
    return image
