import numpy as np
import torch
from typing import *
import time


def yolo_collate_fn(data_dict_list: List[Dict[str, Any]]) -> \
        Dict[str, Union[np.ndarray, List[torch.Tensor]]]:
    """

    :param data_dict_list:
    :return:
    """
    batch_size = len(data_dict_list)
    first_image: np.ndarray = data_dict_list[0]["image"]
    channel, height, width = first_image.shape

    batch_images = np.zeros(shape=(batch_size, channel, height, width), dtype=np.float32)
    batch_targets = [torch.tensor(data_dict["targets"], dtype=torch.float32) if data_dict["targets"] is not None
                     else None for data_dict in data_dict_list]

    for _id, data_dict in enumerate(data_dict_list):
        batch_images[_id] = data_dict["image"]

    tensor_dict = {"batch_images": batch_images,
                   "batch_targets": batch_targets}
    # yolo v5 format
    if "head_3_ground_true" in data_dict_list[0]:
        first_head_3_ground_true: np.ndarray = data_dict_list[0]["head_3_ground_true"]
        anchor_num, height, width, output_size = first_head_3_ground_true.shape
        head_3_ground_true = np.zeros(shape=(batch_size, anchor_num, height, width, output_size), dtype=np.float32)
        for _id, data_dict in enumerate(data_dict_list):
            head_3_ground_true[_id] = data_dict["head_3_ground_true"]
        tensor_dict["head_3_ground_true"] = head_3_ground_true

    if "head_4_ground_true" in data_dict_list[0]:
        first_head_4_ground_true: np.ndarray = data_dict_list[0]["head_4_ground_true"]
        anchor_num, height, width, output_size = first_head_4_ground_true.shape
        head_4_ground_true = np.zeros(shape=(batch_size, anchor_num, height, width, output_size), dtype=np.float32)
        for _id, data_dict in enumerate(data_dict_list):
            head_4_ground_true[_id] = data_dict["head_4_ground_true"]
        tensor_dict["head_4_ground_true"] = head_4_ground_true

    if "head_5_ground_true" in data_dict_list[0]:
        head_5_ground_true: np.ndarray = data_dict_list[0]["head_5_ground_true"]
        anchor_num, height, width, output_size = head_5_ground_true.shape
        head_5_ground_true = np.zeros(shape=(batch_size, anchor_num, height, width, output_size), dtype=np.float32)
        for _id, data_dict in enumerate(data_dict_list):
            head_5_ground_true[_id] = data_dict["head_5_ground_true"]
        tensor_dict["head_5_ground_true"] = head_5_ground_true
    return tensor_dict
