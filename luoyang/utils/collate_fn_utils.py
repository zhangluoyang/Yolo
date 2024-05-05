import torch
import numpy as np
from typing import *
import torchvision


def yolo_v3_collate_fn(data_dict_list: List[Dict[str, Any]]) -> \
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
    tensor_dict = {"batch_images": torch.tensor(batch_images),
                   "batch_targets": batch_targets}
    return tensor_dict


def yolo_v5_collate_fn(data_dict_list: List[Dict[str, Any]]) -> \
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
    tensor_dict = {"batch_images": torch.tensor(batch_images),
                   "batch_targets": batch_targets}
    for i in range(3, 6):
        name = "head_{0}_ground_true".format(i)
        first_head_ground_true: np.ndarray = data_dict_list[0][name]
        anchor_num, height, width, output_size = first_head_ground_true.shape
        head_ground_true = np.zeros(shape=(batch_size, anchor_num, height, width, output_size), dtype=np.float32)
        for _id, data_dict in enumerate(data_dict_list):
            head_ground_true[_id] = data_dict[name]
        tensor_dict[name] = torch.tensor(head_ground_true)
    return tensor_dict


def yolo_v5_face_collate_fn(data_dict_list: List[Dict[str, Any]]) -> \
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

    tensor_dict = {"batch_images": torch.tensor(batch_images),
                   "batch_targets": batch_targets}

    for i in range(3, 6):
        name = "head_{0}_ground_true".format(i)
        first_head_ground_true: np.ndarray = data_dict_list[0][name]
        anchor_num, height, width, output_size = first_head_ground_true.shape
        head_ground_true = np.zeros(shape=(batch_size, anchor_num, height, width, output_size), dtype=np.float32)
        for _id, data_dict in enumerate(data_dict_list):
            head_ground_true[_id] = data_dict[name]
        tensor_dict[name] = head_ground_true

        point_name = "head_{0}_point_mask".format(i)
        head_point_mask: np.ndarray = data_dict_list[0][point_name]
        anchor_num, height, width = head_point_mask.shape
        head_point_mask = np.zeros(shape=(batch_size, anchor_num, height, width), dtype=np.int64)
        for _id, data_dict in enumerate(data_dict_list):
            head_point_mask[_id] = data_dict[point_name]
        tensor_dict[point_name] = head_point_mask
    return tensor_dict


def yolo_v5_v2_collate_fn(data_dict_list: List[Dict[str, Any]]) -> \
        Dict[str, Union[np.ndarray, List[torch.Tensor]]]:
    """

    :param data_dict_list:
    :return:
    """
    images = []
    bbox = []
    for i, data_dict in enumerate(data_dict_list):
        img = data_dict["image"]
        targets_v5 = data_dict["targets_v5"]
        images.append(img)
        # 记录 image_id
        if len(targets_v5) > 0:
            targets_v5[:, 0] = i
            bbox.append(targets_v5)
    if len(bbox) > 0:
        bbox = np.concatenate(bbox, 0)
    batch_targets = [torch.tensor(data_dict["targets"], dtype=torch.float32) if data_dict["targets"] is not None
                     else None for data_dict in data_dict_list]
    tensor_dict = {"batch_images": torch.tensor(np.array(images)),
                   "targets_v5": torch.tensor(bbox),
                   "batch_targets": batch_targets}
    return tensor_dict


def img_collate_fn(label_dict: Dict[str, int]):
    def collate_fn(data_dict_list: List[Dict[str, Any]]) -> Dict[str, Union[np.ndarray, List[torch.Tensor], Any]]:
        batch_images = np.array([data["image"] for data in data_dict_list], dtype=np.float32)
        labels = np.array([label_dict[data["label"]] for data in data_dict_list], dtype=np.int64)
        targets = [data["label"] for data in data_dict_list]

        return {"batch_images": torch.tensor(batch_images),
                "labels": torch.tensor(labels),
                "targets": targets}

    return collate_fn


def yolo_v7_collate_fn(data_dict_list: List[Dict[str, Any]]) -> \
        Dict[str, Union[np.ndarray, List[torch.Tensor]]]:
    """

    :param data_dict_list:
    :return:
    """
    images = []
    bbox = []
    for i, data_dict in enumerate(data_dict_list):
        img = data_dict["image"]
        targets_v7 = data_dict["targets_v7"]
        images.append(img)
        # 记录 image_id
        if len(targets_v7) > 0:
            targets_v7[:, 0] = i
            bbox.append(targets_v7)
    if len(bbox) > 0:
        bbox = np.concatenate(bbox, 0)
    batch_targets = [torch.tensor(data_dict["targets"], dtype=torch.float32) if data_dict["targets"] is not None
                     else None for data_dict in data_dict_list]
    tensor_dict = {"batch_images": torch.tensor(np.array(images)),
                   "targets_v7": torch.tensor(bbox),
                   "batch_targets": batch_targets}
    return tensor_dict


def yolo_v8_collate_fn(img_size: Tuple[int, int] = (640, 640)):
    def collate_fn(data_dict_list: List[Dict[str, Any]]) -> \
            Dict[str, Union[np.ndarray, List[torch.Tensor]]]:
        """

        :param data_dict_list:
        :return:
        """
        images = []
        bbox = []
        for i, data_dict in enumerate(data_dict_list):
            img = data_dict["image"]
            targets_v7 = data_dict["targets_v7"]
            images.append(img)
            # 记录 image_id
            if len(targets_v7) > 0:
                targets_v7[:, 0] = i
                bbox.append(targets_v7)
        if len(bbox) > 0:
            bbox = np.concatenate(bbox, 0)
        batch_targets = [torch.tensor(data_dict["targets"], dtype=torch.float32) if data_dict["targets"] is not None
                         else None for data_dict in data_dict_list]
        batch_size = len(data_dict_list)
        targets = torch.tensor(bbox)
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5)
        else:
            # batch内 每一张输入图片的下标索引
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            # 同一组batch内的每一张图片的bbox数目 不一样 此处会进行填充 (padding)
            out = torch.zeros(batch_size, counts.max(), 5, device=targets.device)

            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            # 转换为输入尺寸
            scale_tensor = torch.tensor([img_size[1], img_size[0], img_size[1], img_size[0]])
            out[..., 1:5] = torchvision.ops.box_convert(out[..., 1:5].mul_(scale_tensor),
                                                        in_fmt="cxcywh",
                                                        out_fmt="xyxy")
        tensor_dict = {"batch_images": torch.tensor(np.array(images)),
                       "targets_v8": out,
                       "batch_targets": batch_targets}
        return tensor_dict

    return collate_fn


def yolo_v6_collate_fn(img_size: int):
    def collate_fn(data_dict_list: List[Dict[str, Any]]):
        batch_size = len(data_dict_list)
        # [x, y, w, h, class_id]
        targets_list = [[[0, 0, 0, 0, 0]] for _ in range(batch_size)]
        labels = [data_dict["targets"] for data_dict in data_dict_list]
        for batch_id, items in enumerate(labels):
            if items is not None:
                targets_list[batch_id].extend(items)
        max_len = max([len(target) for target in targets_list])
        targets_padding_list = []
        for targets in targets_list:
            padding_targets = targets + [[0, 0, 0, 0, -1]] * (max_len - len(targets))
            targets_padding_list.append(padding_targets)

        target_tensor = torch.from_numpy(np.array(targets_padding_list))[:, 1:, :]
        input_feature_scale = torch.tensor([img_size, img_size, img_size, img_size])
        # 坐标转换为 输入层尺寸
        batch_target_tensor = target_tensor[:, :, :4].mul_(input_feature_scale)
        target_tensor[:, :, :4] = torchvision.ops.box_convert(boxes=batch_target_tensor,
                                                              in_fmt="cxcywh",
                                                              out_fmt="xyxy")
        batch_images = np.zeros(shape=(batch_size, 3, img_size, img_size), dtype=np.float32)
        for i, data_dict in enumerate(data_dict_list):
            img = data_dict["image"]
            batch_images[i] = img

        batch_targets = [torch.tensor(data_dict["targets"], dtype=torch.float32) if data_dict["targets"] is not None
                         else None for data_dict in data_dict_list]

        tensor_dict = {"batch_images": torch.tensor(batch_images),
                       "targets_v6": target_tensor,
                       "batch_targets": batch_targets}
        return tensor_dict

    return collate_fn


def yolo_v6_pose_collate_fn(img_size: int):
    def collate_fn(data_dict_list: List[Dict[str, Any]]):
        batch_size = len(data_dict_list)
        # [x, y, w, h, ,17 * 2, ...class_id]
        targets_list = [[[0] * (4 + 17 * 2 + 1)] for _ in range(batch_size)]
        labels = [data_dict["targets"] for data_dict in data_dict_list]
        for batch_id, items in enumerate(labels):
            if items is not None:
                targets_list[batch_id].extend(items)
        max_len = max([len(target) for target in targets_list])
        targets_padding_list = []
        for targets in targets_list:
            padding_targets = targets + [[0] * (4 + 17 * 2) + [-1]] * (max_len - len(targets))
            targets_padding_list.append(padding_targets)

        target_tensor = torch.from_numpy(np.array(targets_padding_list))[:, 1:, :]
        input_feature_scale = torch.tensor([img_size] * 4)
        # 坐标转换为 输入层尺寸 (包含 目标框 和 关键点 两部分)
        batch_target_tensor = target_tensor[:, :, :4].mul_(input_feature_scale)
        target_tensor[:, :, :4] = torchvision.ops.box_convert(boxes=batch_target_tensor,
                                                              in_fmt="cxcywh",
                                                              out_fmt="xyxy")

        target_tensor[:, :, 4:-1] = target_tensor[:, :, 4:-1].mul_(torch.tensor([img_size] * 2 * 17))

        batch_images = np.zeros(shape=(batch_size, 3, img_size, img_size), dtype=np.float32)
        for i, data_dict in enumerate(data_dict_list):
            img = data_dict["image"]
            batch_images[i] = img

        batch_targets = [torch.tensor(data_dict["targets"], dtype=torch.float32) if data_dict["targets"] is not None
                         else None for data_dict in data_dict_list]

        tensor_dict = {"batch_images": torch.tensor(batch_images),
                       "targets_v6": target_tensor,
                       "batch_targets": batch_targets}
        return tensor_dict

    return collate_fn


def yolo_v6_face_collate_fn(img_size: int):
    def collate_fn(data_dict_list: List[Dict[str, Any]]):
        batch_size = len(data_dict_list)
        # [x, y, w, h, ,5 * 2, ...class_id]
        targets_list = [[[0] * (4 + 5 * 2 + 1)] for _ in range(batch_size)]
        labels = [data_dict["targets"] for data_dict in data_dict_list]
        for batch_id, items in enumerate(labels):
            if items is not None:
                targets_list[batch_id].extend(items)
        max_len = max([len(target) for target in targets_list])
        targets_padding_list = []
        for targets in targets_list:
            padding_targets = targets + [[0] * (4 + 5 * 2) + [-1]] * (max_len - len(targets))
            targets_padding_list.append(padding_targets)

        target_tensor = torch.from_numpy(np.array(targets_padding_list))[:, 1:, :]
        input_feature_scale = torch.tensor([img_size] * 4)
        # 坐标转换为 输入层尺寸 (包含 目标框 和 关键点 两部分)
        batch_target_tensor = target_tensor[:, :, :4].mul_(input_feature_scale)
        target_tensor[:, :, :4] = torchvision.ops.box_convert(boxes=batch_target_tensor,
                                                              in_fmt="cxcywh",
                                                              out_fmt="xyxy")

        target_tensor[:, :, 4:-1] = target_tensor[:, :, 4:-1].mul_(torch.tensor([img_size] * 5 * 2))

        batch_images = np.zeros(shape=(batch_size, 3, img_size, img_size), dtype=np.float32)
        for i, data_dict in enumerate(data_dict_list):
            img = data_dict["image"]
            batch_images[i] = img

        batch_targets = [torch.tensor(data_dict["targets"], dtype=torch.float32) if data_dict["targets"] is not None
                         else None for data_dict in data_dict_list]

        tensor_dict = {"batch_images": torch.tensor(batch_images),
                       "targets_v6": target_tensor,
                       "batch_targets": batch_targets}
        return tensor_dict

    return collate_fn
