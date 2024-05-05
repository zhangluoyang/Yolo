import cv2
import copy
import random
import numpy as np
from typing import *
from imgaug import augmenters as aug
from luoyang.transformer.transformer import Transformer
from luoyang.utils.math_utils import get_h_w_point_index


class ImageNorm(Transformer):

    def __init__(self, image_std: Union[List[float], None] = None,
                 image_mean: Union[List[float], None] = None):
        """

        :param image_std: 标准差
        :param image_mean: 均值
        """
        if image_std is None or image_mean is None:
            self.image_std = None
            self.image_mean = None
        else:
            self.image_std = np.array(image_std)
            self.image_mean = np.array(image_mean)

    def transformer(self, data_dict: Union[dict, List[dict]]):
        image = data_dict["image"] / 255
        if self.image_std is not None and self.image_mean is not None:
            image = (image - self.image_mean) / self.image_std
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(dtype=np.float32)
        data_dict["image"] = image


class ReadImage(Transformer):

    def transformer(self, data_dict: dict):
        if "image" not in data_dict:
            image_path = data_dict["image_path"]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            data_dict["image"] = image


class ResizeImage(Transformer):

    def __init__(self,
                 point_num: int,
                 target_height: int,
                 target_width: int,
                 is_train: bool = True,
                 correct_box: bool = True):
        """
        统一图片尺寸 并且归一化
        :param target_height:
        :param target_width:
        :param is_train:
        :param correct_box: 修复目标框
        """
        self.point_num = point_num
        self.target_height = target_height
        self.target_width = target_width
        self.is_train = is_train
        self.correct_box = correct_box

    def transformer(self, data_dict: dict):
        img = data_dict["image"]
        height, width, _ = img.shape
        # 先填充 在 resize (按照长宽比 优先考虑变换最小的)
        resize_ratio = min(1.0 * self.target_width / width, 1.0 * self.target_height / height)
        resize_w = int(resize_ratio * width)
        resize_h = int(resize_ratio * height)
        image_resized = cv2.resize(img, (resize_w, resize_h))
        # 将目标图像放在中间
        image_pad = np.full((self.target_height, self.target_width, 3), fill_value=128, dtype=np.uint8)
        dw = int((self.target_width - resize_w) / 2)
        dh = int((self.target_height - resize_h) / 2)

        image_pad[dh: resize_h + dh, dw: resize_w + dw, :] = image_resized
        # 归一化之后的图片
        data_dict["image"] = image_pad
        if not self.is_train:
            # 预测过程记录下原有的信息 用于预测后还原
            data_dict["ori_height"] = height
            data_dict["ori_width"] = width
            data_dict["resize_ratio"] = resize_ratio
            data_dict["resize_w"] = resize_w
            data_dict["resize_h"] = resize_h
            data_dict["ori_image"] = img
            data_dict["dw"] = dw
            data_dict["dh"] = dh
        if "rect_targets" in data_dict and data_dict["rect_targets"] is not None:
            data_dict["rect_targets"][:, 0] = data_dict["rect_targets"][:, 0] * resize_ratio + dw
            data_dict["rect_targets"][:, 1] = data_dict["rect_targets"][:, 1] * resize_ratio + dh
            data_dict["rect_targets"][:, 2] = data_dict["rect_targets"][:, 2] * resize_ratio + dw
            data_dict["rect_targets"][:, 3] = data_dict["rect_targets"][:, 3] * resize_ratio + dh

            x_point_index, y_point_index = get_h_w_point_index(point_num=self.point_num, start_id=0)
            # 关键点 去掉 mask 样本的影响
            points = data_dict["rect_targets"][..., 4: 4 + 2 * self.point_num]

            point_x_mask = (np.not_equal(points[:, x_point_index], -1)).astype(np.float32)

            point_y_mask = (np.not_equal(points[:, y_point_index], -1)).astype(np.float32)

            points[:, x_point_index] \
                = (points[:, x_point_index] * resize_ratio + dw) * point_x_mask + \
                  (1 - point_x_mask) * points[:, x_point_index]
            points[:, y_point_index] \
                = (points[:, y_point_index] * resize_ratio + dh) * point_y_mask + \
                  (1 - point_y_mask) * points[:, y_point_index]

            data_dict["rect_targets"][..., 4: 4 + 2 * self.point_num] = points


class RestoreSize(Transformer):
    """


    将预测的结果进行还原
    """

    def __init__(self, point_num: int):
        """

        :param point_num:
        """
        super(RestoreSize, self).__init__()
        self.point_num = point_num

    def transformer(self, data_dict: Dict[str, Any]):
        predict: np.ndarray = data_dict["predict"]
        if predict is not None:
            resize_ratio = data_dict["resize_ratio"]
            ori_height = data_dict["ori_height"]
            ori_width = data_dict["ori_width"]

            dw: float = data_dict["dw"]
            dh: float = data_dict["dh"]
            bbox = predict[:, :4].astype(np.int32)
            points = predict[:, 6: 6 + 2 * self.point_num]
            bbox[:, 0] = (bbox[:, 0] - dw) / resize_ratio
            bbox[:, 1] = (bbox[:, 1] - dh) / resize_ratio
            bbox[:, 2] = (bbox[:, 2] - dw) / resize_ratio
            bbox[:, 3] = (bbox[:, 3] - dh) / resize_ratio

            bbox[:, 0] = np.minimum(np.maximum(0, bbox[:, 0]), ori_width)
            bbox[:, 2] = np.minimum(np.maximum(0, bbox[:, 2]), ori_width)

            bbox[:, 1] = np.minimum(np.maximum(0, bbox[:, 1]), ori_height)
            bbox[:, 3] = np.minimum(np.maximum(0, bbox[:, 3]), ori_height)

            points[:, 0::2] = (points[:, 0::2] - dw) / resize_ratio
            points[:, 1::2] = (points[:, 1::2] - dh) / resize_ratio

            points[:, 0::2] = np.minimum(np.maximum(0, points[:, 0::2]), ori_width)
            points[:, 1::2] = np.minimum(np.maximum(0, points[:, 1::2]), ori_height)

            point_xs = points[:, 0::2]
            point_ys = points[:, 1::2]

            point_tuple_x_y = []
            point_confs = []
            for xs, ys, _conf in zip(point_xs, point_ys, predict[:, 6 + 2 * self.point_num: 6 + 3 * self.point_num]):
                _x_y_tuple = []
                for x, y in zip(xs, ys):
                    _x_y_tuple.append((x, y))
                point_tuple_x_y.append(_x_y_tuple)
                point_confs.append(_conf)
            data_dict["predict_boxs"] = bbox
            data_dict["predict_points"] = point_tuple_x_y
            data_dict["points_conf"] = point_confs

            data_dict["predict_label"] = predict[:, -1].astype(np.int32)


def _some_times(obj_f):
    return aug.Sometimes(0.5, obj_f)


class BlurAugment(Transformer):

    def __init__(self):
        self.blur = aug.OneOf([aug.GaussianBlur((0, 3.0)),
                               aug.AverageBlur(k=(2, 7)),
                               aug.MedianBlur(k=(3, 11))])

    def transformer(self, data_dict: dict):
        image = data_dict["image"]
        data_dict["image"] = self.blur.augment_image(image=image)


class SubtractMean(Transformer):

    def __init__(self, rgb_mean: Tuple[int, int, int] = (104, 117, 123)):
        self.rgb_mean = rgb_mean

    def transformer(self, data_dict: Union[dict, List[dict]]):
        """

        :param data_dict:
        :return:
        """
        image = data_dict["image"]
        image = image - self.rgb_mean
        data_dict["image"] = image


class SharpenAugment(Transformer):

    def __init__(self):
        self.sharpen = _some_times(obj_f=aug.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5)))

    def transformer(self, data_dict: dict):
        image = data_dict["image"]
        data_dict["image"] = self.sharpen.augment_image(image=image)


class GaussianNoise(Transformer):

    def __init__(self):
        self.noise = _some_times(aug.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255)))

    def transformer(self, data_dict: dict):
        image = data_dict["image"]
        data_dict["image"] = self.noise.augment_image(image=image)


class Add(Transformer):

    def __init__(self):
        self.add = _some_times(aug.Add(value=(-5, 5), per_channel=0.5))

    def transformer(self, data_dict: dict):
        image = data_dict["image"]
        data_dict["image"] = self.add.augment_image(image=image)


class Multiply(Transformer):

    def __init__(self):
        self.multiply = _some_times(aug.Multiply(mul=(0.8, 1.2), per_channel=0.5))

    def transformer(self, data_dict: dict):
        image = data_dict["image"]
        data_dict["image"] = self.multiply.augment_image(image=image)


class ContrastNormalization(Transformer):

    def __init__(self):
        self.contrast = _some_times(aug.ContrastNormalization(alpha=(0.8, 1.2), per_channel=0.5))

    def transformer(self, data_dict: dict):
        image = data_dict["image"]
        data_dict["image"] = self.contrast.augment_image(image=image)


class HsvArgument(Transformer):

    def __init__(self, h_gain: float = 0.5,
                 s_gain: float = 0.5,
                 v_gain: float = 0.5):
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain

    def transformer(self, data_dict: dict):
        image = data_dict["image"]
        if random.random() > 0.5:
            r = np.random.uniform(-1, 1, 3) * [self.h_gain, self.s_gain, self.v_gain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
            d_type = image.dtype
            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(d_type)
            lut_sat = np.clip(x * r[1], 0, 255).astype(d_type)
            lut_val = np.clip(x * r[2], 0, 255).astype(d_type)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=image)
            data_dict["image"] = image


class ChannelSwap(Transformer):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def transformer(self, data_dict: dict):
        image = data_dict["image"]
        if random.random() > 0.5:
            swap = self.perms[random.randrange(0, len(self.perms))]
            image[:, :, (0, 1, 2)] = image[:, :, swap]
            data_dict["image"] = image


class GenerateRectAndSevenTeenPoint(Transformer):
    """
    """

    def __init__(self, point_num):
        super(GenerateRectAndSevenTeenPoint, self).__init__()
        self.point_num = point_num

    def transformer(self, data_dict: dict):
        image = data_dict["image"]
        img_h, img_w, c = image.shape
        box_num = len(data_dict["box_dicts"])
        # [min_x, min_y, max_x, max_y, point_x, point_y,...,class_id]
        rect_targets = np.zeros(shape=(box_num, 4 + 2 * self.point_num + 1), dtype=np.float32)
        for _id, box in enumerate(data_dict["box_dicts"]):
            x_min, y_min, x_max, y_max = box["xMin"], box["yMin"], box["xMax"], box["yMax"]

            x_min = max(min(x_min, img_w - 1), 0)
            y_min = max(min(y_min, img_h - 1), 0)
            x_max = max(min(x_max, img_w - 1), 0)
            y_max = max(min(y_max, img_h - 1), 0)

            rect_targets[_id][0] = x_min
            rect_targets[_id][1] = y_min
            rect_targets[_id][2] = x_max
            rect_targets[_id][3] = y_max
            points = box["points"]
            confs = box["confs"]
            # key point
            key_points = []
            for point, conf in zip(points, confs):
                if conf in [2]:  # 仅仅预测可见标注点 (其余的部分mask掉)
                    key_points.extend(point)
                else:
                    key_points.extend([-1, -1])
            rect_targets[_id][4: -1] = key_points
            rect_targets[_id][-1] = 0  # person
        # 删除数据 预防后续出现错误
        del data_dict["box_dicts"]
        data_dict["rect_targets"] = rect_targets


class RandomCrop(Transformer):

    def __init__(self,
                 point_num: int,
                 no_crop_radio: float = 0.2,
                 scale_random_a: float = 0.3,
                 scale_random_b: float = 1.0,
                 min_size: int = 1,
                 img_size: int = 1024):
        """
        :param point_num:
        :param no_crop_radio:
        :param scale_random_a:
        :param scale_random_b:
        :param min_size
        :param img_size:
        """
        self.point_num = point_num
        self.no_crop_radio = no_crop_radio
        self.scale_random_a = scale_random_a
        self.scale_random_b = scale_random_b
        self.max_try_times = 250
        self.img_size = img_size
        self.min_size = min_size

    @staticmethod
    def matrix_iof(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """

        :param a:
        :param b:
        :return:
        """
        lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        return area_i / np.maximum(area_a[:, np.newaxis], 1)

    def transformer(self, data_dict: Union[dict, List[dict]]):
        """

        :param data_dict:
        :return:
        """
        image = data_dict["image"]
        height, width, _ = image.shape
        rect_targets = copy.deepcopy(data_dict["rect_targets"])
        c = rect_targets.shape[-1]
        boxes = rect_targets[..., :4]
        points = rect_targets[..., 4: 4 + 2 * self.point_num]
        labels = rect_targets[..., c - 1:c]
        for _ in range(self.max_try_times):
            if len(boxes) == 0:
                return
            if random.uniform(0, 1) <= 0.2:
                scale = 1
            else:
                scale = random.uniform(0.3, 1.)
            w = int(scale * width)
            h = int(scale * height)
            if width == w:
                l = 0
            else:
                l = random.randrange(width - w)
            if height == h:
                t = 0
            else:
                t = random.randrange(height - h)
            roi = np.array((l, t, l + w, t + h))
            value = self.matrix_iof(boxes, roi[np.newaxis])
            flag = (value >= 1)
            if not flag.any():
                continue
            boxes_t = boxes
            labels_t = labels
            points_t = points
            if boxes_t.shape[0] == 0:
                continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

            x_point_index, y_point_index = get_h_w_point_index(point_num=self.point_num, start_id=0)

            point_x_mask = (np.not_equal(points[:, x_point_index], -1)).astype(np.float32)

            point_y_mask = (np.not_equal(points[:, y_point_index], -1)).astype(np.float32)

            points[:, x_point_index] = (points[:, x_point_index] - roi[0]) * point_x_mask + \
                                       points[:, x_point_index] * (1 - point_x_mask)
            points[:, y_point_index] = (points[:, y_point_index] - roi[1]) * point_y_mask + \
                                       points[:, y_point_index] * (1 - point_y_mask)

            b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * self.img_size
            b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * self.img_size
            mask_b = np.minimum(b_w_t, b_h_t) >= self.min_size
            boxes_t = boxes_t[mask_b]
            labels_t = labels_t[mask_b]
            points_t = points_t[mask_b]
            if boxes_t.shape[0] == 0:
                continue
            data_dict["image"] = image_t
            data_dict["rect_targets"] = np.concatenate((boxes_t, points_t, labels_t), axis=-1)
            return


class HorizontalFlip(Transformer):

    def __init__(self, point_num: int):
        self.point_num = point_num
        self.index = [2, 3,
                      4, 5,
                      6, 7,
                      8, 9,
                      10, 11,
                      12, 13,
                      14, 15,
                      16, 17,
                      18, 19,
                      20, 21,
                      22, 23,
                      24, 25,
                      26, 27,
                      28, 29,
                      30, 31,
                      32, 33]
        self.flip_index = [4, 5,
                           2, 3,
                           8, 9,
                           6, 7,
                           12, 13,
                           10, 11,
                           16, 17,
                           14, 15,
                           20, 21,
                           18, 19,
                           24, 25,
                           22, 23,
                           28, 29,
                           26, 27,
                           32, 33,
                           30, 31]

    def transformer(self, data_dict: dict):
        image = data_dict["image"]

        if random.random() > 0.5:
            height, width, _ = image.shape
            image = cv2.flip(image, flipCode=1)
            data_dict["rect_targets"][:, [0, 2]] = width - data_dict["rect_targets"][:, [2, 0]]
            x_point_index, y_point_index = get_h_w_point_index(point_num=self.point_num)
            point_x_mask = (np.not_equal(data_dict["rect_targets"][:, x_point_index], -1)).astype(np.float32)
            data_dict["rect_targets"][:, x_point_index] = (width - data_dict["rect_targets"][:, x_point_index]) \
                                                          * point_x_mask + data_dict["rect_targets"][:,
                                                                           x_point_index] * (1 - point_x_mask)

            if self.point_num == 17:
                # 17 * 2 = 34
                key_points = copy.deepcopy(data_dict["rect_targets"])[:, 4:-1]
                data_dict["rect_targets"][:, 4:-1][:, self.index] = key_points[:, self.flip_index]
            else:
                raise NotImplemented
            data_dict["image"] = image


class GenerateBox(Transformer):

    def __init__(self, point_num: int):
        self.point_num = point_num

    def transformer(self, data_dict: Union[dict, List[dict]]):
        image = data_dict["image"]
        img_height, img_width, _ = image.shape
        if data_dict["rect_targets"] is not None:
            box_targets = np.array(copy.deepcopy(data_dict["rect_targets"]), dtype=np.float32)
            center_x = 0.5 * (box_targets[:, 0] + box_targets[:, 2])
            center_y = 0.5 * (box_targets[:, 1] + box_targets[:, 3])
            width = box_targets[:, 2] - box_targets[:, 0]
            height = box_targets[:, 3] - box_targets[:, 1]

            box_targets[:, 0] = center_x / img_width
            box_targets[:, 1] = center_y / img_height
            box_targets[:, 2] = width / img_width
            box_targets[:, 3] = height / img_height

            points = box_targets[:, 4: 4 + self.point_num * 2]
            x_point_index, y_point_index = get_h_w_point_index(point_num=self.point_num, start_id=0)

            point_x_mask = (np.not_equal(points[:, x_point_index], -1)).astype(np.float32)
            point_y_mask = (np.not_equal(points[:, y_point_index], -1)).astype(np.float32)

            points[:, x_point_index] = (points[:, x_point_index] / img_width) * point_x_mask + (1 - point_y_mask) * \
                                       points[:, x_point_index]

            points[:, y_point_index] = (points[:, y_point_index] / img_height) * point_y_mask + (1 - point_y_mask) * \
                                       points[:, x_point_index]

            box_targets[:, 4: 4 + self.point_num * 2] = points

            data_dict["targets"] = box_targets
        else:
            data_dict["targets"] = None


class FilterBox(Transformer):
    """
    过滤掉目标较小的
    """

    def __init__(self, min_width: int = 1, min_height: int = 1):
        super(FilterBox, self).__init__()
        self.min_width = min_width
        self.min_height = min_height

    def transformer(self, data_dict: Union[dict, List[dict]]):
        """

        :param data_dict:
        :return:
        """
        if "rect_targets" in data_dict and data_dict["rect_targets"] is not None:
            box = data_dict["rect_targets"]
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w >= self.min_width, box_h >= self.min_height)]
            if len(box) == 0:
                data_dict["rect_targets"] = None
            else:
                data_dict["rect_targets"] = box


class FilterPointBeforeResize(Transformer):
    """
    过滤掉超出图片范围的点 坐标 (需要在 Resize 之前搭配使用)
    """

    def __init__(self, point_num: int):
        super(FilterPointBeforeResize, self).__init__()
        self.point_num = point_num

    def transformer(self, data_dict: Union[dict, List[dict]]):
        img_h, img_w, _ = data_dict["image"].shape
        if "rect_targets" in data_dict and data_dict["rect_targets"] is not None:
            for rect_target in data_dict["rect_targets"]:
                rect_target[4:-1][0::2] = np.where(rect_target[4:-1][0::2] < 0, -1, rect_target[4:-1][0::2])
                rect_target[4:-1][0::2] = np.where(rect_target[4:-1][0::2] >= img_w, -1, rect_target[4:-1][0::2])

                rect_target[4:-1][1::2] = np.where(rect_target[4:-1][1::2] < 0, -1, rect_target[4:-1][1::2])
                rect_target[4:-1][1::2] = np.where(rect_target[4:-1][1::2] >= img_h, -1, rect_target[4:-1][1::2])


class YoloV5PoseTargetFormat(Transformer):
    """
    yolo v5 pose 形式 shape(targets) = (?, 1 + 1 + 4 + 2*point_num) [image_id, class_id, x, y, w, h, point_x, point_y, ...]
    """

    def __init__(self, point_num: int = 17):
        super(YoloV5PoseTargetFormat, self).__init__()
        self.point_num = point_num

    def transformer(self, data_dict: Union[dict, List[dict]]):
        # [?, x, y, w, h, ,..., class_id]
        targets = copy.deepcopy(data_dict["targets"])
        if targets is not None:
            # [batch_id, class_id, x, y, w, h, point_x, point_y, ...]
            targets_v5 = np.zeros((len(targets), 1 + 1 + 4 + 2 * self.point_num))
            targets_v5[:, 1] = targets[:, -1]
            targets_v5[:, 2:] = targets[:, :-1]
        else:
            targets_v5 = np.zeros((0, 1 + 1 + 4 + 2 * self.point_num))
        data_dict["targets_v5"] = targets_v5
