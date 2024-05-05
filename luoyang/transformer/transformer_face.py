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

            x_point_index, y_point_index = get_h_w_point_index(point_num=self.point_num)
            data_dict["rect_targets"][:, x_point_index] \
                = data_dict["rect_targets"][:, x_point_index] * resize_ratio + dw
            data_dict["rect_targets"][:, y_point_index] \
                = data_dict["rect_targets"][:, y_point_index] * resize_ratio + dh


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

            x_point_index, y_point_index = get_h_w_point_index(point_num=self.point_num, start_id=0)

            points[:, x_point_index] = (points[:, x_point_index] - dw) / resize_ratio
            points[:, y_point_index] = (points[:, y_point_index] - dh) / resize_ratio

            points[:, x_point_index] = np.minimum(np.maximum(0, points[:, x_point_index]), ori_width)
            points[:, y_point_index] = np.minimum(np.maximum(0, points[:, y_point_index]), ori_height)

            point_xs = points[:, x_point_index]
            point_ys = points[:, y_point_index]

            point_tuple_x_y = []
            for xs, ys in zip(point_xs, point_ys):
                _x_y_tuple = []
                for x, y in zip(xs, ys):
                    _x_y_tuple.append((x, y))
                point_tuple_x_y.append(_x_y_tuple)
            data_dict["predict_boxs"] = bbox
            data_dict["predict_points"] = point_tuple_x_y
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


class GenerateRect(Transformer):
    """
    生成中心点和长宽数据
    """

    def __init__(self):
        super(GenerateRect, self).__init__()

    def transformer(self, data_dict: dict):
        image = data_dict["image"]
        img_h, img_w, c = image.shape
        box_num = len(data_dict["box_dicts"])
        rect_targets = np.zeros(shape=(box_num, 5), dtype=np.float32)
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
            rect_targets[_id][-1] = box["cls"]
        # 删除数据 预防后续出现错误
        del data_dict["box_dicts"]
        data_dict["rect_targets"] = rect_targets


class GenerateRectAndFaceFivePoint(Transformer):
    """
    生成中心点和长宽数据 face 5 点坐标
    """

    def __init__(self, point_num):
        super(GenerateRectAndFaceFivePoint, self).__init__()
        self.point_num = point_num

    def transformer(self, data_dict: dict):
        image = data_dict["image"]
        img_h, img_w, c = image.shape
        box_num = len(data_dict["box_dicts"])
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
            # key point
            key_points = []
            xs = []
            ys = []
            for point in points:
                key_points.extend(point)
                xs.append(point[0])
                ys.append(point[1])
            rect_targets[_id][4: -1] = key_points
            rect_targets[_id][-1] = box["cls"]
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

            points[:, x_point_index] -= roi[0]
            points[:, y_point_index] -= roi[1]

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

    def transformer(self, data_dict: dict):
        image = data_dict["image"]
        if random.random() > 0.5:
            height, width, _ = image.shape
            image = cv2.flip(image, flipCode=1)
            data_dict["rect_targets"][:, [0, 2]] = width - data_dict["rect_targets"][:, [2, 0]]
            x_point_index, y_point_index = get_h_w_point_index(point_num=self.point_num)
            data_dict["rect_targets"][:, x_point_index] = width - data_dict["rect_targets"][:, x_point_index]
            if self.point_num == 5:
                key_points = copy.deepcopy(data_dict["rect_targets"])
                # left eye to right eye
                data_dict["rect_targets"][:, 6: 8] = key_points[:, 4: 6]
                # right eye to left eye
                data_dict["rect_targets"][:, 4: 6] = key_points[:, 6: 8]
                # left mouth to right mouth
                data_dict["rect_targets"][:, 10: 12] = key_points[:, 12: 14]
                # right to left mouth
                data_dict["rect_targets"][:, 12: 14] = key_points[:, 10: 12]
            else:
                raise NotImplemented
            data_dict["image"] = image


#
# class VerticalFlip(Transformer):
#     def transformer(self, data_dict: dict):
#         image = data_dict["image"]
#         if random.random() > 0.5:
#             height, width, _ = image.shape
#             image = cv2.flip(image, flipCode=0)
#             data_dict["rect_targets"][:, [1, 3]] = height - data_dict["rect_targets"][:, [3, 1]]
#             data_dict["image"] = image


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

            x_point_index, y_point_index = get_h_w_point_index(point_num=self.point_num)
            box_targets[:, x_point_index] = box_targets[:, x_point_index] / img_width
            box_targets[:, y_point_index] = box_targets[:, y_point_index] / img_height

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


class FilterPoint(Transformer):

    def __init__(self, point_num: int):
        self.point_num = point_num

    def transformer(self, data_dict: Union[dict, List[dict]]):
        """

        :param data_dict:
        :return:
        """
        if "rect_targets" in data_dict and data_dict["rect_targets"] is not None:
            rect_targets = data_dict["rect_targets"]
            x_point_index, y_point_index = get_h_w_point_index(point_num=self.point_num, start_id=4)
            for rect_target in rect_targets:
                min_x, min_y, max_x, max_y = rect_target[: 4]
                point_xs = rect_target[x_point_index]
                point_ys = rect_target[y_point_index]

                p_min_x, p_max_x = min(point_xs), max(point_xs)
                p_min_y, p_max_y = min(point_ys), max(point_ys)
                # 有一个点超出范围 整个关键点都无效
                if p_min_x < min_x or p_min_y < min_y or p_max_x > max_x or p_max_y > max_y:
                    rect_target[x_point_index] = -1
                    rect_target[y_point_index] = -1


class YoloV5FaceTarget(Transformer):

    def __init__(self,
                 anchor_num: int,
                 input_shape: int,
                 class_num: int,
                 point_num: int,
                 strides: List[int],
                 anchors: List[Tuple[int, int]],
                 anchors_mask=None):
        super(YoloV5FaceTarget, self).__init__()
        if anchors_mask is None:
            anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.anchors_mask = anchors_mask
        self.strides = strides
        self.anchor_num = anchor_num
        self.input_shape = input_shape
        self.class_num = class_num
        self.anchors = np.array(anchors)
        self.threshold = 4.0
        self.point_num = point_num

    def transformer(self, data_dict: Union[dict, List[dict]]):
        """

        :param data_dict:
        :return:
        """
        targets: np.ndarray = data_dict["targets"]
        y_trues = []
        box_best_ratio = []
        point_masks = []
        for _layer_id in range(len(self.strides)):
            stride = self.strides[len(self.strides) - _layer_id - 1]
            feature_size = self.input_shape // stride
            y_trues.append(
                np.zeros(shape=(self.anchor_num, feature_size, feature_size, self.class_num + 2 * self.point_num + 5)))
            point_masks.append(
                np.zeros(shape=(self.anchor_num, feature_size, feature_size), dtype=np.int64)
            )
            box_best_ratio.append(np.zeros((self.anchor_num, feature_size, feature_size), dtype='float32'))
        for _layer_id in range(len(self.strides)):
            stride = self.strides[len(self.strides) - _layer_id - 1]
            feature_size = self.input_shape // stride
            scale_anchors = self.anchors / stride
            if targets is not None and len(targets) != 0:
                batch_target = np.zeros_like(targets)
                # cx, cy, w, h in feature layer
                batch_target[:, [0, 2]] = targets[:, [0, 2]] * feature_size
                batch_target[:, [1, 3]] = targets[:, [1, 3]] * feature_size
                batch_target[:, 4] = targets[:, 4]

                # key point in feature layer
                batch_target[:, 4: 4 + 2 * self.point_num] = targets[:, 4: 4 + 2 * self.point_num] \
                                                             * feature_size

                ratios_of_gt_anchors = np.expand_dims(batch_target[:, 2:4], 1) / np.expand_dims(scale_anchors, 0)
                ratios_of_anchors_gt = np.expand_dims(scale_anchors, 0) / np.expand_dims(batch_target[:, 2:4], 1)
                # 4 ~ 4 [ture_box_num, 9]
                ratios = np.concatenate([ratios_of_gt_anchors, ratios_of_anchors_gt], axis=-1)
                max_ratios = np.max(ratios, axis=-1)

                for t, ratio in enumerate(max_ratios):
                    over_threshold = ratio < self.threshold
                    over_threshold[np.argmin(ratio)] = True

                    for k, mask in enumerate(self.anchors_mask[_layer_id]):
                        if not over_threshold[mask]:
                            continue

                        i = int(np.floor(batch_target[t, 0]))
                        j = int(np.floor(batch_target[t, 1]))

                        offsets = self.get_near_points(batch_target[t, 0], batch_target[t, 1], i, j)

                        for offset in offsets:
                            local_i = i + offset[0]
                            local_j = j + offset[1]

                            if local_i >= feature_size or local_i < 0 or local_j >= feature_size or local_j < 0:
                                continue

                            if box_best_ratio[_layer_id][k, local_j, local_i] != 0:
                                if box_best_ratio[_layer_id][k, local_j, local_i] > ratio[mask]:
                                    y_trues[_layer_id][k, local_j, local_i, :] = 0
                                else:
                                    continue
                            c = int(batch_target[t, -1])
                            y_trues[_layer_id][k, local_j, local_i, 0:4] = batch_target[t, 0:4]
                            y_trues[_layer_id][k, local_j, local_i, 4] = 1
                            y_trues[_layer_id][k, local_j, local_i, 5:5 + 2 * self.point_num] = \
                                batch_target[t, 4: 4 + 2 * self.point_num]
                            y_trues[_layer_id][k, local_j, local_i, 5 + 2 * self.point_num + c] = 1

                            box_best_ratio[_layer_id][k, local_j, local_i] = ratio[mask]
        _ids = list(range(3, 3 + len(self.strides)))
        _ids = _ids[::-1]
        for _id, y_true, point_mask in zip(_ids, y_trues, point_masks):
            anchor_index, h_index, w_index, = np.where(y_true[:, :, :, 5] > 0)
            point_mask[anchor_index, h_index, w_index] = 1
            data_dict["head_{0}_ground_true".format(_id)] = y_true
            data_dict["head_{0}_point_mask".format(_id)] = point_mask

    @staticmethod
    def get_near_points(x: float, y: float, i: int, j: int):
        """

        :param x:
        :param y:
        :param i:
        :param j:
        :return:
        """
        sub_x = x - i
        sub_y = y - j
        if sub_x > 0.5 and sub_y > 0.5:
            return [[0, 0], [1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y > 0.5:
            return [[0, 0], [-1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y < 0.5:
            return [[0, 0], [-1, 0], [0, -1]]
        else:
            return [[0, 0], [1, 0], [0, -1]]


class YoloV5FaceTargetFormat(Transformer):
    """
    yolo v5 face 形式 shape(targets) = (?, 6)  [image_id, class_id, x, y, w, h, ....]
    """

    def __init__(self, point_num: int = 5):
        super(YoloV5FaceTargetFormat, self).__init__()
        self.point_num = point_num

    def transformer(self, data_dict: Union[dict, List[dict]]):
        # [?, x, y, w, h, ,..., class_id]
        targets = copy.deepcopy(data_dict["targets"])
        if targets is not None:
            targets_v5 = np.zeros((len(targets), 1 + 1 + 4 + 2 * self.point_num))
            targets_v5[:, 1] = targets[:, -1]
            targets_v5[:, 2:] = targets[:, :-1]
        else:
            targets_v5 = np.zeros((0, 1 + 1 + 4 + 2 * 5))
        data_dict["targets_v5"] = targets_v5
