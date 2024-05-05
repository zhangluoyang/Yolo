import cv2
import copy
import random
import numpy as np
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from typing import List, Tuple, Union, Dict, Any


class Transformer(object):

    def transformer(self, data_dict: Union[dict, List[dict]]):
        raise NotImplemented

    def batch_transformer(self, data_dict_list: Union[dict, List[dict]]) -> Dict[str, Any]:
        raise NotImplemented


class ReadImage(Transformer):

    @staticmethod
    def cvt_color(image):
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            return image
        else:
            image = image.convert('RGB')
            return image

    def transformer(self, data_dict: dict):
        if "image" not in data_dict:
            image_path = data_dict["image_path"]
            # print(image_path)
            image = Image.open(image_path)
            image = self.cvt_color(image)
            data_dict["image"] = image


class GenerateRect(Transformer):
    """
    生成中心点和长宽数据
    """

    def transformer(self, data_dict: dict):
        box_num = len(data_dict["box_dicts"])
        rect_targets = np.zeros(shape=(box_num, 5), dtype=np.int64)
        for _id, box in enumerate(data_dict["box_dicts"]):
            rect_targets[_id][0] = box["xMin"]
            rect_targets[_id][1] = box["yMin"]
            rect_targets[_id][2] = box["xMax"]
            rect_targets[_id][3] = box["yMax"]
            rect_targets[_id][4] = box["cls"]
        data_dict["rect_targets"] = rect_targets


class WarpAndResizeImage(Transformer):
    """
    图像的扭曲 和 resize
    """

    def __init__(self,
                 target_height: int,
                 target_width: int,
                 jitter: float = 0.3,
                 min_scale: float = 0.25,
                 max_scale: float = 2.0):
        super(WarpAndResizeImage, self).__init__()

        self.target_height = target_height
        self.target_width = target_width
        self.jitter = jitter
        self.min_scale = min_scale
        self.max_scale = max_scale

    @staticmethod
    def random(a: float = 0, b: float = 1) -> float:
        """

        :param a:
        :param b:
        :return:
        """
        return np.random.rand() * (b - a) + a

    def transformer(self, data_dict: Union[dict, List[dict]]):
        """

        :param data_dict:
        :return:
        """
        img: JpegImageFile = data_dict["image"]
        img_width, img_height = img.size
        jitter_r = self.random(1 - self.jitter, 1 + self.jitter) / self.random(1 - self.jitter, 1 + self.jitter)
        new_ar = img_width / img_height * jitter_r
        scale = self.random(self.min_scale, self.max_scale)
        if new_ar < 1:
            nh = int(scale * self.target_height)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * self.target_width)
            nh = int(nw / new_ar)

        img = img.resize((nw, nh), Image.BICUBIC)
        # 将图片 随机的贴到图中
        dx = int(self.random(0, img_width - nw))
        dy = int(self.random(0, img_height - nh))
        new_image = Image.new('RGB', (self.target_width, self.target_height), (128, 128, 128))
        new_image.paste(img, (dx, dy))
        img = new_image
        # 处理后的图像
        data_dict["image"] = img
        # 处理后的box
        if "rect_targets" in data_dict and data_dict["rect_targets"] is not None:
            box = data_dict["rect_targets"]
            box[:, [0, 2]] = box[:, [0, 2]] * nw / img_width + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / img_height + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > self.target_width] = self.target_width
            box[:, 3][box[:, 3] > self.target_height] = self.target_height
            data_dict["rect_targets"] = box


class ResizeImage(Transformer):

    def __init__(self,
                 target_height: int,
                 target_width: int,
                 is_train: bool = True):
        """
        统一图片尺寸 并且归一化
        :param target_height:
        :param target_width:
        :param is_train:
        """
        self.target_height = target_height
        self.target_width = target_width
        self.is_train = is_train

    def transformer(self, data_dict: dict):
        img: JpegImageFile = data_dict["image"]
        img_width, img_height = img.size
        # 先填充 在 resize (按照长宽比 优先考虑变换最小的)
        resize_ratio = min(1.0 * self.target_width / img_width, 1.0 * self.target_height / img_height)
        resize_w = int(resize_ratio * img_width)
        resize_h = int(resize_ratio * img_height)
        dw = int((self.target_width - resize_w) / 2)
        dh = int((self.target_height - resize_h) / 2)

        # 将目标图像放在中间
        image_resized = img.resize((resize_w, resize_h), Image.BICUBIC)
        new_image = Image.new("RGB", (self.target_width, self.target_height), (128, 128, 128))
        new_image.paste(image_resized, (dw, dh))
        # 归一化之后的图片
        data_dict["image"] = new_image
        if not self.is_train:
            # 预测过程记录下原有的信息 用于预测后还原
            data_dict["ori_height"] = img_height
            data_dict["ori_width"] = img_width
            data_dict["resize_ratio"] = resize_ratio
            data_dict["resize_w"] = resize_w
            data_dict["resize_h"] = resize_h
            data_dict["ori_image"] = img
            data_dict["dw"] = dw
            data_dict["dh"] = dh
        if "rect_targets" in data_dict and data_dict["rect_targets"] is not None:
            data_dict["rect_targets"][:, [0, 2]] = data_dict["rect_targets"][:, [0, 2]] * (resize_w / img_width) + dw
            data_dict["rect_targets"][:, [1, 3]] = data_dict["rect_targets"][:, [1, 3]] * (resize_h / img_height) + dh


class HorizontalFlip(Transformer):

    def transformer(self, data_dict: dict):
        """
        翻转
        :param data_dict:
        :return:
        """
        image = data_dict["image"]
        if random.random() > 0.5:
            img_width, img_height = image.size
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            data_dict["image"] = image
            if "rect_targets" in data_dict:
                data_dict["rect_targets"][:, [0, 2]] = img_width - data_dict["rect_targets"][:, [2, 0]]


class ToNumpy(Transformer):

    def transformer(self, data_dict: Union[dict, List[dict]]):
        data_dict["image"] = np.array(data_dict["image"], dtype=np.int8)


class ImageNorm(Transformer):

    def __init__(self, image_std: Union[List[float], None] = None,
                 image_mean: Union[List[float], None] = None):
        """

        :param image_std: 标准差
        :param image_mean: 均值
        """
        if image_std is not None and image_mean is not None:
            self.image_std = np.array(image_std)
            self.image_mean = np.array(image_mean)
        else:
            self.image_std = None
            self.image_mean = None

    def transformer(self,
                    data_dict: Union[dict, List[dict]]):
        """
        像素值归一化
        :param data_dict:
        :return:
        """
        image = np.array(data_dict["image"]) / 255
        if self.image_std is not None and self.image_mean is not None:
            image = (image - self.image_mean) / self.image_std
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(dtype=np.float32)
        data_dict["image"] = image


class HsvArgument(Transformer):

    def __init__(self, h_gain: float = 0.1,
                 s_gain: float = 0.7,
                 v_gain: float = 0.4):
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain

    def transformer(self, data_dict: dict):
        image = np.array(data_dict["image"], np.uint8)
        if random.random() > 0.5:
            r = np.random.uniform(-1, 1, 3) * [self.h_gain, self.s_gain, self.v_gain] + 1
            hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
            d_type = image.dtype
            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(d_type)
            lut_sat = np.clip(x * r[1], 0, 255).astype(d_type)
            lut_val = np.clip(x * r[2], 0, 255).astype(d_type)

            image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
            data_dict["image"] = Image.fromarray(image)


class RestoreSize(Transformer):
    """


    将预测的结果进行还原
    """

    def __init__(self):
        super(RestoreSize, self).__init__()

    def transformer(self, data_dict: Dict[str, Any]):
        predict: np.ndarray = data_dict["predict"]
        if predict is not None:
            resize_ratio = data_dict["resize_ratio"]
            ori_height = data_dict["ori_height"]
            ori_width = data_dict["ori_width"]

            dw: float = data_dict["dw"]
            dh: float = data_dict["dh"]
            bbox = predict[:, :4].astype(np.int32)
            bbox[:, 0] = (bbox[:, 0] - dw) / resize_ratio
            bbox[:, 1] = (bbox[:, 1] - dh) / resize_ratio
            bbox[:, 2] = (bbox[:, 2] - dw) / resize_ratio
            bbox[:, 3] = (bbox[:, 3] - dh) / resize_ratio

            bbox[:, 0] = np.minimum(np.maximum(0, bbox[:, 0]), ori_width)
            bbox[:, 2] = np.minimum(np.maximum(0, bbox[:, 2]), ori_width)

            bbox[:, 1] = np.minimum(np.maximum(0, bbox[:, 1]), ori_height)
            bbox[:, 3] = np.minimum(np.maximum(0, bbox[:, 3]), ori_height)

            data_dict["predict_boxs"] = bbox
            data_dict["predict_label"] = predict[:, -1].astype(np.int32)
        else:
            data_dict["predict_boxs"] = []
            data_dict["predict_label"] = []


class FilterBox(Transformer):
    """
    过滤掉目标较小的
    """

    def __init__(self, min_width: int = 1, min_height: int = 1):
        super(FilterBox, self).__init__()
        self.min_width = min_width
        self.min_height = min_height

    def transformer(self, data_dict: Union[dict, List[dict]]):
        """targets

        :param data_dict:
        :return:
        """
        if "rect_targets" in data_dict and data_dict["rect_targets"] is not None:
            box = data_dict["rect_targets"]
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > self.min_width, box_h > self.min_height)]
            if len(box) == 0:
                data_dict["rect_targets"] = None
            else:
                data_dict["rect_targets"] = box


class GenerateBox(Transformer):

    def transformer(self, data_dict: Union[dict, List[dict]]):
        image = data_dict["image"]
        img_width, img_height = image.size
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
            data_dict["targets"] = box_targets
        else:
            data_dict["targets"] = None


class YoloV7TargetFormat(Transformer):
    """
    yolo v7 形式 shape(targets) = (?, 6)  [image_id, class_id, x, y, w, h]
    """

    def transformer(self, data_dict: Union[dict, List[dict]]):
        targets = copy.deepcopy(data_dict["targets"])
        if targets is not None:
            targets_v7 = np.zeros((len(targets), 6))
            targets_v7[:, 1] = targets[:, -1]
            targets_v7[:, 2:] = targets[:, :4]
        else:
            targets_v7 = np.zeros((1, 6))
        data_dict["targets_v7"] = targets_v7


class Mosaic(Transformer):
    """
    数据增强
    """

    def __init__(self,
                 target_height: int = 416,
                 target_width: int = 416,
                 min_offset: float = 0.3,
                 max_offset: float = 0.7,
                 min_scale: float = 0.4,
                 max_scale: float = 1.0,
                 jitter: float = 0.3,
                 padding_pix: int = 128):
        super(Mosaic, self).__init__()
        self.target_height = target_height
        self.target_width = target_width
        self.min_offset = min_offset
        self.max_offset = max_offset
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.jitter = jitter
        self.padding_pix = padding_pix

    @staticmethod
    def merge_box(box_list, cut_x, cut_y):
        """

        :param box_list:
        :param cut_x:
        :param cut_y:
        :return:
        """
        merge_bbox = []
        for i in range(len(box_list)):
            for box in box_list[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cut_y or x1 > cut_x:
                        continue
                    if y2 >= cut_y >= y1:
                        y2 = cut_y
                    if x2 >= cut_x >= x1:
                        x2 = cut_x

                if i == 1:
                    if y2 < cut_y or x1 > cut_x:
                        continue
                    if y2 >= cut_y >= y1:
                        y1 = cut_y
                    if x2 >= cut_x >= x1:
                        x2 = cut_x

                if i == 2:
                    if y2 < cut_y or x2 < cut_x:
                        continue
                    if y2 >= cut_y >= y1:
                        y1 = cut_y
                    if x2 >= cut_x >= x1:
                        x1 = cut_x

                if i == 3:
                    if y1 > cut_y or x2 < cut_x:
                        continue
                    if y2 >= cut_y >= y1:
                        y2 = cut_y
                    if x2 >= cut_x >= x1:
                        x1 = cut_x
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def batch_transformer(self, data_dict_list: Union[dict, List[dict]]) -> Dict[str, Any]:
        """

        :param data_dict_list:
        :return:
        """
        min_offset_x = WarpAndResizeImage.random(self.min_offset, self.max_offset)
        min_offset_y = WarpAndResizeImage.random(self.min_offset, self.max_offset)

        image_list = []
        box_list = []
        index = 0

        for data_dict in data_dict_list:
            image = data_dict["image"]
            img_width, img_height = image.size
            jitter_r = WarpAndResizeImage.random(1 - self.jitter, 1 + self.jitter) \
                       / WarpAndResizeImage.random(1 - self.jitter,
                                                   1 + self.jitter)
            new_ar = img_width / img_height * jitter_r
            scale = WarpAndResizeImage.random(self.min_scale, self.max_scale)
            if new_ar < 1:
                nh = int(scale * self.target_height)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * self.target_width)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            if index == 0:
                dx = int(self.target_width * min_offset_x) - nw
                dy = int(self.target_height * min_offset_y) - nh
            elif index == 1:
                dx = int(self.target_width * min_offset_x) - nw
                dy = int(self.target_height * min_offset_y)
            elif index == 2:
                dx = int(self.target_width * min_offset_x)
                dy = int(self.target_height * min_offset_y)
            elif index == 3:
                dx = int(self.target_width * min_offset_x)
                dy = int(self.target_height * min_offset_y) - nh
            else:
                raise NotImplemented

            new_image = Image.new('RGB', (self.target_width, self.target_height), (self.padding_pix,
                                                                                   self.padding_pix,
                                                                                   self.padding_pix))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)
            index = index + 1

            if "rect_targets" in data_dict and data_dict["rect_targets"] is not None:
                box = data_dict["rect_targets"]
                box[:, [0, 2]] = box[:, [0, 2]] * nw / img_width + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / img_height + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > self.target_width] = self.target_width
                box[:, 3][box[:, 3] > self.target_height] = self.target_height
                box_list.append(box)
            image_list.append(image_data)

        cut_x = int(self.target_width * min_offset_x)
        cut_y = int(self.target_height * min_offset_y)

        new_image = np.zeros([self.target_height, self.target_width, 3])
        new_image[:cut_y, :cut_x, :] = image_list[0][:cut_y, :cut_x, :]
        new_image[cut_y:, :cut_x, :] = image_list[1][cut_y:, :cut_x, :]
        new_image[cut_y:, cut_x:, :] = image_list[2][cut_y:, cut_x:, :]
        new_image[:cut_y, cut_x:, :] = image_list[3][:cut_y, cut_x:, :]

        new_image = np.array(new_image, np.uint8)
        new_boxes = self.merge_box(box_list, cut_x, cut_y)
        if len(new_boxes) == 0:
            new_boxes = None
        else:
            new_boxes = np.array(new_boxes)
        return {"image": Image.fromarray(new_image),
                "rect_targets": new_boxes}


class YoloV5Target(Transformer):

    def __init__(self,
                 anchor_num: int,
                 input_shape: int,
                 class_num: int,
                 strides: List[int],
                 anchors: List[Tuple[int, int]],
                 anchors_mask=None):
        super(YoloV5Target, self).__init__()
        if anchors_mask is None:
            anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.anchors_mask = anchors_mask
        self.strides = strides
        self.anchor_num = anchor_num
        self.input_shape = input_shape
        self.class_num = class_num
        self.anchors = np.array(anchors)
        self.threshold = 4.0

    def transformer(self, data_dict: Union[dict, List[dict]]):
        """

        :param data_dict:
        :return:
        """
        targets: np.ndarray = data_dict["targets"]
        y_true = []
        box_best_ratio = []
        for _layer_id in range(len(self.strides)):
            stride = self.strides[len(self.strides) - _layer_id - 1]
            feature_size = self.input_shape // stride
            y_true.append(np.zeros(shape=(self.anchor_num, feature_size, feature_size, self.class_num + 5)))
            box_best_ratio.append(np.zeros((self.anchor_num, feature_size, feature_size), dtype='float32'))
        for _layer_id in range(len(self.strides)):
            stride = self.strides[len(self.strides) - _layer_id - 1]
            # 特征层尺寸
            feature_size = self.input_shape // stride
            # 缩放后的anchor大小
            scale_anchors = self.anchors / stride
            if targets is not None and len(targets) != 0:
                batch_target = np.zeros_like(targets)
                # cx, cy, w, h in feature layer
                batch_target[:, [0, 2]] = targets[:, [0, 2]] * feature_size
                batch_target[:, [1, 3]] = targets[:, [1, 3]] * feature_size
                batch_target[:, 4] = targets[:, 4]
                # 分别计算 ground_true 除以 anchor 和 anchor 除以 ground_true 最终取 (1/4~4) 作为候选anchor
                ratios_of_gt_anchors = np.expand_dims(batch_target[:, 2:4], 1) / np.expand_dims(scale_anchors, 0)
                ratios_of_anchors_gt = np.expand_dims(scale_anchors, 0) / np.expand_dims(batch_target[:, 2:4], 1)
                # 4 ~ 4 [ture_box_num, 9, 4]
                ratios = np.concatenate([ratios_of_gt_anchors, ratios_of_anchors_gt], axis=-1)
                # [ture_box_num, 9]
                max_ratios = np.max(ratios, axis=-1)
                # 此处预处理 速度很慢
                for t, ratio in enumerate(max_ratios):
                    # 满足 1/4~4的作为后选集
                    over_threshold = ratio < self.threshold
                    # 最接近的 也作为候选集
                    over_threshold[np.argmin(ratio)] = True

                    for k, mask in enumerate(self.anchors_mask[_layer_id]):
                        # 当前层anchor是否满足
                        if not over_threshold[mask]:
                            continue
                        # 取出左上角 坐标
                        i = int(np.floor(batch_target[t, 0]))
                        j = int(np.floor(batch_target[t, 1]))
                        # 根据中心点的距离 确定 上下左右的 候选 anchor
                        offsets = self.get_near_points(batch_target[t, 0], batch_target[t, 1], i, j)
                        for offset in offsets:
                            local_i = i + offset[0]
                            local_j = j + offset[1]
                            # 不超出特征层尺寸
                            if local_i >= feature_size or local_i < 0 or local_j >= feature_size or local_j < 0:
                                continue

                            if box_best_ratio[_layer_id][k, local_j, local_i] != 0:
                                # 保证每一个 anchor 只能与一个 ground_true 匹配
                                if box_best_ratio[_layer_id][k, local_j, local_i] > ratio[mask]:
                                    y_true[_layer_id][k, local_j, local_i, :] = 0
                                else:
                                    continue
                            c = int(batch_target[t, 4])
                            y_true[_layer_id][k, local_j, local_i, :4] = batch_target[t, :4]
                            # obj confidence
                            y_true[_layer_id][k, local_j, local_i, 4] = 1
                            # class
                            y_true[_layer_id][k, local_j, local_i, c + 5] = 1
                            box_best_ratio[_layer_id][k, local_j, local_i] = ratio[mask]
        for _id, _y_true in zip([5, 4, 3], y_true):
            data_dict["head_{0}_ground_true".format(_id)] = _y_true

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
