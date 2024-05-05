"""
预测阶段的数据预处理
"""
import cv2
import math
import numpy as np
from typing import *


class Process(object):

    def __init__(self, target_size: Tuple[int, int],
                 image_std: Union[List[float], None] = None,
                 image_mean: Union[List[float], None] = None):
        super(Process, self).__init__()
        self.target_size = target_size

        self.image_std, self.image_std = None, None

        if image_std is not None:
            self.image_std = np.array(image_std)
        if image_mean is not None:
            self.image_std = np.array(image_std)

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplemented

    def restore(self, predict: np.ndarray, inv_m: np.ndarray) -> np.ndarray:
        raise NotImplemented


class ObjectDetectProcess(Process):
    """
    目标检测
    """

    def __init__(self, target_size: Tuple[int, int]):
        super(ObjectDetectProcess, self).__init__(target_size=target_size)

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """

        :param image: 待预测的图片
        :return: input_image (模型输入的矩阵), inv_m (放射变换的逆矩阵)
        """
        # 第一步 计算缩放比例
        height, width, _ = image.shape
        target_height, target_width = self.target_size
        scale_radio = min(target_height / height, target_width / width)
        # 第二步 计算x、y轴移动位置
        delta_x = math.ceil((target_width - scale_radio * width) / 2)
        delta_y = math.ceil((target_height - scale_radio * height) / 2)
        # 第三步 创建放射变换矩阵
        m = np.array([[scale_radio, 0, delta_x],
                      [0, scale_radio, delta_y]])
        warped_image = cv2.warpAffine(image, m, dsize=(target_height, target_width),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(128, 128, 128))
        # 第四步 计算放射变换逆矩阵 (坐标点的还原)
        inv_m = np.array([[1 / scale_radio, 0, -delta_x / scale_radio],
                          [0, 1 / scale_radio, -delta_y / scale_radio]])
        # inv_m = cv2.invertAffineTransform(m)

        warped_image = warped_image / 255
        warped_image = np.transpose(warped_image, (2, 0, 1))
        warped_image = warped_image.astype(dtype=np.float32)

        return warped_image, inv_m

    def restore(self, predict: Union[np.ndarray, None], inv_m: np.ndarray):
        """

        :param predict: yolo预测结果 [[min_x, min_y, max_x, max_y, obj_conf, class_conf, class_id]]
        :param inv_m:
        :return:
        """
        if predict is not None:
            xy_xy = predict[:, :4].reshape(-1, 2)
            xy_xy = np.dot(xy_xy, inv_m[:, :2]) + inv_m[:, 2]
            predict[:, :4] = xy_xy.reshape(-1, 4)
        return predict
