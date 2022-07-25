import os
from typing import List, Tuple


class Param(object):

    def __init__(self, batch_size: int):
        # 类别数目
        self.class_num: int = 20
        # 大于此值的框 不认为是有目标 (obj_mask=0) 也不认为没有目标 (no_obj_mask=0)
        self.ignore_threshold: float = 0.5
        # 当前程序的运行目录
        self.work_dir = os.getcwd().replace("\\", "/")
        # 特征层下采样 (yolo 输出层的下采样)
        self.strides: List[int] = NotImplemented

        self.anchors: List[List[Tuple[int, int]]] = NotImplemented

        self.anchor_index: List[Tuple[int, int, int]] = NotImplemented

        self.feature_size: List[int] = NotImplemented

        self.batch_size: int = batch_size

        self.fine_tune_batch_size: int = batch_size
        self.anchor_num = 3

        self.epochs = 200
        self.fine_epochs = 50
        self.train_epochs = 50

        self.early_stop = 50
        self.start_step = 0

        self.init_lr = 1e-4
        self.warm_lr = 2e-3
        self.min_lr = 2e-4

        self.conf_threshold: float = 0.1
        self.nms_threshold: float = 0.3
        self.ap_iou_threshold: float = 0.3


class Yolo3Param(Param):

    def __init__(self, batch_size: int = 12):
        """
        yolo v3 的配置参数
        """
        super(Yolo3Param, self).__init__(batch_size=batch_size)
        # darknet 53 结构
        self.dark_blocks: List[int] = [1, 2, 8, 8, 4]

        # darknet 53 路径
        self.darknet53_weight_path = "{0}/resource/darknet53_weights.pth".format(self.work_dir)

        self.darknet53_output_filters = [64, 128, 256, 512, 1024]
        # 锚点
        self.anchors: List[Tuple[int, int]] = [(10, 13), (16, 30), (33, 23),
                                               (30, 61), (62, 45), (59, 119),
                                               (116, 90), (156, 198), (373, 326)]
        # 当前层所属anchor下标
        self.anchor_index: List[Tuple[int, int, int]] = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]

        # 输入的图片名称
        self.input_image = "image"
        # yolo v3 下采样的倍数
        self.strides = [8, 16, 32]

        self.feature_size = [52, 26, 13]

        # yolo route 层输出名称
        self.route_names = ["route_3", "route_4", "route_5"]

        self.class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor']

        self.class_num = len(self.class_names)

        self.img_size: int = 416

        self.epochs = 100
        self.fine_epochs = 50
        self.train_epochs = 50

        self.batch_size = 14
        self.fine_tune_batch_size = 28


class Yolo4Param(Param):
    """
    yolo v4 的配置参数
    """

    def __init__(self, batch_size: int = 6):
        super(Yolo4Param, self).__init__(batch_size=batch_size)
        self.dark_blocks: List[int] = [1, 2, 8, 8, 4]
        self.darknet_csp_weight_path = "{0}/resource/csp_draknet_weights.pth".format(self.work_dir)

        self.darknet53_output_filters = [64, 128, 256, 512, 1024]
        # 锚点
        self.anchors: List[Tuple[int, int]] = [(10, 13), (16, 30), (33, 23),
                                               (30, 61), (62, 45), (59, 119),
                                               (116, 90), (156, 198), (373, 326)]
        # 当前层所属anchor下标
        self.anchor_index: List[Tuple[int, int, int]] = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]

        self.fine_tune_batch_size = 16

        # 输入的图片名称
        self.input_image = "image"
        # yolo v3 下采样的倍数
        self.strides = [8, 16, 32]

        self.feature_size = [52, 26, 13]

        # yolo route 层输出名称
        self.route_names = ["route_3", "route_4", "route_5"]

        self.class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor']

        self.class_num = len(self.class_names)

        self.img_size: int = 416

        self.epochs = 100
        self.fine_epochs = 50
        self.train_epochs = 50

        self.batch_size = 6
        self.fine_tune_batch_size = 12


class Yolo5Param(Param):
    """
    yolo v4 的配置参数
    """

    def __init__(self, batch_size: int = 6,
                 m_type: str = "l"):
        super(Yolo5Param, self).__init__(batch_size=batch_size)
        self.dark_blocks: List[int] = [1, 2, 8, 8, 4]
        self.m_type = m_type
        if m_type == "l":
            self.darknet_csp_weight_path = "{0}/resource/cspdarknet_l_backbone.pth".format(self.work_dir)
        elif m_type == "x":
            self.darknet_csp_weight_path = "{0}/resource/cspdarknet_x_backbone.pth".format(self.work_dir)
        elif m_type == "s":
            self.darknet_csp_weight_path = "{0}/resource/cspdarknet_s_backbone.pth".format(self.work_dir)
        elif m_type == "m":
            self.darknet_csp_weight_path = "{0}/resource/cspdarknet_m_backbone.pth".format(self.work_dir)
        else:
            raise NotImplemented

        self.darknet53_output_filters = [64, 128, 256, 512, 1024]
        # 锚点
        self.anchors: List[Tuple[int, int]] = [(10, 13), (16, 30), (33, 23),
                                               (30, 61), (62, 45), (59, 119),
                                               (116, 90), (156, 198), (373, 326)]
        # 当前层所属anchor下标
        self.anchor_index: List[Tuple[int, int, int]] = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]

        self.fine_tune_batch_size = 16

        # 输入的图片名称
        self.input_image = "image"
        # yolo v3 下采样的倍数
        self.strides = [8, 16, 32]

        self.feature_size = [52, 26, 13]

        # yolo route 层输出名称
        self.route_names = ["route_3", "route_4", "route_5"]

        self.class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                            'sheep', 'sofa', 'train', 'tvmonitor']

        self.class_num = len(self.class_names)

        self.img_size: int = 416

        self.epochs = 100
        self.fine_epochs = 50
        self.train_epochs = 50

        self.batch_size = 6
        self.fine_tune_batch_size = 12
