from src.param.Param import Yolo3Param, Yolo4Param, Yolo5Param
from src.transformer.transformer import *


def yolo_v3_train_transformer(param: Yolo3Param):
    return [ReadImage(),
            GenerateRect(),
            WarpAndResizeImage(target_height=param.img_size,
                               target_width=param.img_size),
            HorizontalFlip(),
            HsvArgument(),
            FilterBox(),
            GenerateBox(),
            ImageNorm()]


def yolo_v3_test_transformer(param: Yolo3Param):
    return [ReadImage(),
            GenerateRect(),
            ResizeImage(target_height=param.img_size,
                        target_width=param.img_size,
                        is_train=True),
            FilterBox(),
            GenerateBox(),
            ImageNorm()]


def yolo_v3_predict_transformer(param: Union[Yolo4Param, Yolo3Param, Yolo5Param]):
    return [ReadImage(),
            ResizeImage(target_height=param.img_size,
                        target_width=param.img_size,
                        is_train=False),
            ImageNorm()]


def yolo_v4_train_transformer(param: Yolo4Param):
    mosaic_head_transformer = [ReadImage(),
                               GenerateRect(),
                               HorizontalFlip()]

    mosaic_transformer = Mosaic(target_height=param.img_size,
                                target_width=param.img_size)

    mosaic_tail_transformer = [HsvArgument(),
                               FilterBox(),
                               GenerateBox(),
                               ImageNorm()]

    no_mosaic_transformer = [ReadImage(),
                             GenerateRect(),
                             WarpAndResizeImage(target_height=param.img_size,
                                                target_width=param.img_size),
                             HorizontalFlip(),
                             HsvArgument(),
                             FilterBox(),
                             GenerateBox(),
                             ImageNorm()]

    return [mosaic_head_transformer, mosaic_transformer, mosaic_tail_transformer], no_mosaic_transformer


def yolo_v4_test_transformer(param: Yolo4Param):
    return [ReadImage(),
            GenerateRect(),
            ResizeImage(target_height=param.img_size,
                        target_width=param.img_size,
                        is_train=True),
            FilterBox(),
            GenerateBox(),
            ImageNorm()]


def yolo_v4_predict_transformer(param: Yolo4Param):
    return [ReadImage(),
            ResizeImage(target_height=param.img_size,
                        target_width=param.img_size,
                        is_train=False),
            ImageNorm()]


def yolo_v5_train_transformer(param: Yolo5Param):
    mosaic_head_transformer = [ReadImage(),
                               GenerateRect(),
                               HorizontalFlip()]

    mosaic_transformer = Mosaic(target_height=param.img_size,
                                target_width=param.img_size)

    mosaic_tail_transformer = [HsvArgument(),
                               FilterBox(),
                               GenerateBox(),
                               ImageNorm(),
                               YoloV5Target(anchor_num=3,
                                            input_shape=param.img_size,
                                            class_num=param.class_num,
                                            strides=param.strides,
                                            anchors=param.anchors)]

    no_mosaic_transformer = [ReadImage(),
                             GenerateRect(),
                             WarpAndResizeImage(target_height=param.img_size,
                                                target_width=param.img_size),
                             HorizontalFlip(),
                             HsvArgument(),
                             FilterBox(),
                             GenerateBox(),
                             ImageNorm(),
                             YoloV5Target(anchor_num=3,
                                          input_shape=param.img_size,
                                          class_num=param.class_num,
                                          strides=param.strides,
                                          anchors=param.anchors)]

    return [mosaic_head_transformer, mosaic_transformer, mosaic_tail_transformer], no_mosaic_transformer


def yolo_v5_test_transformer(param: Yolo5Param):
    return [ReadImage(),
            GenerateRect(),
            ResizeImage(target_height=param.img_size,
                        target_width=param.img_size,
                        is_train=True),
            FilterBox(),
            GenerateBox(),
            ImageNorm(),
            YoloV5Target(anchor_num=3,
                         input_shape=param.img_size,
                         class_num=param.class_num,
                         strides=param.strides,
                         anchors=param.anchors)]
