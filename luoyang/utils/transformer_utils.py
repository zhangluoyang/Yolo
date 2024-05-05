from luoyang.transformer.transformer import *
from luoyang.param.Param import *
import luoyang.transformer.transformer_face as transformer_face
import luoyang.transformer.transformer_pose as transformer_pose


def yolo_v3_train_transformer(param: Yolo3Param):
    return [ReadImage(),
            GenerateRect(),
            WarpAndResizeImage(target_height=param.img_size,
                               target_width=param.img_size),
            HorizontalFlip(),
            HsvArgument(),
            FilterBox(),
            GenerateBox(),
            ImageNorm(image_std=param.image_std,
                      image_mean=param.image_mean)]


def yolo_v3_test_transformer(param: Yolo3Param):
    return [ReadImage(),
            GenerateRect(),
            ResizeImage(target_height=param.img_size,
                        target_width=param.img_size,
                        is_train=True),
            FilterBox(),
            GenerateBox(),
            ImageNorm(image_mean=param.image_mean,
                      image_std=param.image_std)]


def yolo_v3_predict_transformer(param: Union[Yolo4Param, Yolo3Param, Yolo5Param]):
    return [ReadImage(),
            ResizeImage(target_height=param.img_size,
                        target_width=param.img_size,
                        is_train=False),
            ImageNorm(image_mean=param.image_mean,
                      image_std=param.image_std)]


def yolo_v4_train_transformer(param: Yolo4Param):
    mosaic_head_transformer = [ReadImage(),
                               GenerateRect(),
                               HorizontalFlip()]

    mosaic_transformer = Mosaic(target_height=param.img_size,
                                target_width=param.img_size)

    mosaic_tail_transformer = [HsvArgument(),
                               FilterBox(),
                               GenerateBox(),
                               ImageNorm(image_std=param.image_std,
                                         image_mean=param.image_mean)]

    no_mosaic_transformer = [ReadImage(),
                             GenerateRect(),
                             WarpAndResizeImage(target_height=param.img_size,
                                                target_width=param.img_size),
                             HorizontalFlip(),
                             HsvArgument(),
                             FilterBox(),
                             GenerateBox(),
                             ImageNorm(image_std=param.image_std,
                                       image_mean=param.image_mean)]

    return [mosaic_head_transformer, mosaic_transformer, mosaic_tail_transformer], no_mosaic_transformer


def yolo_v4_test_transformer(param: Yolo4Param):
    return [ReadImage(),
            GenerateRect(),
            ResizeImage(target_height=param.img_size,
                        target_width=param.img_size,
                        is_train=True),
            FilterBox(),
            GenerateBox(),
            ImageNorm(image_std=param.image_std,
                      image_mean=param.image_mean)]


def yolo_v4_predict_transformer(param: Yolo4Param):
    return [ReadImage(),
            ResizeImage(target_height=param.img_size,
                        target_width=param.img_size,
                        is_train=False),
            ImageNorm(image_std=param.image_std,
                      image_mean=param.image_mean)]


def yolo_v5_train_transformer(param: Yolo5Param):
    mosaic_head_transformer = [ReadImage(),
                               GenerateRect(),
                               HorizontalFlip()]

    mosaic_transformer = Mosaic(target_height=param.img_size,
                                target_width=param.img_size)

    mosaic_tail_transformer = [HsvArgument(),
                               FilterBox(),
                               GenerateBox(),
                               ImageNorm(image_mean=param.image_mean,
                                         image_std=param.image_std),
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
                             ImageNorm(image_mean=param.image_mean,
                                       image_std=param.image_std),
                             YoloV5Target(anchor_num=3,
                                          input_shape=param.img_size,
                                          class_num=param.class_num,
                                          strides=param.strides,
                                          anchors=param.anchors)]

    return [mosaic_head_transformer, mosaic_transformer, mosaic_tail_transformer], no_mosaic_transformer


def yolo_v5_test_transformer(param: Union[Yolo5Param, Yolo7Param]):
    return [ReadImage(),
            GenerateRect(),
            ResizeImage(target_height=param.img_size,
                        target_width=param.img_size,
                        is_train=True),
            FilterBox(),
            GenerateBox(),
            ImageNorm(image_mean=param.image_mean,
                      image_std=param.image_std),
            YoloV5Target(anchor_num=3,
                         input_shape=param.img_size,
                         class_num=param.class_num,
                         strides=param.strides,
                         anchors=param.anchors)]


def yolo_v5_predict_transformer(param: Yolo5Param):
    return [ReadImage(),
            ResizeImage(target_height=param.img_size,
                        target_width=param.img_size,
                        is_train=False),
            ImageNorm(image_mean=param.image_mean,
                      image_std=param.image_std)]


def get_yolo_v5_face_train_transformer(param: Union[Yolo5FaceParam, Yolo7FaceParam]):
    """
    数据增强以及预处理相关
    :param param:
    :return:
    """
    return [transformer_face.ReadImage(),
            transformer_face.GenerateRectAndFaceFivePoint(point_num=param.point_num),
            transformer_face.RandomCrop(point_num=param.point_num,
                                        img_size=param.img_size,
                                        min_size=1),
            transformer_face.HsvArgument(),
            transformer_face.HorizontalFlip(point_num=param.point_num),
            transformer_face.ResizeImage(point_num=param.point_num,
                                         target_height=param.img_size,
                                         target_width=param.img_size,
                                         is_train=True),
            transformer_face.FilterBox(min_width=1, min_height=1),
            transformer_face.FilterPoint(point_num=param.point_num),
            transformer_face.GenerateBox(point_num=param.point_num),
            transformer_face.ImageNorm(),
            transformer_face.YoloV5FaceTargetFormat()]


def get_yolo_v6_pose_train_transformer(param: Yolo6PoseParam):
    """

    :param param:
    :return:
    """
    return [transformer_pose.ReadImage(),
            transformer_pose.GenerateRectAndSevenTeenPoint(point_num=param.point_num),
            transformer_pose.RandomCrop(point_num=param.point_num,
                                        img_size=param.img_size,
                                        min_size=1),
            transformer_pose.HsvArgument(),
            transformer_pose.HorizontalFlip(point_num=param.point_num),
            transformer_pose.FilterPointBeforeResize(point_num=param.point_num),
            transformer_pose.ResizeImage(point_num=param.point_num,
                                         target_height=param.img_size,
                                         target_width=param.img_size,
                                         is_train=True),
            transformer_pose.FilterBox(min_width=1, min_height=1),
            transformer_pose.GenerateBox(point_num=param.point_num),
            transformer_pose.ImageNorm()]


def get_yolo_v6_pose_test_transformer(param: Yolo6PoseParam):
    """

        :param param:
        :return:
        """
    return [transformer_pose.ReadImage(),
            transformer_pose.GenerateRectAndSevenTeenPoint(point_num=param.point_num),
            transformer_pose.ResizeImage(point_num=param.point_num,
                                         target_height=param.img_size,
                                         target_width=param.img_size,
                                         is_train=True),
            transformer_pose.FilterBox(min_width=1, min_height=1),
            transformer_pose.GenerateBox(point_num=param.point_num),
            transformer_pose.ImageNorm()]


def yolo_v6_pose_predict_transformer(param: Yolo6PoseParam):
    return [transformer_pose.ReadImage(),
            transformer_pose.ResizeImage(target_height=param.img_size,
                                         target_width=param.img_size,
                                         is_train=False,
                                         point_num=param.point_num),
            transformer_pose.ImageNorm(image_mean=param.image_mean,
                                       image_std=param.image_std)]


def yolo_v6_face_train_transformer(param: Yolo6FaceParam):
    return [transformer_face.ReadImage(),
            transformer_face.GenerateRectAndFaceFivePoint(point_num=param.point_num),
            transformer_face.RandomCrop(point_num=param.point_num,
                                        img_size=param.img_size,
                                        min_size=1),
            transformer_face.HsvArgument(),
            transformer_face.HorizontalFlip(point_num=param.point_num),
            transformer_face.ResizeImage(point_num=param.point_num,
                                         target_height=param.img_size,
                                         target_width=param.img_size,
                                         is_train=True),
            transformer_face.FilterBox(min_width=1, min_height=1),
            transformer_face.FilterPoint(point_num=param.point_num),
            transformer_face.GenerateBox(point_num=param.point_num),
            transformer_face.ImageNorm()]


def yolo_v6_face_test_transformer(param: Yolo6FaceParam):
    return [transformer_face.ReadImage(),
            transformer_face.GenerateRectAndFaceFivePoint(point_num=param.point_num),
            transformer_face.ResizeImage(point_num=param.point_num,
                                         target_height=param.img_size,
                                         target_width=param.img_size,
                                         is_train=True),
            transformer_face.FilterBox(min_width=1, min_height=1),
            transformer_face.FilterPoint(point_num=param.point_num),
            transformer_face.GenerateBox(point_num=param.point_num),
            transformer_face.ImageNorm()]


def yolo_v6_face_predict_transformer(param: Yolo6FaceParam):
    return [transformer_face.ReadImage(),
            transformer_face.ResizeImage(target_height=param.img_size,
                                         target_width=param.img_size,
                                         is_train=False,
                                         point_num=param.point_num),
            transformer_face.ImageNorm(image_mean=param.image_mean,
                                       image_std=param.image_std)]


def get_yolo_v5_pose_train_transformer(param: Union[Yolo5PoseParam, Yolo7PoseParam]):
    """

    :param param:
    :return:
    """
    return [transformer_pose.ReadImage(),
            transformer_pose.GenerateRectAndSevenTeenPoint(point_num=param.point_num),
            transformer_pose.RandomCrop(point_num=param.point_num,
                                        img_size=param.img_size,
                                        min_size=1),
            transformer_pose.HsvArgument(),
            transformer_pose.HorizontalFlip(point_num=param.point_num),
            transformer_pose.FilterPointBeforeResize(point_num=param.point_num),
            transformer_pose.ResizeImage(point_num=param.point_num,
                                         target_height=param.img_size,
                                         target_width=param.img_size,
                                         is_train=True),
            transformer_pose.FilterBox(min_width=1, min_height=1),
            transformer_pose.GenerateBox(point_num=param.point_num),
            transformer_pose.ImageNorm(),
            transformer_pose.YoloV5PoseTargetFormat(point_num=param.point_num)]


def get_yolo_v5_pose_test_transformer(param: Union[Yolo5PoseParam, Yolo7PoseParam]):
    """

        :param param:
        :return:
        """
    return [transformer_pose.ReadImage(),
            transformer_pose.GenerateRectAndSevenTeenPoint(point_num=param.point_num),
            transformer_pose.ResizeImage(point_num=param.point_num,
                                         target_height=param.img_size,
                                         target_width=param.img_size,
                                         is_train=True),
            transformer_pose.FilterBox(min_width=1, min_height=1),
            transformer_pose.GenerateBox(point_num=param.point_num),
            transformer_pose.ImageNorm(),
            transformer_pose.YoloV5PoseTargetFormat(point_num=param.point_num)]


def yolo_v5_pose_predict_transformer(param: Union[Yolo5PoseParam, Yolo7PoseParam]):
    return [transformer_pose.ReadImage(),
            transformer_pose.ResizeImage(target_height=param.img_size,
                                         target_width=param.img_size,
                                         is_train=False,
                                         point_num=param.point_num),
            transformer_pose.ImageNorm(image_mean=param.image_mean,
                                       image_std=param.image_std)]


def get_yolo_v5_face_train_deprecated_transformer(param: Yolo5FaceParam):
    """
    数据增强以及预处理相关
    :param param:
    :return:
    """
    return [transformer_face.ReadImage(),
            transformer_face.GenerateRectAndFaceFivePoint(point_num=param.point_num),
            transformer_face.RandomCrop(point_num=param.point_num,
                                        img_size=param.img_size,
                                        min_size=1),
            transformer_face.HsvArgument(),
            transformer_face.HorizontalFlip(point_num=param.point_num),
            transformer_face.ResizeImage(point_num=param.point_num,
                                         target_height=param.img_size,
                                         target_width=param.img_size,
                                         is_train=True),
            transformer_face.FilterBox(min_width=1, min_height=1),
            transformer_face.FilterPoint(point_num=param.point_num),
            transformer_face.GenerateBox(point_num=param.point_num),
            transformer_face.ImageNorm(),
            transformer_face.YoloV5FaceTarget(anchor_num=3,
                                              point_num=param.point_num,
                                              input_shape=param.img_size,
                                              class_num=param.class_num,
                                              strides=param.strides,
                                              anchors=param.anchors)
            ]


def get_yolo_v5_face_test_transformer(param: Union[Yolo5FaceParam, Yolo7FaceParam]):
    return [transformer_face.ReadImage(),
            transformer_face.GenerateRectAndFaceFivePoint(point_num=param.point_num),
            transformer_face.ResizeImage(point_num=param.point_num,
                                         target_height=param.img_size,
                                         target_width=param.img_size,
                                         is_train=True),
            transformer_face.FilterBox(min_width=1, min_height=1),
            transformer_face.FilterPoint(point_num=param.point_num),
            transformer_face.GenerateBox(point_num=param.point_num),
            transformer_face.ImageNorm(),
            transformer_face.YoloV5FaceTargetFormat()]


def get_yolo_v5_face_test_deprecated_transformer(param: Yolo5FaceParam):
    return [transformer_face.ReadImage(),
            transformer_face.GenerateRectAndFaceFivePoint(point_num=param.point_num),
            transformer_face.ResizeImage(point_num=param.point_num,
                                         target_height=param.img_size,
                                         target_width=param.img_size,
                                         is_train=True),
            transformer_face.FilterBox(min_width=1, min_height=1),
            transformer_face.FilterPoint(point_num=param.point_num),
            transformer_face.GenerateBox(point_num=param.point_num),
            transformer_face.ImageNorm(),
            transformer_face.YoloV5FaceTarget(anchor_num=3,
                                              point_num=param.point_num,
                                              input_shape=param.img_size,
                                              class_num=param.class_num,
                                              strides=param.strides,
                                              anchors=param.anchors)]


def yolo_v5_face_predict_transformer(param: Union[Yolo5FaceParam, Yolo7FaceParam]):
    return [transformer_face.ReadImage(),
            transformer_face.ResizeImage(target_height=param.img_size,
                                         target_width=param.img_size,
                                         is_train=False,
                                         point_num=param.point_num),
            transformer_face.ImageNorm(image_mean=param.image_mean,
                                       image_std=param.image_std)]


def yolo_v6_train_transformer(param: Yolo6Param):
    mosaic_head_transformer = [
        ReadImage(),
        GenerateRect(),
        HorizontalFlip()]

    mosaic_transformer = Mosaic(target_height=param.img_size,
                                target_width=param.img_size)

    mosaic_tail_transformer = [HsvArgument(),
                               FilterBox(),
                               GenerateBox(),
                               ImageNorm(image_std=param.image_std,
                                         image_mean=param.image_mean)]

    no_mosaic_transformer = [ReadImage(),
                             GenerateRect(),
                             WarpAndResizeImage(target_height=param.img_size,
                                                target_width=param.img_size),
                             HorizontalFlip(),
                             HsvArgument(),
                             FilterBox(),
                             GenerateBox(),
                             ImageNorm(image_std=param.image_std,
                                       image_mean=param.image_mean)]
    return [mosaic_head_transformer, mosaic_transformer, mosaic_tail_transformer], no_mosaic_transformer


def yolo_v6_test_transformer(param: Yolo6Param):
    return [ReadImage(),
            GenerateRect(),
            ResizeImage(target_height=param.img_size,
                        target_width=param.img_size,
                        is_train=True),
            FilterBox(),
            GenerateBox(),
            ImageNorm(image_std=param.image_std,
                      image_mean=param.image_mean)]


def yolo_v6_predict_transformer(param: Yolo6Param):
    return [ReadImage(),
            ResizeImage(target_height=param.img_size,
                        target_width=param.img_size,
                        is_train=False),
            ImageNorm(image_mean=param.image_mean,
                      image_std=param.image_std)]


def yolo_v7_train_transformer(param: Yolo7Param):
    mosaic_head_transformer = [ReadImage(),
                               GenerateRect(),
                               HorizontalFlip()]

    mosaic_transformer = Mosaic(target_height=param.img_size,
                                target_width=param.img_size)

    mosaic_tail_transformer = [HsvArgument(),
                               FilterBox(),
                               GenerateBox(),
                               ImageNorm(image_mean=param.image_mean,
                                         image_std=param.image_std),
                               YoloV7TargetFormat()]

    no_mosaic_transformer = [ReadImage(),
                             GenerateRect(),
                             WarpAndResizeImage(target_height=param.img_size,
                                                target_width=param.img_size),
                             HorizontalFlip(),
                             HsvArgument(),
                             FilterBox(),
                             GenerateBox(),
                             ImageNorm(image_std=param.image_std,
                                       image_mean=param.image_mean),
                             YoloV7TargetFormat()]

    return [mosaic_head_transformer, mosaic_transformer, mosaic_tail_transformer], no_mosaic_transformer


def yolo_v7_test_transformer(param: Union[Yolo5Param, Yolo7Param]):
    return [ReadImage(),
            GenerateRect(),
            ResizeImage(target_height=param.img_size,
                        target_width=param.img_size,
                        is_train=True),
            FilterBox(),
            GenerateBox(),
            ImageNorm(image_std=param.image_std,
                      image_mean=param.image_mean),
            YoloV7TargetFormat()]


def yolo_v7_predict_transformer(param: Yolo7Param):
    return [ReadImage(),
            ResizeImage(target_height=param.img_size,
                        target_width=param.img_size,
                        is_train=False),
            ImageNorm(image_mean=param.image_mean,
                      image_std=param.image_std)]
