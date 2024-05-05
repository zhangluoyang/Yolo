import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from typing import *
from luoyang.yolov6_pose.YoloV6Pose import YoloV6Pose
from luoyang.param.Param import Yolo6PoseParam
import luoyang.utils.draw_utils as draw_utils
import luoyang.utils.transformer_utils as transformer_utils
from luoyang.transformer.transformer_pose import Transformer, RestoreSize
from luoyang.predict.YoloPredict import YoloPosePredict


def process(image_path,
            transformers: List[Transformer]):
    data_dict = {"image_path": image_path}
    for transformer in transformers:
        transformer.transformer(data_dict=data_dict)
    return data_dict


def torch_detect():
    param = Yolo6PoseParam(m_type="s")
    param.conf_threshold = 0.5
    model_path = "../../yolo_model/yolo_v6_s_pose.pth"
    image_path = r"../resource/person.png"
    predict_transformers = transformer_utils.yolo_v6_pose_predict_transformer(param=param)
    data_dict = process(image_path=image_path, transformers=predict_transformers)
    yolo_v6_pose = YoloV6Pose(param=param)
    yolo_v6_pose.load_state_dict(torch.load(model_path))
    yolo_v6_pose.eval()
    yolo_v6_pose.cuda()
    yolo_v6_pose.fuse()
    predicts = yolo_v6_pose.predict(tensor_dict={"batch_images": torch.tensor([data_dict["image"]]).cuda()})["predicts"]
    data_dict["predict"] = predicts[0]
    restore_size = RestoreSize(point_num=param.point_num)
    restore_size.transformer(data_dict=data_dict)
    boxes = data_dict["predict_boxs"]
    class_ids = data_dict["predict_label"]
    class_names = [param.class_names[int(_id)] for _id in class_ids]
    img = draw_utils.draw_bbox_labels(data_dict["ori_image"], boxes=boxes, labels=class_names)

    points_conf = data_dict["points_conf"]
    predict_points = data_dict["predict_points"]

    for _points_conf, _predict_points in zip(points_conf, predict_points):
        img = draw_utils.draw_skeleton(img=img,
                                       point_confs=_points_conf,
                                       points=_predict_points)

    plt.imshow(img)
    plt.waitforbuttonpress()


def to_onnx():
    """
    转换为 onnx
    :return:
    """
    param = Yolo6PoseParam(m_type="s")
    model_path = "../../yolo_model/yolo_v6_s_pose.pth"
    yolo_v6_face = YoloV6Pose(param=param)
    yolo_v6_face.load_state_dict(torch.load(model_path, map_location="cpu"))
    yolo_v6_face.eval()
    yolo_v6_face.fuse()
    yolo_v6_face.cpu()
    onnx_path = "../../yolo_model/yolo_v6_s_pose.onnx"
    yolo_v6_face.to_onnx(onnx_path=onnx_path)


def onnx_detect():
    param = Yolo6PoseParam(m_type="s")
    predict_transformers = transformer_utils.yolo_v3_predict_transformer(param=param)
    image_path = r"../resource/person.png"
    data_dict = process(image_path=image_path, transformers=predict_transformers)
    onnx_path = "../../yolo_model/yolo_v6_s_pose.onnx"
    device = "cpu"
    yolo = YoloPosePredict(onnx_path=onnx_path,
                           point_num=param.point_num,
                           device=device,
                           input_size=(param.img_size, param.img_size),
                           output_size=16,
                           conf_threshold=0.5)
    yolo.predict(feed_dict={"images": np.array([data_dict["image"]])})
    predicts = yolo.predict(feed_dict={"images": np.array([data_dict["image"]])})
    data_dict["predict"] = predicts[0]
    restore_size = RestoreSize(point_num=param.point_num)
    restore_size.transformer(data_dict=data_dict)

    boxes = data_dict["predict_boxs"]
    class_ids = data_dict["predict_label"]
    class_names = [param.class_names[int(_id)] for _id in class_ids]
    img = draw_utils.draw_bbox_labels(data_dict["ori_image"], boxes=boxes, labels=class_names)
    points_conf = data_dict["points_conf"]
    predict_points = data_dict["predict_points"]

    for _points_conf, _predict_points in zip(points_conf, predict_points):
        img = draw_utils.draw_skeleton(img=img,
                                       point_confs=_points_conf,
                                       points=_predict_points,
                                       confidence=0.3)
    plt.imshow(img)
    plt.waitforbuttonpress()


if __name__ == "__main__":
    # to_onnx()
    onnx_detect()