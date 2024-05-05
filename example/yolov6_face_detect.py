import os
import cv2
import torch
import numpy as np
from typing import *
from PIL import Image
import matplotlib.pyplot as plt
from luoyang.yolov6_face.YoloV6Face import YoloV6Face
from luoyang.param.Param import Yolo6FaceParam
import luoyang.utils.draw_utils as draw_utils
from luoyang.predict.YoloPredict import YoloFacePredict
import luoyang.utils.transformer_utils as transformer_utils
from luoyang.transformer.transformer_face import Transformer, RestoreSize

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def process(image_path,
            transformers: List[Transformer]):
    data_dict = {"image_path": image_path}
    for transformer in transformers:
        transformer.transformer(data_dict=data_dict)
    return data_dict


def torch_detect():
    param = Yolo6FaceParam(m_type="s")
    # param.conf_threshold = 0.5
    model_path = "../../yolo_model/yolo_v6_s_face.pth"
    image_path = r"../resource/person.png"
    predict_transformers = transformer_utils.yolo_v6_face_predict_transformer(param=param)
    data_dict = process(image_path=image_path, transformers=predict_transformers)
    yolo_v6_face = YoloV6Face(param=param)
    yolo_v6_face.load_state_dict(torch.load(model_path))
    yolo_v6_face.eval()
    yolo_v6_face.cuda()
    predicts = yolo_v6_face.predict(tensor_dict={"batch_images": torch.tensor([data_dict["image"]]).cuda()})["predicts"]
    data_dict["predict"] = predicts[0]
    restore_size = RestoreSize(point_num=param.point_num)
    restore_size.transformer(data_dict=data_dict)
    boxes = data_dict["predict_boxs"]
    class_ids = data_dict["predict_label"]
    predict_points = data_dict["predict_points"]
    class_names = [param.class_names[int(_id)] for _id in class_ids]
    img = draw_utils.draw_bbox_labels(data_dict["ori_image"], boxes=boxes, labels=class_names)
    for points in predict_points:
        img = draw_utils.draw_points(img=img, points=points)
    plt.imshow(img)
    plt.waitforbuttonpress()


def to_onnx():
    """
    转换为 onnx
    :return:
    """
    param = Yolo6FaceParam(m_type="s")
    model_path = "../../yolo_model/yolo_v6_s_face.pth"
    yolo_v6_face = YoloV6Face(param=param)
    yolo_v6_face.load_state_dict(torch.load(model_path, map_location="cpu"))
    yolo_v6_face.eval()
    yolo_v6_face.fuse()
    yolo_v6_face.cuda()
    onnx_path = "../../yolo_model/yolo_v6_s_face.onnx"
    yolo_v6_face.to_onnx(onnx_path=onnx_path)


def onnx_detect():
    param = Yolo6FaceParam(m_type="s")
    predict_transformers = transformer_utils.yolo_v3_predict_transformer(param=param)
    image_path = r"../resource/person.png"
    data_dict = process(image_path=image_path, transformers=predict_transformers)
    onnx_path = "../../yolo_model/yolo_v6_s_face.onnx"
    device = "cpu"
    yolo = YoloFacePredict(onnx_path=onnx_path,
                           point_num=param.point_num,
                           device=device,
                           input_size=(param.img_size, param.img_size),
                           output_size=16,
                           conf_threshold=0.3)
    yolo.predict(feed_dict={"images": np.array([data_dict["image"]])})
    predicts = yolo.predict(feed_dict={"images": np.array([data_dict["image"]])})
    data_dict["predict"] = predicts[0]
    restore_size = RestoreSize(point_num=param.point_num)
    restore_size.transformer(data_dict=data_dict)

    boxes = data_dict["predict_boxs"]
    class_ids = data_dict["predict_label"]
    predict_points = data_dict["predict_points"]
    class_names = [param.class_names[int(_id)] for _id in class_ids]
    img = draw_utils.draw_bbox_labels(data_dict["ori_image"], boxes=boxes, labels=class_names)
    for points in predict_points:
        img = draw_utils.draw_points(img=img, points=points)
    plt.imshow(img)
    plt.waitforbuttonpress()


if __name__ == "__main__":
    # torch_detect()
    # to_onnx()
    onnx_detect()
