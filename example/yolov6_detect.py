"""
yolo v6 检测器
"""
import os
import cv2
import time
import torch
import numpy as np
from typing import *
from PIL import Image
import matplotlib.pyplot as plt
from luoyang.yolov6.YoloV6 import YoloV6
from luoyang.param.Param import Yolo6Param
import luoyang.utils.draw_utils as draw_utils
from luoyang.predict.YoloPredict import YoloPredict
import luoyang.utils.transformer_utils as transformer_utils
from luoyang.transformer.transformer import Transformer, RestoreSize


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def process(image_path,
            transformers: List[Transformer]):
    data_dict = {"image_path": image_path}
    for transformer in transformers:
        transformer.transformer(data_dict=data_dict)
    return data_dict


def torch_detect():
    param = Yolo6Param(m_type="s")
    param.conf_threshold = 0.3
    model_path = "../../yolo_model/yolo_v6_s.pth"
    image_path = r"../resource/person.png"
    predict_transformers = transformer_utils.yolo_v6_predict_transformer(param=param)
    data_dict = process(image_path=image_path, transformers=predict_transformers)

    yolo_v6 = YoloV6(param=param)
    yolo_v6.load_state_dict(torch.load(model_path, map_location="cpu"))
    device = "cpu"
    yolo_v6.eval()
    # yolo_v6.fuse()
    predicts = yolo_v6.predict(tensor_dict={"batch_images": torch.tensor([data_dict["image"]]).cpu()})["predicts"]
    data_dict["predict"] = predicts[0]
    restore_size = RestoreSize()
    restore_size.transformer(data_dict=data_dict)
    boxes = data_dict["predict_boxs"]
    class_ids = data_dict["predict_label"]
    class_names = [param.class_names[int(_id)] for _id in class_ids]
    img = draw_utils.draw_bbox_labels(data_dict["ori_image"], boxes=boxes, labels=class_names)
    plt.imshow(img)
    plt.waitforbuttonpress()


def to_onnx():
    """
    转换为 onnx
    :return:
    """
    param = Yolo6Param(m_type="s")
    model_path = "../../yolo_model/yolo_v6_s.pth"
    yolo_v6 = YoloV6(param=param)
    yolo_v6.load_state_dict(torch.load(model_path, map_location="cpu"))
    yolo_v6.eval()
    yolo_v6.fuse()
    yolo_v6.cuda()
    onnx_path = "../../yolo_model/yolo_v6_s.onnx"
    yolo_v6.to_onnx(onnx_path=onnx_path)


def onnx_detect():
    param = Yolo6Param(m_type="s")
    predict_transformers = transformer_utils.yolo_v6_predict_transformer(param=param)
    image_path = r"../resource/sheet.jpeg"
    data_dict = process(image_path=image_path, transformers=predict_transformers)
    onnx_path = "../../yolo_model/yolo_v6_s.onnx"
    device = "cpu"
    yolo = YoloPredict(onnx_path=onnx_path,
                       device=device,
                       input_size=(param.img_size, param.img_size),
                       output_size=25,
                       conf_threshold=0.2)
    predicts = yolo.predict(feed_dict={"images": np.array([data_dict["image"]])})
    data_dict["predict"] = predicts[0]
    restore_size = RestoreSize()
    restore_size.transformer(data_dict=data_dict)
    boxes = data_dict["predict_boxs"]
    class_ids = data_dict["predict_label"]
    class_names = [param.class_names[int(_id)] for _id in class_ids]

    img = draw_utils.draw_bbox_labels(data_dict["ori_image"], boxes=boxes, labels=class_names)
    plt.imshow(img)
    plt.waitforbuttonpress()


def onnx_video():
    param = Yolo6Param(m_type="s")
    param.conf_threshold = 0.3
    predict_transformers = transformer_utils.yolo_v6_predict_transformer(param=param)
    cap = cv2.VideoCapture(0)

    onnx_path = "../../yolo_model/yolo_v6_s.onnx"
    device = "cpu"
    yolo = YoloPredict(onnx_path=onnx_path,
                       device=device,
                       input_size=(param.img_size, param.img_size),
                       output_size=25,
                       conf_threshold=0.3)
    restore_size = RestoreSize()
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # yolo v6 预测
        data_dict = {"image": Image.fromarray(frame)}
        for transformer in predict_transformers:
            transformer.transformer(data_dict=data_dict)
        predicts = yolo.predict(feed_dict={"images": np.array([data_dict["image"]])})
        data_dict["predict"] = predicts[0]
        restore_size.transformer(data_dict=data_dict)
        boxes = data_dict["predict_boxs"]
        class_ids = data_dict["predict_label"]
        class_names = [param.class_names[int(_id)] for _id in class_ids]
        img = draw_utils.draw_bbox_labels(data_dict["ori_image"], boxes=boxes, labels=class_names)
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("yolo_v4_video", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    # to_onnx()
    torch_detect()
    # onnx_detect()