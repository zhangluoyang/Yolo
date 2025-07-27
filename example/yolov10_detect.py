"""
yolo v10 检测器
"""


import cv2
import torch
import numpy as np
from typing import *
from PIL import Image
import matplotlib.pyplot as plt
from luoyang.yolov10.YoloV10 import YoloV10
from luoyang.param.Param import Yolo10Param
import luoyang.utils.draw_utils as draw_utils
from luoyang.predict.YoloPredict import YoloPredict
import luoyang.utils.transformer_utils as transformer_utils
from luoyang.transformer.transformer import Transformer, RestoreSize


def process(image_path,
            transformers: List[Transformer]):
    data_dict = {"image_path": image_path}
    for transformer in transformers:
        transformer.transformer(data_dict=data_dict)
    return data_dict


def convert_yolo_to_luoyang():
    old_static = torch.load("/home/zhangluoyang/.cache/modelscope/hub/models/yolo_master/yolov10-weights/yolov10x.pt")
    old_tuples = list(old_static["model"].state_dict().items())

    param = Yolo10Param(m_type="x")
    param.class_num = 80
    yolo_task = YoloV10(param=param)

    new_tuples = list(yolo_task.state_dict().items())

    load_static = {}
    for (new_name, new_value), (old_name, old_value) in zip(new_tuples, old_tuples):
        load_static[new_name] = old_value

    yolo_task.load_state_dict(load_static, strict=False)
    torch.save(yolo_task.state_dict(), "/home/zhangluoyang/yolo_model/yolo_v10.pth")

def torch_detect():
    ## 加载模型
    param = Yolo10Param(m_type="x")
    param.class_num = 80
    param.conf_threshold = 0.3

    model_path = "/home/zhangluoyang/yolo_model/yolo_v10_x.pth"
    image_path = r"../resource/person.png"
    predict_transformers = transformer_utils.yolo_v3_predict_transformer(param=param)
    data_dict = process(image_path=image_path, transformers=predict_transformers)
    yolo_v10 = YoloV10(param=param)
    yolo_v10.load_state_dict(torch.load(model_path, map_location="cpu"))
    yolo_v10.eval()
    yolo_v10.cpu()
    predicts = yolo_v10.predict(tensor_dict={"batch_images": torch.tensor([data_dict["image"]]).cpu(),
                                             "threshold": param.conf_threshold})["predicts"]
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
    param = Yolo10Param(m_type="x")
    param.class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    param.class_num = 80
    param.conf_threshold = 0.3

    model_path = "/home/zhangluoyang/yolo_model/yolo_v10_x.pth"
    yolo_v10 = YoloV10(param=param)
    yolo_v10.load_state_dict(torch.load(model_path, map_location="cpu"))
    yolo_v10.eval()
    yolo_v10.fuse()
    onnx_path = "/home/zhangluoyang/yolo_model/yolo_v10/yolo.onnx"
    yolo_v10.to_onnx(onnx_path=onnx_path)


def onnx_detect():
    param = Yolo10Param(m_type="x")
    param.class_num = 80
    param.conf_threshold = 0.3
    predict_transformers = transformer_utils.yolo_v3_predict_transformer(param=param)
    image_path = r"../resource/person.png"
    data_dict = process(image_path=image_path, transformers=predict_transformers)
    onnx_path = "/home/zhangluoyang/yolo_model/yolo_v10/yolo.onnx"
    device = "cuda:0"
    yolo = YoloPredict(onnx_path=onnx_path,
                       device=device,
                       input_size=(param.img_size, param.img_size),
                       output_size=25,
                       conf_threshold=0.2,
                       need_nms=False,
                       )
    yolo.predict(feed_dict={"images": np.array([data_dict["image"]])})
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

if __name__ == "__main__":
    torch_detect()
    to_onnx()
    onnx_detect()