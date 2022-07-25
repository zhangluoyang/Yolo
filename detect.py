import numpy as np
import os
from typing import List
import matplotlib.pyplot as plt
import torch
import time
from src.param.Param import Yolo4Param, Yolo3Param, Yolo5Param
import src.utils.draw_utils as draw_utils
from src.transformer.transformer import RestoreSize
from src.transformer.transformer import Transformer
import src.utils.transformer_utils as transformer_utils
from src.onnx.YoloPredict import YoloPredict
from src.task.YoloV3 import YoloV3
from src.task.YoloV4 import YoloV4
from src.task.YoloV5 import YoloV5

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def process(image_path,
            transformers: List[Transformer]):
    data_dict = {"image_path": image_path}
    for transformer in transformers:
        transformer.transformer(data_dict=data_dict)
    return data_dict


def torch_detect():
    param = Yolo4Param(batch_size=1)
    onnx_path = r"./yolo_v4/yolo_v4_model.pth"
    input_size = (416, 416)
    device = "cuda:0"
    output_size = 25
    predict_transformers = transformer_utils.yolo_v3_predict_transformer(param=param)
    _image_path = r"./resource/street.jpg"
    data_dict = process(image_path=_image_path, transformers=predict_transformers)
    yolo = YoloV4(param)
    yolo.to(device)
    yolo.load_state_dict(torch.load(onnx_path, map_location="cpu"))
    yolo.eval()
    predicts = yolo.predict(tensor_dict={"batch_images": torch.tensor([data_dict["image"]]).to(device)})["predicts"]
    data_dict["predict"] = predicts[0]
    restoreSize = RestoreSize()
    restoreSize.transformer(data_dict=data_dict)
    boxes = data_dict["predict_boxs"]
    class_ids = data_dict["predict_label"]
    class_names = [param.class_names[int(_id)] for _id in class_ids]
    img = draw_utils.draw_bbox_labels(data_dict["ori_image"], boxes=boxes, labels=class_names)
    plt.imshow(img)
    plt.waitforbuttonpress()


def onnx_detect():
    param = Yolo5Param(batch_size=1)
    predict_transformers = transformer_utils.yolo_v3_predict_transformer(param=param)
    _image_path = r"./resource/street.jpg"
    data_dict = process(image_path=_image_path, transformers=predict_transformers)

    yolo = YoloPredict(onnx_path=r"yolo_v5/yolo_v5.onnx",
                       device="cuda",
                       input_size=(param.img_size, param.img_size),
                       output_size=25,
                       feature_sizes=param.feature_size,
                       strides=param.strides,
                       conf_threshold=0.3)
    predicts = yolo.predict(feed_dict={"images": np.array([data_dict["image"]])})
    data_dict["predict"] = predicts[0]
    restoreSize = RestoreSize()
    restoreSize.transformer(data_dict=data_dict)
    boxes = data_dict["predict_boxs"]
    class_ids = data_dict["predict_label"]
    class_names = [param.class_names[int(_id)] for _id in class_ids]

    img = draw_utils.draw_bbox_labels(data_dict["ori_image"], boxes=boxes, labels=class_names)
    plt.imshow(img)
    plt.waitforbuttonpress()


if __name__ == "__main__":
    onnx_detect()
