import os
import cv2
import torch
import numpy as np
from typing import *
from PIL import Image
import matplotlib.pyplot as plt
from luoyang.yolov5_face.YoloV5Face import YoloV5Face
from luoyang.param.Param import Yolo5FaceParam
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
    param = Yolo5FaceParam(m_type="m")
    model_path = "../../yolo_model/yolo_v5_face.pth"
    image_path = r"../resource/girl.jpeg"
    predict_transformers = transformer_utils.yolo_v5_face_predict_transformer(param=param)
    data_dict = process(image_path=image_path, transformers=predict_transformers)
    yolo_v5 = YoloV5Face(param=param)
    yolo_v5.load_state_dict(torch.load(model_path))
    yolo_v5.eval()
    yolo_v5.cuda()
    yolo_v5.fuse()
    predicts = yolo_v5.predict(tensor_dict={"batch_images": torch.tensor([data_dict["image"]]).cuda()})["predicts"]
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
    param = Yolo5FaceParam(m_type="m")
    model_path = "../../yolo_model/yolo_v5_face.pth"
    yolo_v5 = YoloV5Face(param=param)
    yolo_v5.load_state_dict(torch.load(model_path, map_location="cpu"))
    yolo_v5.eval()
    yolo_v5.fuse()
    onnx_path = "../../yolo_model/yolo_v5_face.onnx"
    yolo_v5.to_onnx(onnx_path=onnx_path)


def onnx_detect():
    param = Yolo5FaceParam(m_type="m")
    predict_transformers = transformer_utils.yolo_v3_predict_transformer(param=param)
    image_path = r"../resource/person.png"
    data_dict = process(image_path=image_path, transformers=predict_transformers)
    onnx_path = "../../yolo_model/yolo_v5_face.onnx"
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


def onnx_video():
    param = Yolo5FaceParam(m_type="m")
    param.conf_threshold = 0.3
    predict_transformers = transformer_utils.yolo_v3_predict_transformer(param=param)
    cap = cv2.VideoCapture(0)
    onnx_path = "../../yolo_model/yolo_v5_face.onnx"
    device = "cpu"
    yolo = YoloFacePredict(onnx_path=onnx_path,
                           point_num=param.point_num,
                           device=device,
                           input_size=(param.img_size, param.img_size),
                           output_size=25,
                           conf_threshold=0.3)
    restore_size = RestoreSize(point_num=param.point_num)
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # yolo 预测
        data_dict = {"image": frame}
        for transformer in predict_transformers:
            transformer.transformer(data_dict=data_dict)
        predicts = yolo.predict(feed_dict={"images": np.array([data_dict["image"]])})
        data_dict["predict"] = predicts[0]
        restore_size.transformer(data_dict=data_dict)
        boxes = data_dict["predict_boxs"]
        class_ids = data_dict["predict_label"]
        predict_points = data_dict["predict_points"]
        class_names = [param.class_names[int(_id)] for _id in class_ids]
        img = draw_utils.draw_bbox_labels(data_dict["ori_image"], boxes=boxes, labels=class_names)
        for points in predict_points:
            img = draw_utils.draw_points(img=img, points=points)
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("yolo_v5_face_video", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    torch_detect()
