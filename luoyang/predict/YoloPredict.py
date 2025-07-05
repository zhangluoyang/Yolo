import torch
import numpy as np
from typing import *
from luoyang.predict.Predict import Predict
import luoyang.utils.torch_utils as torch_utils


class YoloPredict(Predict):
    def __init__(self, onnx_path: str,
                 device: str,
                 input_size: Tuple[int, int],
                 output_size: int,
                 conf_threshold: float = 0.1,
                 nms_threshold: float = 0.3,
                 outputs: Union[None, List[str]] = None):
        super(YoloPredict, self).__init__(onnx_path=onnx_path,
                                          device=device)
        self.output_size = output_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        self.outputs = outputs
        if outputs is None:
            self.outputs: List[str] = ["head_predicts"]

    def predict(self, feed_dict: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """

        :param feed_dict:
        :return:
        """
        output_nps = self.sess_predict(feed_dict=feed_dict)
        batch_nms_predicts = torch_utils.non_max_suppression(prediction=torch.tensor(output_nps).to(self.device),
                                                             conf_threshold=self.conf_threshold,
                                                             nms_threshold=self.nms_threshold,
                                                             img_size=self.input_size[0])
        return [None if nms_predicts is None else nms_predicts.detach().cpu().numpy()
                for nms_predicts in batch_nms_predicts]


class YoloFacePredict(Predict):
    def __init__(self, onnx_path: str,
                 device: str,
                 point_num: int,
                 input_size: Tuple[int, int],
                 output_size: int,
                 conf_threshold: float = 0.1,
                 nms_threshold: float = 0.3,
                 outputs: Union[None, List[str]] = None):
        super(YoloFacePredict, self).__init__(onnx_path=onnx_path,
                                              device=device)
        self.point_num = point_num
        self.output_size = output_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        self.outputs = outputs
        if outputs is None:
            self.outputs: List[str] = ["head_predicts"]

    def predict(self, feed_dict: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """

        :param feed_dict:
        :return:
        """
        output_nps = self.sess_predict(feed_dict=feed_dict)
        batch_nms_predicts = torch_utils.non_max_suppression_with_point(
            point_num=self.point_num,
            prediction=torch.tensor(output_nps).to(self.device),
            conf_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            img_size=self.input_size[0])
        return [None if nms_predicts is None else nms_predicts.detach().cpu().numpy()
                for nms_predicts in batch_nms_predicts]


class YoloPosePredict(Predict):
    def __init__(self, onnx_path: str,
                 device: str,
                 point_num: int,
                 input_size: Tuple[int, int],
                 output_size: int,
                 conf_threshold: float = 0.1,
                 nms_threshold: float = 0.3,
                 outputs: Union[None, List[str]] = None):
        super(YoloPosePredict, self).__init__(onnx_path=onnx_path,
                                              device=device)
        self.point_num = point_num
        self.output_size = output_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        self.outputs = outputs
        if outputs is None:
            self.outputs: List[str] = ["head_predicts"]

    def predict(self, feed_dict: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """

        :param feed_dict:
        :return:
        """
        output_nps = self.sess_predict(feed_dict=feed_dict)
        batch_nms_predicts = torch_utils.non_max_suppression_with_point_conf(point_num=self.point_num,
                                                                             prediction=torch.tensor(output_nps).to(
                                                                                 self.device),
                                                                             conf_threshold=self.conf_threshold,
                                                                             nms_threshold=self.nms_threshold,
                                                                             img_size=self.input_size[0])
        return [None if nms_predicts is None else nms_predicts.detach().cpu().numpy()
                for nms_predicts in batch_nms_predicts]
