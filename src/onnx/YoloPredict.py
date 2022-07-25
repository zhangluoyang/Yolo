import torch
import numpy as np
from typing import List, Dict, Tuple
from src.onnx.Predict import Predict
import src.utils.torch_utils as torch_utils


class YoloPredict(Predict):
    def __init__(self, onnx_path: str,
                 device: str,
                 input_size: Tuple[int, int],
                 output_size: int,
                 feature_sizes=None,
                 strides=None,
                 anchor_num: int = 3,
                 conf_threshold: float = 0.1,
                 nms_threshold: float = 0.3):
        super(YoloPredict, self).__init__(onnx_path=onnx_path,
                                          device=device)
        if strides is None:
            strides = [8, 16, 32]
        if feature_sizes is None:
            feature_sizes = [52, 26, 13]
        self.input_size = input_size
        self.anchor_num = anchor_num
        self.output_size = output_size
        self.strides = strides
        self.feature_sizes = feature_sizes
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.outputs: List[str] = ["head3_predicts", "head4_predicts", "head5_predicts"]

    def predict(self, feed_dict: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """

        :param feed_dict:
        :return:
        """
        output_nps = self.sess_run(outputs=self.outputs, feed_dict=feed_dict)
        batch_predict_list = []
        for output_np, stride, feature_size in zip(output_nps, self.strides, self.feature_sizes):
            _, anchor_num, height, width, output_size = np.shape(output_np)
            output_tensor = torch.tensor(output_np).to(self.device)
            output_tensor[..., :4] = output_tensor[..., :4] * stride
            batch_predict_list.append(output_tensor.view(-1, feature_size * feature_size * self.anchor_num,
                                                         self.output_size))
        batch_predicts = torch.cat(batch_predict_list, dim=1)
        batch_predicts[..., 0] = torch.clamp(batch_predicts[..., 0], min=0, max=self.input_size[0] - 1)
        batch_predicts[..., 1] = torch.clamp(batch_predicts[..., 1], min=0, max=self.input_size[1] - 1)
        batch_predicts[..., 2] = torch.clamp(batch_predicts[..., 2], min=0, max=self.input_size[0] - 1)
        batch_predicts[..., 3] = torch.clamp(batch_predicts[..., 3], min=0, max=self.input_size[1] - 1)

        batch_nms_predicts = torch_utils.non_max_suppression(prediction=batch_predicts,
                                                             conf_threshold=self.conf_threshold,
                                                             nms_threshold=self.nms_threshold,
                                                             img_size=self.input_size[0])

        return [None if nms_predicts is None else nms_predicts.detach().cpu().numpy()
                for nms_predicts in batch_nms_predicts]
