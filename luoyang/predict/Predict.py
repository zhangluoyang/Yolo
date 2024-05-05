import onnxruntime
import numpy as np
from typing import *


class Predict(object):

    def __init__(self, onnx_path: str,
                 device: str):
        super(Predict, self).__init__()
        self.onnx_path = onnx_path
        self.device = device
        self.session = onnxruntime.InferenceSession(onnx_path,
                                                    providers=["CUDAExecutionProvider"])
        self.outputs = NotImplemented

    def sess_run(self, outputs: List[str], feed_dict: Dict[str, np.ndarray]) -> List[np.ndarray]:
        return self.session.run(outputs, input_feed=feed_dict)

    def io_biding_run(self):
        raise NotImplemented

    def predict(self, feed_dict: Dict[str, np.ndarray]):
        raise NotImplemented

    def sess_predict(self, feed_dict: Dict[str, np.ndarray]):
        output_nps = self.sess_run(outputs=self.outputs, feed_dict=feed_dict)[0]
        return output_nps
