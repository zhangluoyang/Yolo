import onnxruntime
import numpy as np
from typing import List, Dict


class Predict(object):

    def __init__(self, onnx_path: str,
                 device: str):
        super(Predict, self).__init__()
        self.onnx_path = onnx_path
        self.device = device
        self.session = onnxruntime.InferenceSession(onnx_path,
                                                    providers=["CUDAExecutionProvider"])
        print(self.session.get_providers())

    def sess_run(self, outputs: List[str], feed_dict: Dict[str, np.ndarray]) -> List[np.ndarray]:
        return self.session.run(outputs, input_feed=feed_dict)

    def predict(self, feed_dict: Dict[str, np.ndarray]):
        raise NotImplemented
