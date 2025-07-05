import onnxruntime
import numpy as np
import json
from typing import *
from onnxruntime import InferenceSession
from service_streamer import ManagedModel
from luoyang.transformer.transformer import Transformer, RestoreSize


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


class ManagePredict(ManagedModel):

    def __init__(self, gpu_id: Union[int, None] = None):
        super().__init__(gpu_id)
        self.session = NotImplemented
        self.outputs = NotImplemented
        self.config_dict: Dict[str, Any] = NotImplemented
        self.transformers: List[Transformer] = NotImplemented
        self.restore_size = RestoreSize()

    def init_model(self, *args, **kwargs):
        model_dir = kwargs["model_dir"]
        config_dict = json.load(open("{0}/config.json".format(model_dir), encoding="utf-8"))
        onnx_path = "{0}/{1}".format(model_dir, config_dict["onnx_name"])
        if self.gpu_id != -1:
            self.session = onnxruntime.InferenceSession(onnx_path,
                                                        providers=["CUDAExecutionProvider"])
        else:
            self.session = onnxruntime.InferenceSession(onnx_path)

        self.config_dict = config_dict

    def predict(self, feed_dicts: List[Dict[str, Any]]):
        raise NotImplemented

    def sess_predict(self, feed_dict: Dict[str, np.ndarray]):
        output_nps = self.sess_run(outputs=self.outputs, feed_dict=feed_dict)[0]
        return output_nps

    def sess_run(self, outputs: List[str], feed_dict: Dict[str, np.ndarray]) -> List[np.ndarray]:
        return self.session.run(outputs, input_feed=feed_dict)
