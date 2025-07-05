import torch
import time
import os
from typing import *
import numpy as np
import luoyang.utils.torch_utils as torch_utils
from luoyang.predict.Predict import ManagePredict
from luoyang.transformer.transformer import ResizeImage, ImageNorm, DecodeImage
import psutil


class YoloPredict(ManagePredict):
    def __init__(self, gpu_id: Union[int, None] = None):
        super(YoloPredict, self).__init__(gpu_id=gpu_id)

        self.input_size = NotImplemented
        self.output_size = NotImplemented
        self.conf_threshold = NotImplemented
        self.nms_threshold = NotImplemented
        self.outputs = NotImplemented
        self.nms = NotImplemented
        self.labels: List[str] = NotImplemented

    @staticmethod
    def get_child_process_ids(parent_pid):
        # 获取特定父进程的所有子进程
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
        # 获取并返回子进程ID列表
        return [p.pid for p in children]

    def init_model(self, *args, **kwargs):
        super().init_model(*args, **kwargs)
        self.input_size = (self.config_dict["input_size"], self.config_dict["input_size"])
        self.output_size = self.config_dict["output_size"]
        self.conf_threshold = self.config_dict["conf_threshold"]
        self.nms_threshold = self.config_dict["nms_threshold"]
        self.outputs: List[str] = self.config_dict.get("outputs", ["head_predicts"])
        self.nms: bool = self.config_dict.get("nms", True)
        self.labels = self.config_dict["lables"]

        self.transformers = [DecodeImage(),
                             ResizeImage(target_height=self.input_size[0],
                                         target_width=self.input_size[1],
                                         is_train=False),
                             ImageNorm(image_mean=self.config_dict.get("mean", None),
                                       image_std=self.config_dict.get("std", None))]
        pid = os.getpid()
        all_pids = self.get_child_process_ids(os.getppid())

        index = all_pids.index(pid) - 1
        print("初始化work:{0}等待{1}秒".format(index, index * kwargs.get("init_time_delta", 10)))
        time.sleep(index * kwargs.get("init_time_delta", 10))
        print("初始化work:{0}".format(index))
        try:
            images = np.random.randn(kwargs.get("batch_size", 1), 3, self.input_size[0], self.input_size[0]).astype(
                dtype=np.float32)
            self.sess_predict(feed_dict={"images": images})
            print("初始化word:{0}成功".format(index))
        except Exception as e:
            "{0}".format(e)
            print("初始化word:{0}失败".format(index))

    def predict(self, feed_dicts: List[Dict[str, Any]]) -> List[Any]:
        """

        :param feed_dicts:
        :return:
        """

        for feed_dict in feed_dicts:
            for transformer in self.transformers:
                transformer.transformer(data_dict=feed_dict)

        images_list = [feed_dict["image"] for feed_dict in feed_dicts]
        images = np.array(images_list)
        print("输入尺寸:{0}, gid:{1}, pid:{2}".format(np.shape(images), os.getgid(), os.getpid()))
        output_nps = self.sess_predict(feed_dict={"images": images})
        return ["" for _ in range(len(output_nps))]
        # if self.nms:
        #     batch_nms_predicts = torch_utils.non_max_suppression(prediction=torch.tensor(output_nps),
        #                                                          conf_threshold=self.conf_threshold,
        #                                                          nms_threshold=self.nms_threshold,
        #                                                          img_size=self.input_size[0])
        # else:
        #     raise NotImplemented
        # outputs = []
        # for feed_dict, nms_predicts in zip(feed_dicts, batch_nms_predicts):
        #     feed_dict["predict"] = None if nms_predicts is None else nms_predicts.detach().cpu().numpy()
        #     self.restore_size.transformer(data_dict=feed_dict)
        #     outputs.append({"predict_boxs": feed_dict["predict_boxs"].tolist(),
        #                     "predict_label": [self.labels[label_index] for label_index in feed_dict["predict_label"]]})
        # return outputs
