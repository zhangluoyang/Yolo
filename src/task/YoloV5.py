import torch
from typing import List, Dict, Union, Any
from src.param.Param import Yolo5Param
from src.model.Layer import MetricLayer
from src.model.yolo.YoloV5Body import YoloV5Body
from src.model.yolo.YoloLoss import YoloV5Loss
from src.model.yolo.YoloHead import YoloV5Head
from src.model.Layer import TaskLayer, LossLayer
import src.utils.torch_utils as torch_utils
from src.model.metric.DetectMetric import DetectMetric
import time

_depth_dict = {'s': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33}
_width_dict = {'s': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25}


class YoloV5(TaskLayer):

    def __init__(self, param: Yolo5Param,
                 anchor_num: int = 3):
        super(YoloV5, self).__init__()
        self.param = param

        self.body_output_size = 5 + self.param.class_num
        self.body_net = YoloV5Body(anchor_num=anchor_num,
                                   num_classes=param.class_num,
                                   wid_mul=_width_dict[self.param.m_type],
                                   dep_mul=_depth_dict[self.param.m_type],
                                   pretrained_path=self.param.darknet_csp_weight_path)

        self.yolo_head3 = YoloV5Head(stride=self.param.strides[0],
                                     input_size=(self.param.img_size // 8, self.param.img_size // 8),
                                     yolo_output_size=self.body_output_size,
                                     anchors=param.anchors[: 3])

        self.yolo_head4 = YoloV5Head(stride=self.param.strides[1],
                                     yolo_output_size=self.body_output_size,
                                     input_size=(self.param.img_size // 16, self.param.img_size // 16),
                                     anchors=param.anchors[3: 6])

        self.yolo_head5 = YoloV5Head(stride=self.param.strides[2],
                                     input_size=(self.param.img_size // 32, self.param.img_size // 32),
                                     yolo_output_size=self.body_output_size,
                                     anchors=param.anchors[6:])

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """

        :param images:
        :return:
        """
        out5, out4, out3 = self.body_net(images)

        head5_predicts = self.yolo_head5(out5)
        head4_predicts = self.yolo_head4(out4)
        head3_predicts = self.yolo_head3(out3)
        return [head3_predicts, head4_predicts, head5_predicts]

    def _feed_forward(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]):
        """

        :param tensor_dict:
        :return:
        """
        out5, out4, out3 = self.body_net(tensor_dict["batch_images"])
        head5_predicts = self.yolo_head5(out5)
        head4_predicts = self.yolo_head4(out4)
        head3_predicts = self.yolo_head3(out3)
        tensor_dict["head3_predicts"] = head3_predicts
        tensor_dict["head4_predicts"] = head4_predicts
        tensor_dict["head5_predicts"] = head5_predicts
        return tensor_dict

    def feed_forward(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]):
        """

        :param tensor_dict:
        :return:
        """
        tensor_dict = self._feed_forward(tensor_dict=tensor_dict)
        return tensor_dict

    def predict(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]) -> Dict[str, Any]:
        """

        :param tensor_dict:
        :return:
        """
        batch_predict_list = []
        if "batch_targets" in tensor_dict:
            tensor_dict = self.feed_forward(tensor_dict=tensor_dict)
        else:
            tensor_dict = self._feed_forward(tensor_dict=tensor_dict)
        for name, stride, feature_size in zip(["head3_predicts", "head4_predicts", "head5_predicts"],
                                              self.param.strides,
                                              self.param.feature_size):
            predict = torch.clone(tensor_dict[name])
            predict[..., :4] = predict[..., :4] * stride
            batch_predict_list.append(predict.view(-1, feature_size * feature_size * 3, self.body_output_size))
        batch_predicts = torch.cat(batch_predict_list, dim=1)
        batch_predicts[..., 0] = torch.clamp(batch_predicts[..., 0], min=0, max=self.param.img_size - 1)
        batch_predicts[..., 1] = torch.clamp(batch_predicts[..., 1], min=0, max=self.param.img_size - 1)
        batch_predicts[..., 2] = torch.clamp(batch_predicts[..., 2], min=0, max=self.param.img_size - 1)
        batch_predicts[..., 3] = torch.clamp(batch_predicts[..., 3], min=0, max=self.param.img_size - 1)

        batch_nms_predicts = torch_utils.non_max_suppression(prediction=batch_predicts,
                                                             conf_threshold=self.param.conf_threshold,
                                                             nms_threshold=self.param.nms_threshold,
                                                             img_size=self.param.img_size)

        tensor_dict["predicts"] = [None if nms_predicts is None else nms_predicts.detach().cpu().numpy()
                                   for nms_predicts in batch_nms_predicts]
        return tensor_dict

    def build_loss_layer(self) -> List[LossLayer]:
        # return [LambdaLoss(name="loss")]
        head_3_loss = YoloV5Loss(loss_name="3",
                                 head_predict_name="head3_predicts",
                                 ground_true_name="head_3_ground_true",
                                 weight=4.0)
        head_4_loss = YoloV5Loss(loss_name="4",
                                 head_predict_name="head4_predicts",
                                 ground_true_name="head_4_ground_true",
                                 weight=1.0)
        head_5_loss = YoloV5Loss(loss_name="5",
                                 head_predict_name="head5_predicts",
                                 ground_true_name="head_5_ground_true",
                                 weight=0.4)

        return [head_5_loss, head_4_loss, head_3_loss]

    def build_metric_layer(self) -> List[MetricLayer]:
        detect_metric = DetectMetric(image_size=(self.param.img_size, self.param.img_size),
                                     class_names=self.param.class_names,
                                     predict_name="predicts",
                                     target_name="batch_targets",
                                     conf_threshold=self.param.conf_threshold,
                                     ap_iou_threshold=self.param.ap_iou_threshold,
                                     nms_threshold=self.param.nms_threshold)
        return [detect_metric]

    def update_fine_tune_param(self):
        for param in self.body_net.backbone.parameters():
            param.requires_grad = False

    def update_train_param(self):
        for param in self.body_net.backbone.parameters():
            param.requires_grad = True

    def to_onnx(self, onnx_path: str):
        """

        :param onnx_path:
        :return:
        """
        self.eval()
        data = torch.rand(size=(1, 3, self.param.img_size, self.param.img_size))
        input_names = ["images"]
        output_names = ["head3_predicts", "head4_predicts", "head5_predicts"]
        torch.onnx.export(self, data, onnx_path,
                          export_params=True,
                          opset_version=11,
                          input_names=input_names,
                          output_names=output_names)