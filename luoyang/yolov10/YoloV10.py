import torch
from typing import *
from luoyang.param.Param import Yolo10Param
from luoyang.yolov10.YoloV10Body import YoloV10Body, Conv
from luoyang.model.Layer import TaskLayer
from luoyang.yolov10.YoloV10Head import YoloV10Head
import luoyang.utils.torch_utils as torch_utils
from luoyang.model.Layer import MetricLayer
from luoyang.model.metric.DetectMetric import DetectMetric
from luoyang.model.Layer import LossLayer
from luoyang.utils.torch_utils import fuse_conv_and_bn

import torchvision.ops

class YoloV10(TaskLayer):

    def __init__(self, param: Yolo10Param):
        super(YoloV10, self).__init__()
        self.param = param
        feature_size = [param.img_size // stride for stride in param.strides]

        self.body_net = YoloV10Body(num_classes=param.class_num,
                                    phi=param.m_type,
                                    pretrained_path=param.pretrain_path)
        self.head = YoloV10Head(reg_max=16,
                                stride=param.strides,
                                feature_size=feature_size,
                                num_classes=param.class_num,
                                use_dfl=True)

    def _forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        :param images:
        :return:
        """
        x = self.body_net(images)
        predict_scores, predict_scaled_bounding_boxes, stride_tensor, anchor_points, predict_distribute = self.head(x)

        predict_bounding_boxes = predict_scaled_bounding_boxes * stride_tensor

        predict_cx_cy_w_h = torchvision.ops.box_convert(boxes=predict_bounding_boxes,
                                                        in_fmt="xyxy",
                                                        out_fmt="cxcywh")

        predict_config, _ = torch.max(predict_scores, dim=-1, keepdim=True)
        batch_predicts = torch.cat([predict_cx_cy_w_h, predict_config, predict_scores], dim=-1)
        batch_predicts = batch_predicts.reshape(shape=(-1, int(batch_predicts.size(1)), int(batch_predicts.size(2))))
        return [batch_predicts]

    def _feed_forward(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]):
        """

        :param tensor_dict:
        :return:
        """
        x = self.body_net(tensor_dict["batch_images"])
        predict_scores, predict_scaled_bounding_boxes, stride_tensor, anchor_points, predict_distribute = self.head(x)
        tensor_dict["predict_scores"] = predict_scores
        tensor_dict["predict_scaled_bounding_boxes"] = predict_scaled_bounding_boxes
        tensor_dict["stride_tensor"] = stride_tensor
        tensor_dict["anchor_points"] = anchor_points
        tensor_dict["predict_distribute"] = predict_distribute

        return tensor_dict

    def forward(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]):
        """

        :param tensor_dict:
        :return:
        """
        self._feed_forward(tensor_dict=tensor_dict)
        stride_tensor = tensor_dict["stride_tensor"].detach()
        # 坐标、锚点 均转换到输入尺寸
        predict_scores = tensor_dict["predict_scores"].detach()
        predict_bounding_boxes = tensor_dict["predict_scaled_bounding_boxes"].detach() * stride_tensor
        anchor_points = tensor_dict["anchor_points"].detach() * stride_tensor
        # ground_true
        # 正负样本 (bs, max_box_num_in_batch, 5)
        targets = tensor_dict["targets_v8"]
        gt_labels, gt_bbox_es = targets.split((1, 4), 2)
        # 用于表示 哪些是padding的样本
        mask_gt = gt_bbox_es.sum(2, keepdim=True).gt_(0)

        target_labels, target_bbox_es, target_scores, fg_mask, target_gt_idx = \
            self.task_aligned_assigner(pd_scores=predict_scores,
                                       pd_bbox_es=predict_bounding_boxes,
                                       anc_points=anchor_points,
                                       gt_labels=gt_labels,
                                       gt_bbox_es=gt_bbox_es,
                                       mask_gt=mask_gt)
        tensor_dict["target_labels"] = target_labels
        tensor_dict["target_bbox_es"] = target_bbox_es
        tensor_dict["target_scores"] = target_scores
        tensor_dict["fg_mask"] = fg_mask
        tensor_dict["target_gt_idx"] = target_gt_idx
        return tensor_dict

    def predict(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]) -> Dict[str, Any]:
        """

        :param tensor_dict:
        :return:
        """
        if "batch_targets" in tensor_dict:
            self.forward(tensor_dict=tensor_dict)
        else:
            self._feed_forward(tensor_dict=tensor_dict)
        threshold = tensor_dict["threshold"]
        predict_scaled_bounding_boxes = tensor_dict["predict_scaled_bounding_boxes"]
        predict_scores = tensor_dict["predict_scores"]
        stride_tensor = tensor_dict["stride_tensor"]
        predict_bounding_boxes = predict_scaled_bounding_boxes * stride_tensor
        predict_configs, class_ids = torch.max(predict_scores, dim=-1, keepdim=True)

        tensor_predicts = torch.cat([predict_bounding_boxes, predict_configs, predict_configs, class_ids], dim=-1)
        predicts = []
        for tensor_predict in tensor_predicts:
            np_predict = tensor_predict[tensor_predict[:, -2] >= threshold].detach().cpu().numpy()
            if len(np_predict) > 0:
                predicts.append(np_predict)
            else:
                predicts.append(None)
        tensor_dict["predicts"] = predicts
        return tensor_dict

    def build_loss_layer(self) -> List[LossLayer]:
        raise NotImplemented

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
        raise NotImplemented

    def update_train_param(self):
        raise NotImplemented

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuse_forward
        return self


    def to_onnx(self, onnx_path: str):
        """

        :param onnx_path:
        :return:
        """
        # 重惨化
        self.eval()
        self.to("cpu")
        self.forward = self._forward
        data = torch.rand(size=(1, 3, self.param.img_size, self.param.img_size)).to("cpu")
        input_names = ["images"]
        output_names = ["head_predicts"]
        dynamic_axes = {"images": {0: "batch"},
                        "head_predicts": {0: "batch"}}
        torch.onnx.export(self, data, onnx_path,
                          export_params=True,
                          opset_version=12,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=dynamic_axes)
