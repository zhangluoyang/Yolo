import torch
from typing import *
import torchvision.ops
from luoyang.param.Param import Yolo6Param
from luoyang.model.Layer import MetricLayer
from luoyang.yolov6.YoloV6Body import YoloV6Body, RepVGGBlock, ConvModule
from luoyang.model.Layer import TaskLayer, LossLayer
from luoyang.model.metric.DetectMetric import DetectMetric
from luoyang.yolov6.YoloV6Loss import YoloV6Loss
import luoyang.utils.torch_utils as torch_utils


class YoloV6(TaskLayer):

    def __init__(self, param: Yolo6Param):
        super(YoloV6, self).__init__()
        self.param = param
        self.yolo_body = YoloV6Body(param=param)
        if param.pretrain_path is not None:
            state_dict = torch.load(self.param.pretrain_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)

    def forward(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]):
        """

        :param tensor_dict:
        :return:
        """
        features = self.yolo_body(tensor_dict["batch_images"])
        anchors_xx_yy, anchor_points_xy, num_anchors_list, stride_tensor, cls_score_tensor, \
        predict_xy_xy, reg_distribute_tensor = self.yolo_body.head(features)
        tensor_dict["anchors_xx_yy"] = anchors_xx_yy
        tensor_dict["anchor_points_xy"] = anchor_points_xy
        tensor_dict["num_anchors_list"] = num_anchors_list
        tensor_dict["stride_tensor"] = stride_tensor
        tensor_dict["cls_score_tensor"] = cls_score_tensor
        tensor_dict["predict_xy_xy"] = predict_xy_xy
        tensor_dict["reg_distribute_tensor"] = reg_distribute_tensor
        return tensor_dict

    def _forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """

        :param images:
        :return:
        """
        features = self.yolo_body(images)
        _, _, _, _, cls_score_tensor, predict_xy_xy, _ = self.yolo_body.head(features)
        # 保持与其它模型一致的输入
        predict_cx_cy_w_h = torchvision.ops.box_convert(boxes=predict_xy_xy,
                                                        in_fmt="xyxy",
                                                        out_fmt="cxcywh")
        predict_config = torch.clamp(cls_score_tensor[:, :, 0:1], min=1, max=1)
        batch_predicts = torch.cat([predict_cx_cy_w_h, predict_config, cls_score_tensor], dim=-1)
        return [batch_predicts]

    def predict(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]) -> Dict[str, Any]:
        """

        :param tensor_dict:
        :return:
        """
        tensor_dict = self.forward(tensor_dict)
        batch = tensor_dict["batch_images"].size(0)
        predict_xy_xy = tensor_dict["predict_xy_xy"]
        predict_config = torch.ones((batch,
                                     predict_xy_xy.shape[1], 1),
                                    device=predict_xy_xy.device,
                                    dtype=predict_xy_xy.dtype)
        cls_score_tensor = tensor_dict["cls_score_tensor"]
        # [batch, anchor_num, 4 + 1 + class_num]
        batch_predicts = torch.cat([predict_xy_xy, predict_config, cls_score_tensor], dim=-1)

        batch_nms_predicts = torch_utils.non_max_suppression(prediction=batch_predicts,
                                                             conf_threshold=self.param.conf_threshold,
                                                             nms_threshold=self.param.nms_threshold,
                                                             in_fmt="xyxy",
                                                             img_size=self.param.img_size)

        tensor_dict["predicts"] = [None if nms_predicts is None else nms_predicts.detach().cpu().numpy()
                                   for nms_predicts in batch_nms_predicts]
        return tensor_dict

    def build_loss_layer(self) -> List[LossLayer]:
        loss = YoloV6Loss(target_name="targets_v6",
                          anchors_xy_xy_name="anchors_xx_yy",
                          anchor_points_xy_name="anchor_points_xy",
                          num_anchors_name="num_anchors_list",
                          predict_xy_xy_name="predict_xy_xy",
                          predict_score_name="cls_score_tensor",
                          stride_name="stride_tensor",
                          num_classes=self.param.class_num,
                          img_size=self.param.img_size,
                          use_dfl=self.param.use_dfl,
                          reg_distribute_name="reg_distribute_tensor",
                          loss_name="",
                          reg_max=16,
                          warmup_epoch=1)
        return [loss]

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

        for param in self.parameters():
            param.requires_grad = False

        for param in self.yolo_body.head.parameters():
            param.requires_grad = True

    def update_train_param(self):

        for param in self.parameters():
            param.requires_grad = True

    def fuse(self):
        for m in self.modules():
            if type(m) is ConvModule and hasattr(m, "bn"):
                m.conv = torch_utils.fuse_conv_and_bn(conv=m.conv, bn=m.bn)
                delattr(m, "bn")
                m.forward = m.forward_fuse

        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()

    def to_onnx(self, onnx_path: str):
        """

        :param onnx_path:
        :return:
        """
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
                          opset_version=11,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=dynamic_axes)
