import torch
from typing import List, Dict, Union, Any
from src.param.Param import Yolo3Param
from src.model.Layer import MetricLayer
from src.model.yolo.YoloV3Body import YoloV3Body
from src.model.yolo.YoloLoss import YoloLoss
from src.model.yolo.YoloHead import YoloHead
from src.model.yolo.YoloTarget import YoloTarget
from src.model.yolo.YoloIgnore import YoloIgnore
from src.model.Layer import TaskLayer, LossLayer
import src.utils.torch_utils as torch_utils
from src.model.metric.DetectMetric import DetectMetric


class YoloV3(TaskLayer):

    def __init__(self, param: Yolo3Param):
        super(YoloV3, self).__init__()
        self.param = param

        self.body_output_size = 5 + self.param.class_num
        self.body_net = YoloV3Body(output_size=self.body_output_size * 3,
                                   darknet_pretrain_path=param.darknet53_weight_path)

        self.yolo_head3 = YoloHead(stride=self.param.strides[0],
                                   input_size=(self.param.img_size // 8, self.param.img_size // 8),
                                   yolo_output_size=self.body_output_size,
                                   anchors=param.anchors[: 3])

        self.yolo_head4 = YoloHead(stride=self.param.strides[1],
                                   yolo_output_size=self.body_output_size,
                                   input_size=(self.param.img_size // 16, self.param.img_size // 16),
                                   anchors=param.anchors[3: 6])

        self.yolo_head5 = YoloHead(stride=self.param.strides[2],
                                   input_size=(self.param.img_size // 32, self.param.img_size // 32),
                                   yolo_output_size=self.body_output_size,
                                   anchors=param.anchors[6:])

        self.yolo_target3 = YoloTarget(anchors=param.anchors,
                                       stride=self.param.strides[0],
                                       body_output_size=self.body_output_size,
                                       anchor_mask=[0, 1, 2],
                                       input_size=(self.param.img_size // 8, self.param.img_size // 8))

        self.yolo_target4 = YoloTarget(anchors=param.anchors,
                                       stride=self.param.strides[1],
                                       body_output_size=self.body_output_size,
                                       anchor_mask=[3, 4, 5],
                                       input_size=(self.param.img_size // 16, self.param.img_size // 16))

        self.yolo_target5 = YoloTarget(anchors=param.anchors,
                                       stride=self.param.strides[2],
                                       body_output_size=self.body_output_size,
                                       anchor_mask=[6, 7, 8],
                                       input_size=(self.param.img_size // 32, self.param.img_size // 32))

        self.yolo_ignore3 = YoloIgnore(input_size=(self.param.img_size // 8, self.param.img_size // 8),
                                       ignore_threshold=0.5)

        self.yolo_ignore4 = YoloIgnore(input_size=(self.param.img_size // 16, self.param.img_size // 16),
                                       ignore_threshold=0.5)

        self.yolo_ignore5 = YoloIgnore(input_size=(self.param.img_size // 32, self.param.img_size // 32),
                                       ignore_threshold=0.5)

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """

        :param images:
        :return:
        """
        out5, out4, out3 = self.body_net(images)
        head3_predicts = self.yolo_head3(out3)
        head4_predicts = self.yolo_head4(out4)
        head5_predicts = self.yolo_head5(out5)

        return [head3_predicts, head4_predicts, head5_predicts]

    def _feed_forward(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]):
        """

        :param tensor_dict:
        :return:
        """
        out5, out4, out3 = self.body_net(tensor_dict["batch_images"])

        head3_predicts = self.yolo_head3(out3)

        head4_predicts = self.yolo_head4(out4)

        head5_predicts = self.yolo_head5(out5)

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
        head3_predicts = tensor_dict["head3_predicts"]
        head4_predicts = tensor_dict["head4_predicts"]
        head5_predicts = tensor_dict["head5_predicts"]
        batch_targets = tensor_dict["batch_targets"]
        device = head3_predicts.device

        head_3_ground_true, head_3_no_obj_mask = self.yolo_target3(batch_targets)
        head_4_ground_true, head_4_no_obj_mask = self.yolo_target4(batch_targets)
        head_5_ground_true, head_5_no_obj_mask = self.yolo_target5(batch_targets)

        head_3_no_obj_mask = self.yolo_ignore3(head3_predicts[..., :4], batch_targets, head_3_no_obj_mask)

        head_4_no_obj_mask = self.yolo_ignore4(head4_predicts[..., :4], batch_targets, head_4_no_obj_mask)

        head_5_no_obj_mask = self.yolo_ignore5(head5_predicts[..., :4], batch_targets, head_5_no_obj_mask)

        tensor_dict["head_3_ground_true"] = head_3_ground_true.to(device)
        tensor_dict["head_3_no_obj_mask"] = head_3_no_obj_mask.bool().to(device)

        tensor_dict["head_4_ground_true"] = head_4_ground_true.to(device)
        tensor_dict["head_4_no_obj_mask"] = head_4_no_obj_mask.bool().to(device)

        tensor_dict["head_5_ground_true"] = head_5_ground_true.to(device)
        tensor_dict["head_5_no_obj_mask"] = head_5_no_obj_mask.bool().to(device)
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

        head_3_loss = YoloLoss(loss_name="3",
                               head_predict_name="head3_predicts",
                               ground_true_name="head_3_ground_true",
                               no_obj_mask_name="head_3_no_obj_mask",
                               box_loss_type="g_iou_loss",
                               weight=4.0)
        head_4_loss = YoloLoss(loss_name="4",
                               head_predict_name="head4_predicts",
                               ground_true_name="head_4_ground_true",
                               no_obj_mask_name="head_4_no_obj_mask",
                               box_loss_type="g_iou_loss",
                               weight=1.0)
        head_5_loss = YoloLoss(loss_name="5",
                               head_predict_name="head5_predicts",
                               ground_true_name="head_5_ground_true",
                               no_obj_mask_name="head_5_no_obj_mask",
                               box_loss_type="g_iou_loss",
                               weight=0.4)

        return [head_3_loss, head_4_loss, head_5_loss]

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
        for param in self.body_net.darknet53.parameters():
            param.requires_grad = False

    def update_train_param(self):
        for param in self.body_net.darknet53.parameters():
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