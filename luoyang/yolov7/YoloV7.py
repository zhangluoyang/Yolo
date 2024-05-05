import torch
from typing import List, Dict, Union, Any
from luoyang.param.Param import Yolo7Param
from luoyang.model.Layer import MetricLayer
import luoyang.utils.torch_utils as torch_utils
from luoyang.yolov7.YoloV7Loss import YoloV7Loss
from luoyang.yolov7.YoloV7Head import YoloV7Head
from luoyang.model.Layer import TaskLayer, LossLayer
from luoyang.yolov7.YoloV7Body import YoloV7Body, Conv, RepConv, _fuse_conv_and_bn
from luoyang.model.metric.DetectMetric import DetectMetric
from luoyang.yolov7.YoloTarget import YoloCandidateTarget, YoloSimpleOTATarget


class YoloV7(TaskLayer):

    def __init__(self, param: Yolo7Param,
                 anchor_num: int = 3):
        super(YoloV7, self).__init__()
        self.param = param

        self.body_output_size = 5 + self.param.class_num
        self.body_net = YoloV7Body(anchor_num=anchor_num,
                                   num_classes=param.class_num,
                                   phi=param.m_type,
                                   pretrained_path=self.param.pretrain_path)

        self.yolo_head3 = YoloV7Head(stride=self.param.strides[0],
                                     input_size=(self.param.img_size // 8, self.param.img_size // 8),
                                     yolo_output_size=self.body_output_size,
                                     anchors=param.anchors[: 3])

        self.yolo_head4 = YoloV7Head(stride=self.param.strides[1],
                                     yolo_output_size=self.body_output_size,
                                     input_size=(self.param.img_size // 16, self.param.img_size // 16),
                                     anchors=param.anchors[3: 6])

        self.yolo_head5 = YoloV7Head(stride=self.param.strides[2],
                                     input_size=(self.param.img_size // 32, self.param.img_size // 32),
                                     yolo_output_size=self.body_output_size,
                                     anchors=param.anchors[6:])
        # 生成候选框
        self.candidate_target = YoloCandidateTarget(anchors=[self.param.anchors[:3],
                                                             self.param.anchors[3:6],
                                                             self.param.anchors[6:]],
                                                    strides=self.param.strides,
                                                    anchor_num=anchor_num)
        #  动态 选择
        self.dynamic_target = YoloSimpleOTATarget(image_size=(self.param.img_size, self.param.img_size),
                                                  num_classes=self.param.class_num,
                                                  strides=self.param.strides)

    def _forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """

        :param images:
        :return:
        """
        out5, out4, out3 = self.body_net(images)

        head5_predicts = self.yolo_head5(out5)
        head4_predicts = self.yolo_head4(out4)
        head3_predicts = self.yolo_head3(out3)
        batch_predict_list = []
        for predict, stride, feature_size in zip([head3_predicts, head4_predicts, head5_predicts],
                                                 self.param.strides,
                                                 self.param.feature_size):
            predict[..., :4] = predict[..., :4] * stride
            batch_predict_list.append(predict.view(-1, feature_size * feature_size * 3, self.body_output_size))
        batch_predicts = torch.cat(batch_predict_list, dim=1)
        batch_predicts[..., 0] = torch.clamp(batch_predicts[..., 0], min=0, max=self.param.img_size - 1)
        batch_predicts[..., 1] = torch.clamp(batch_predicts[..., 1], min=0, max=self.param.img_size - 1)
        batch_predicts[..., 2] = torch.clamp(batch_predicts[..., 2], min=0, max=self.param.img_size - 1)
        batch_predicts[..., 3] = torch.clamp(batch_predicts[..., 3], min=0, max=self.param.img_size - 1)
        return [batch_predicts]

    def _feed_forward(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]):
        """

        :param tensor_dict:
        :return:
        """
        out5, out4, out3 = self.body_net(tensor_dict["batch_images"])
        # yolo head 预测
        head5_predicts = self.yolo_head5(out5)
        head4_predicts = self.yolo_head4(out4)
        head3_predicts = self.yolo_head3(out3)
        tensor_dict["head3_predicts"] = head3_predicts
        tensor_dict["head4_predicts"] = head4_predicts
        tensor_dict["head5_predicts"] = head5_predicts
        return tensor_dict

    def forward(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]):
        """

        :param tensor_dict:
        :return:
        """
        tensor_dict = self._feed_forward(tensor_dict=tensor_dict)
        # 8 16 32 倍 下采样的预测结果
        yolo_head_predicts = [tensor_dict["head3_predicts"],
                              tensor_dict["head4_predicts"],
                              tensor_dict["head5_predicts"]]
        targets_v7 = tensor_dict["targets_v7"]
        # 与 yolo5 一样 生成候选正样本
        indices, anchors = self.candidate_target(yolo_head_predicts,
                                                 targets_v7)
        # 动态选择正样本
        matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchors = \
            self.dynamic_target(yolo_head_predicts,
                                targets_v7,
                                indices,
                                anchors)

        for _id, i in enumerate(range(3, 6)):
            tensor_dict["matching_bs_{0}".format(i)] = matching_bs[_id]
            tensor_dict["matching_as_{0}".format(i)] = matching_as[_id]
            tensor_dict["matching_gjs_{0}".format(i)] = matching_gjs[_id]
            tensor_dict["matching_gis_{0}".format(i)] = matching_gis[_id]
            tensor_dict["matching_targets_{0}".format(i)] = matching_targets[_id]

        return tensor_dict

    def predict(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]) -> Dict[str, Any]:
        """

        :param tensor_dict:
        :return:
        """
        batch_predict_list = []
        if "batch_targets" in tensor_dict:
            tensor_dict = self.forward(tensor_dict=tensor_dict)
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
        head_3_loss = YoloV7Loss(loss_name="3",
                                 head_predict_name="head3_predicts",
                                 matching_b_name="matching_bs_3",
                                 matching_a_name="matching_as_3",
                                 matching_gj_name="matching_gjs_3",
                                 matching_gi_name="matching_gis_3",
                                 matching_target_name="matching_targets_3",
                                 weight=4.0)
        head_4_loss = YoloV7Loss(loss_name="4",
                                 head_predict_name="head4_predicts",
                                 matching_b_name="matching_bs_4",
                                 matching_a_name="matching_as_4",
                                 matching_gj_name="matching_gjs_4",
                                 matching_gi_name="matching_gis_4",
                                 matching_target_name="matching_targets_4",
                                 weight=1.0)
        head_5_loss = YoloV7Loss(loss_name="5",
                                 head_predict_name="head5_predicts",
                                 matching_b_name="matching_bs_5",
                                 matching_a_name="matching_as_5",
                                 matching_gj_name="matching_gjs_5",
                                 matching_gi_name="matching_gis_5",
                                 matching_target_name="matching_targets_5",
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

    def fuse(self):
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse_rep_vgg_block()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = _fuse_conv_and_bn(m.conv, m.bn)
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
                          opset_version=11,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=dynamic_axes)
