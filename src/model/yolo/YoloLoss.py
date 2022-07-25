from typing import Dict, Union, List

import torch

from src.model.Layer import LossLayer
from src.model.loss.GIouLoss import GIouLoss
from src.model.loss.BceSmoothLoss import BceSmoothLoss
from src.model.loss.CIouLoss import CIouLoss
from src.model.loss.DIouLoss import DIouLoss


class YoloLoss(LossLayer):

    def __init__(self,
                 loss_name: str,
                 head_predict_name: str,
                 ground_true_name: str,
                 no_obj_mask_name: str,
                 box_loss_type: str,
                 weight: float):

        super(YoloLoss, self).__init__()
        self.loss_name = loss_name
        self.head_predict_name = head_predict_name
        self.ground_true_name = ground_true_name
        self.no_obj_mask_name = no_obj_mask_name
        self.box_loss_type = box_loss_type
        self.weight = weight
        if box_loss_type == "g_iou_loss":
            self.box_loss = GIouLoss()
        elif box_loss_type == "d_iou_loss":
            self.box_loss = DIouLoss()
        elif box_loss_type == "c_iou_loss":
            self.box_loss = CIouLoss()
        self.conf_loss = BceSmoothLoss()
        self.cls_loss = BceSmoothLoss(reduce="mean")

        self._lambda_box = 0.05
        self._lambda_conf = 5
        self._lambda_cls = 0.25

    def loss_feed_forward(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]) -> \
            Dict[str, torch.Tensor]:
        head_predict = tensor_dict[self.head_predict_name]
        ground_true = tensor_dict[self.ground_true_name]
        no_obj_mask = tensor_dict[self.no_obj_mask_name]

        obj_mask = ground_true[..., 4] == 1
        obj_mask_num = obj_mask.float().sum()
        conf_loss = torch.mean(self.conf_loss(head_predict[..., 4][obj_mask | no_obj_mask.bool()],
                                              ground_true[..., 4][obj_mask | no_obj_mask.bool()])) \
                    * self._lambda_conf * self.weight

        if obj_mask_num != 0:
            class_loss = self.cls_loss(head_predict[..., 5:][obj_mask],
                                       ground_true[..., 5:][obj_mask]) * self._lambda_cls
            box_loss = self.box_loss(head_predict[..., :4][obj_mask],
                                     ground_true[..., :4][obj_mask]) * self._lambda_box
            loss = conf_loss + class_loss + box_loss
        else:
            loss = conf_loss
        return {"{0}_loss".format(self.loss_name): loss}


class YoloV5Loss(LossLayer):
    def __init__(self,
                 loss_name: str,
                 head_predict_name: str,
                 ground_true_name: str,
                 weight: float):
        super(YoloV5Loss, self).__init__()
        self.loss_name = loss_name
        self.head_predict_name = head_predict_name
        self.ground_true_name = ground_true_name
        self.weight = weight

        self.box_loss = GIouLoss(return_g_iou=True)
        self.conf_loss = BceSmoothLoss()
        self.cls_loss = BceSmoothLoss(reduce="mean")

        self._lambda_box = 0.05
        self._lambda_conf = 0.4225
        self._lambda_cls = 0.125

    def loss_feed_forward(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]) -> Dict[
        str, torch.Tensor]:
        """

        :param tensor_dict:
        :return:
        """
        # [b, a_num,  h, w, (5 + num_classes)]
        head_predict = tensor_dict[self.head_predict_name]
        # [b, a_num,  h, w, (5 + num_classes)]
        ground_true = tensor_dict[self.ground_true_name]

        obj_mask = ground_true[..., 4] == 1
        obj_mask_num = obj_mask.float().sum()

        conf = head_predict[..., 4]
        if obj_mask_num != 0:
            class_loss = self.cls_loss(head_predict[..., 5:][obj_mask],
                                       ground_true[..., 5:][obj_mask]) * self._lambda_cls

            _box_loss, g_iou = self.box_loss(head_predict[..., :4],
                                             ground_true[..., :4])
            box_loss = torch.mean(_box_loss[obj_mask]) * self._lambda_box

            t_obj = torch.where(obj_mask, g_iou.detach().clamp(0), torch.zeros_like(ground_true[..., 4]))
            conf_loss = torch.mean(self.conf_loss(conf, t_obj)) * self._lambda_conf * self.weight
            loss = class_loss + box_loss + conf_loss
        else:
            t_obj = torch.zeros_like(ground_true[..., 4])
            conf_loss = torch.mean(self.conf_loss(conf, t_obj)) * self._lambda_conf * self.weight
            loss = conf_loss
        return {"{0}_loss".format(self.loss_name): loss}
