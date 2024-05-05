import torch
from typing import Dict, Union, List
from luoyang.model.Layer import LossLayer
from luoyang.model.loss.GIouLoss import GIouLoss
from luoyang.model.loss.BceSmoothLoss import BceSmoothLoss


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

    def loss_feed_forward(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]) -> \
            Dict[str, torch.Tensor]:
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
