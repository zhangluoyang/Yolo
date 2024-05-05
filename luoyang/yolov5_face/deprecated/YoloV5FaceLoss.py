import torch
from typing import Dict, Union, List
from luoyang.model.Layer import LossLayer
from luoyang.model.loss.GIouLoss import GIouLoss
from luoyang.model.loss.WingLoss import WingLoss
from luoyang.model.loss.BceSmoothLoss import BceSmoothLoss


class YoloV5FaceLoss(LossLayer):
    def __init__(self,
                 loss_name: str,
                 point_num: int,
                 head_predict_name: str,
                 ground_true_name: str,
                 point_mask_name: str,
                 weight: float):
        super(YoloV5FaceLoss, self).__init__()
        self.loss_name = loss_name
        self.head_predict_name = head_predict_name
        self.ground_true_name = ground_true_name
        self.point_mask_name = point_mask_name
        self.weight = weight
        self.point_num = point_num

        self.box_loss = GIouLoss(return_g_iou=True)
        self.conf_loss = BceSmoothLoss()
        self.point_loss = WingLoss()
        self.cls_loss = BceSmoothLoss(reduce="mean")

        self._lambda_box = 0.05
        self._lambda_conf = 0.4225
        self._lambda_cls = 0.125
        self._point_conf = 0.05

    def loss_feed_forward(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]) -> \
            Dict[str, torch.Tensor]:
        """

        :param tensor_dict:
        :return:
        """
        # [b, a_num,  h, w, (5 + num_classes)]
        head_predict = tensor_dict[self.head_predict_name]
        # [b, a_num,  h, w, (5 + num_classes + 2 * point_num)]
        ground_true = tensor_dict[self.ground_true_name]
        device = head_predict.device
        obj_mask = ground_true[..., 4] == 1
        obj_mask_num = obj_mask.float().sum()
        t_obj = torch.zeros_like(head_predict[..., 0], device=device)

        conf = head_predict[..., 4]
        if obj_mask_num != 0:
            class_loss = self.cls_loss(head_predict[..., 5 + 2 * self.point_num:][obj_mask],
                                       ground_true[..., 5 + 2 * self.point_num:][obj_mask]) * self._lambda_cls

            _box_loss, g_iou = self.box_loss(head_predict[..., :4][obj_mask],
                                             ground_true[..., :4][obj_mask])
            box_loss = torch.mean(_box_loss) * self._lambda_box
            t_obj[obj_mask] = g_iou
            conf_loss = torch.mean(self.conf_loss(conf, t_obj)) * self._lambda_conf * self.weight
            predict_point = head_predict[..., 5: 5 + 2 * self.point_num]
            target_point = ground_true[..., 5: 5 + 2 * self.point_num]
            point_mask = tensor_dict[self.point_mask_name]
            point_mask_bool = point_mask.bool()
            point_loss = self.point_loss(predict_point[obj_mask & point_mask_bool],
                                         target_point[obj_mask & point_mask_bool]) \
                         * self._point_conf
            loss = class_loss + box_loss + conf_loss + point_loss
        else:
            conf_loss = torch.mean(self.conf_loss(conf, t_obj)) * self._lambda_conf * self.weight
            loss = conf_loss
        return {"{0}_loss".format(self.loss_name): loss}
