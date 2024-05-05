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
                 match_name: str,
                 weight: float):
        super(YoloV5FaceLoss, self).__init__()
        self.loss_name = loss_name
        self.head_predict_name = head_predict_name
        self.ground_true_name = ground_true_name
        self.match_name = match_name
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
        device = head_predict.device
        conf = head_predict[..., 4]
        t_obj = torch.zeros_like(head_predict[..., 0], device=device)
        # shape(match_target) = (3, g_num, 1 + 1 + 4 + 2 * p_num)
        # # [img_id, class_id, x, y, w, h, point ......]
        match_b, match_a, match_y, match_x, match_points, match_bbox, match_c = tensor_dict[self.match_name]

        if match_b is None or match_b.shape[0] == 0:
            conf_loss = torch.mean(self.conf_loss(conf, t_obj)) * self._lambda_conf * self.weight
            loss = conf_loss
        else:
            batch_size = match_b.shape[0]
            # [g_num * 3, 4 + 1 + 2 * point_num + 1]
            matching_prediction = head_predict[match_b, match_a, match_y, match_x]
            matching_cls = match_c.long()
            one_hot = torch.full_like(matching_prediction[..., 5 + 2 * self.point_num:], fill_value=0, device=device)
            one_hot[range(batch_size), matching_cls] = 1

            class_loss = self.cls_loss(matching_prediction[..., 5 + 2 * self.point_num:],
                                       one_hot) * self._lambda_cls

            _box_loss, g_iou = self.box_loss(matching_prediction[..., :4],
                                             match_bbox)

            box_loss = torch.mean(_box_loss) * self._lambda_box

            t_obj[match_b, match_a, match_y, match_x] = g_iou.detach().clamp(0).type(t_obj.dtype)
            conf_loss = torch.mean(self.conf_loss(conf, t_obj)) * self._lambda_conf * self.weight

            predict_point = matching_prediction[..., 5: 5 + 2 * self.point_num]
            # 仅仅计算正常的点
            point_mask = match_points[..., 6] > 0
            point_loss = self.point_loss(predict_point[point_mask], match_points[point_mask]) * self._point_conf
            loss = class_loss + box_loss + conf_loss + point_loss

        return {"{0}_loss".format(self.loss_name): loss}
