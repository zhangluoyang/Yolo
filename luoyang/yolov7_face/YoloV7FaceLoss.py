import torch
from typing import Dict, Union, List
from luoyang.model.loss.WingLoss import WingLoss
from luoyang.model.Layer import LossLayer
from luoyang.model.loss.CIouLoss import CIouLoss
from luoyang.model.loss.BceSmoothLoss import BceSmoothLoss


class YoloV7FaceLoss(LossLayer):

    def __init__(self,
                 point_num: int,
                 loss_name: str,
                 matching_b_name: str,
                 matching_a_name: str,
                 matching_gj_name: str,
                 matching_gi_name: str,
                 matching_target_name: str,
                 head_predict_name: str,
                 weight: float):
        """
        :param point_num
        :param loss_name:
        :param matching_b_name:  image_id
        :param matching_a_name:  anchor_id
        :param matching_gj_name: y_id
        :param matching_gi_name: x_id
        :param matching_target_name: ground_true
        :param head_predict_name yolo head predict
        :param weight
        """
        super(YoloV7FaceLoss, self).__init__()
        self.point_num = point_num
        self.loss_name = loss_name
        self.matching_b_name = matching_b_name
        self.matching_a_name = matching_a_name
        self.matching_gj_name = matching_gj_name
        self.matching_gi_name = matching_gi_name
        self.matching_target_name = matching_target_name
        # self.matching_anchor_name = matching_anchor_name
        self.head_predict_name = head_predict_name
        self.weight = weight

        self.box_loss = CIouLoss(return_c_iou=True)
        self.conf_loss = BceSmoothLoss(reduce="mean")
        self.cls_loss = BceSmoothLoss(reduce="mean")
        self.point_loss = WingLoss()

        self._lambda_box = 0.05
        self._lambda_conf = 0.4225
        self._lambda_cls = 0.125
        self._point_conf = 0.05

    def loss_feed_forward(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]):
        """

        :param tensor_dict:
        :return:
        """
        matching_b = tensor_dict[self.matching_b_name]
        matching_a = tensor_dict[self.matching_a_name]
        matching_gj = tensor_dict[self.matching_gj_name]
        matching_gi = tensor_dict[self.matching_gi_name]
        matching_target = tensor_dict[self.matching_target_name]

        # [b, a_num,  h, w, (5 + num_classes)]
        head_predict = tensor_dict[self.head_predict_name]
        device = head_predict.device
        # 当前特征层预测的尺寸
        feature_map_size = torch.tensor(head_predict.shape, device=device)[[3, 2, 3, 2]].type_as(head_predict)

        cls_loss = torch.zeros(1, device=device)
        box_loss = torch.zeros(1, device=device)
        obj_loss = torch.zeros(1, device=device)
        point_loss = torch.zeros(1, device=device)

        t_obj = torch.zeros_like(head_predict[..., 0], device=device)

        batch_size = matching_b.shape[0]
        if batch_size != 0:
            # 根据索引 找到对应的预测值
            # 相对于特征层的坐标
            matching_prediction = head_predict[matching_b, matching_a, matching_gj, matching_gi]
            # ground_true 相对于特征层的坐标
            selected_tbox = matching_target[..., 2:6] * feature_map_size
            # ground_true point 相对于特征层的坐标
            selected_t_point: torch.Tensor = matching_target[..., 6:]
            selected_t_point[..., 0::2] = selected_t_point[..., 0::2] * feature_map_size[3]
            selected_t_point[..., 1::2] = selected_t_point[..., 1::2] * feature_map_size[2]

            predict_point = matching_prediction[..., 5: 5 + 2 * self.point_num]

            point_mask = selected_t_point > 0
            point_loss = self.point_loss(predict_point[point_mask], selected_t_point[point_mask])

            box_loss, c_iou = self.box_loss(matching_prediction[..., :4],
                                            selected_tbox)
            box_loss = torch.mean(box_loss)
            t_obj[matching_b, matching_a, matching_gj, matching_gi] = c_iou.detach().clamp(0).type(t_obj.dtype)
            # ground_true 的类别
            matching_cls = matching_target[..., 1].long()
            one_hot = torch.full_like(matching_prediction[..., 5 + 2 * self.point_num:], fill_value=0, device=device)
            one_hot[range(batch_size), matching_cls] = 1

            cls_loss = self.cls_loss(matching_prediction[..., 5 + 2 * self.point_num:],
                                     one_hot) * self._lambda_cls

        obj_loss += self.conf_loss(head_predict[..., 4], t_obj) * self.weight

        loss = cls_loss * self._lambda_cls + box_loss * self._lambda_box + obj_loss * self._lambda_conf
        loss = loss + point_loss * self._point_conf

        return {"{0}_loss".format(self.loss_name): loss}
