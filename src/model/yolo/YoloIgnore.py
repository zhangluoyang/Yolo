import torch
import torchvision
import torch.nn as nn
from typing import Tuple, List, Union


class YoloIgnore(nn.Module):

    def __init__(self,
                 input_size: Tuple[int, int],
                 ignore_threshold: float = 0.5):
        super(YoloIgnore, self).__init__()
        self.input_size = input_size
        self.ignore_threshold = ignore_threshold

    def forward(self, batch_predict_boxes: torch.Tensor,
                batch_targets: List[Union[torch.Tensor, None]],
                no_obj_mask: torch.Tensor) -> torch.Tensor:
        """

        :param batch_predict_boxes:  [batch_size, a_num, in_h, in_w, 4]
        :param batch_targets
        :param no_obj_mask
        :return:
        """
        batch_size = len(batch_targets)
        for b in range(batch_size):
            if batch_targets[b] is not None and len(batch_targets[b]) > 0:
                target_boxes = torch.zeros_like(batch_targets[b])

                target_boxes[:, [0, 2]] = batch_targets[b][:, [0, 2]] * self.input_size[1]
                target_boxes[:, [1, 3]] = batch_targets[b][:, [1, 3]] * self.input_size[0]
                target_boxes = target_boxes[..., :4]

                target_boxes = torchvision.ops.box_convert(boxes=target_boxes,
                                                           in_fmt="cxcywh",
                                                           out_fmt="xyxy").to(batch_predict_boxes.device)
                predict_boxes = torchvision.ops.box_convert(boxes=batch_predict_boxes[b].view(-1, 4),
                                                            in_fmt="cxcywh",
                                                            out_fmt="xyxy")

                iou = torchvision.ops.box_iou(boxes1=target_boxes, boxes2=predict_boxes)
                if len(iou) > 0:
                    max_iou, _ = torch.max(iou, dim=0)
                    max_iou = max_iou.view(batch_predict_boxes[b].size()[:3])
                    no_obj_mask[b][max_iou > self.ignore_threshold] = 0
                else:
                    print("此处出现问题 暂时不清楚怎么回事哦 我只能把输入的数据记录下来哦!!!!!!!!!!!!!!!!!")
                    print("target_boxes:{0}, predict_boxes:{1}".format(target_boxes.shape, predict_boxes.shape))
        return no_obj_mask

