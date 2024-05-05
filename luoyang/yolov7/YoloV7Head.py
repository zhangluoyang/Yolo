import torch
import numpy as np
import torch.nn as nn
from typing import List, Tuple


class YoloV7Head(nn.Module):
    """
    预测处于当前特征层的坐标 (stride表示下采样比例)
    """

    def __init__(self, stride: int,
                 yolo_output_size: int,
                 input_size: Tuple[int, int],
                 anchors: List[Tuple[int, int]]):
        super(YoloV7Head, self).__init__()
        self.stride = stride
        self.yolo_output_size = yolo_output_size
        self.anchors = torch.tensor(np.array(anchors))
        self.input_size = input_size
        self.a_num = len(self.anchors)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """

        :param input_tensor:
        :return:
        """
        bs = input_tensor.size(0)
        in_h, in_w = self.input_size

        input_tensor = input_tensor.view(bs,
                                         self.anchors.size(0),
                                         self.yolo_output_size, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        input_tensor = torch.sigmoid(input_tensor)
        x = input_tensor[..., 0]
        y = input_tensor[..., 1]
        w = input_tensor[..., 2]
        h = input_tensor[..., 3]

        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            bs * self.a_num, 1, 1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            bs * self.a_num, 1, 1).view(y.shape).type_as(x)
        scaled_anchors_l = self.anchors / self.stride

        anchor_w = scaled_anchors_l.index_select(1, torch.LongTensor([0])).type_as(input_tensor)
        anchor_h = scaled_anchors_l.index_select(1, torch.LongTensor([1])).type_as(input_tensor)

        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        # x y 偏移量
        boxes_x = torch.unsqueeze(2 * x - 0.5 + grid_x, -1)
        boxes_y = torch.unsqueeze(2 * y - 0.5 + grid_y, -1)
        # w h
        boxes_w = torch.unsqueeze((2 * w) ** 2 * anchor_w, -1)
        boxes_h = torch.unsqueeze((2 * h) ** 2 * anchor_h, -1)

        return torch.cat([boxes_x, boxes_y, boxes_w, boxes_h, input_tensor[..., 4:]], dim=-1)
