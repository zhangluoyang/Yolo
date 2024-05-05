import torch
import numpy as np
import torch.nn as nn
from typing import List, Tuple


class YoloV7FaceHead(nn.Module):
    """
    预测处于当前特征层的坐标 (stride表示下采样比例)
    """

    def __init__(self, stride: int,
                 point_num: int,
                 yolo_output_size: int,
                 input_size: Tuple[int, int],
                 anchors: List[Tuple[int, int]]):
        super(YoloV7FaceHead, self).__init__()
        self.stride = stride
        self.point_num = point_num
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

        input_tensor = input_tensor.view(bs, self.anchors.size(0), self.yolo_output_size, in_h, in_w) \
            .permute(0, 1, 3, 4, 2).contiguous()

        # box
        input_tensor[..., : 5] = torch.sigmoid(input_tensor[..., :5])
        # class
        input_tensor[..., 5 + 2 * self.point_num:] = \
            torch.sigmoid(input_tensor[..., 5 + 2 * self.point_num:])
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

        boxes_x = torch.unsqueeze(2 * x - 0.5 + grid_x, -1)
        boxes_y = torch.unsqueeze(2 * y - 0.5 + grid_y, -1)
        boxes_w = torch.unsqueeze((2 * w) ** 2 * anchor_w, -1)
        boxes_h = torch.unsqueeze((2 * h) ** 2 * anchor_h, -1)

        _anchor_w = torch.unsqueeze(anchor_w, dim=-1)
        _anchor_h = torch.unsqueeze(anchor_h, dim=-1)
        _grid_x = torch.unsqueeze(grid_x, dim=-1)
        _grid_y = torch.unsqueeze(grid_y, dim=-1)

        _anchor_wh = torch.cat(tensors=(_anchor_w, _anchor_h), dim=-1)
        _anchor_wh = _anchor_wh.repeat((1, 1, 1, 1, self.point_num))

        _grid_xy = torch.cat(tensors=(_grid_x, _grid_y), dim=-1)
        _grid_xy = _grid_xy.repeat((1, 1, 1, 1, self.point_num))
        # 关键点的预测
        input_tensor[..., 5: 5 + 2 * self.point_num] = input_tensor[...,
                                                       5: 5 + 2 * self.point_num] * _anchor_wh + _grid_xy
        return torch.cat([boxes_x, boxes_y, boxes_w, boxes_h, input_tensor[..., 4:]], dim=-1)
