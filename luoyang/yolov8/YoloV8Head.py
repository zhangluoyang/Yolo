import torch
from typing import *
import torch.nn as nn
import luoyang.utils.torch_utils as torch_utils
from luoyang.yolov8.YoloV8Body import make_anchors


class YoloV8Head(nn.Module):

    def __init__(self, reg_max: int,
                 num_classes: int,
                 stride: List[int],
                 feature_size: List[int],
                 use_dfl: bool = True):
        super(YoloV8Head, self).__init__()
        self.output_size = 4 * reg_max + num_classes
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.proj = nn.Parameter(torch.arange(reg_max, dtype=torch.float), requires_grad=False)
        self.feature_size = feature_size
        anchor_points, stride_tensor = make_anchors(feature_size=feature_size,
                                                    strides=stride,
                                                    grid_cell_offset=0.5)
        self.anchor_points = nn.Parameter(anchor_points, requires_grad=False)
        self.stride_tensor = nn.Parameter(stride_tensor, requires_grad=False)

    def bbox_decode(self, anchor_points: torch.Tensor,
                    predict_distance: torch.Tensor):
        if self.use_dfl:
            # batch, anchors, channels
            b, a, c = predict_distance.shape
            predict_distance = predict_distance.view(b, a, 4, c // 4).softmax(3).matmul(
                self.proj.to(predict_distance.device).type(predict_distance.dtype))
        return torch_utils.dist2bbox(distance=predict_distance,
                                     anchor_points=anchor_points,
                                     output_format="xyxy")

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                             torch.Tensor, torch.Tensor]:
        """

        :param features:
        :return:
        """
        predict_distribute, predict_scores = torch.cat([xi.view(features[0].size(0),
                                                                self.output_size, -1) for xi in features], 2).split(
            (self.reg_max * 4, self.num_classes), 1)
        # [b, 8400, class_num]
        predict_scores = predict_scores.permute(0, 2, 1).contiguous()
        # [b, 8400, 16 * 4]
        predict_distribute = predict_distribute.permute(0, 2, 1).contiguous()
        # 解码 对应特征层的输出结果
        predict_scaled_bounding_boxes = self.bbox_decode(anchor_points=self.anchor_points,
                                                         predict_distance=predict_distribute)

        # 最终预测值
        predict_scores = torch.sigmoid(predict_scores)

        return predict_scores, predict_scaled_bounding_boxes, self.stride_tensor, self.anchor_points, predict_distribute
