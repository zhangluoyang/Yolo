import torch
import torchvision
import torch.nn as nn
from typing import List, Tuple, Union


class YoloTarget(nn.Module):

    def __init__(self,
                 anchors: List[Tuple[int, int]],
                 stride: int,
                 body_output_size: int,
                 anchor_mask: List[int],
                 input_size: Tuple[int, int]):
        super(YoloTarget, self).__init__()
        self.input_size = input_size
        self.body_output_size = body_output_size
        self.anchor_num = len(anchors)
        self.anchors = anchors
        self.stride = stride
        self.anchor_mask = anchor_mask
        self.anchor_mask_num = len(anchor_mask)

    def forward(self, batch_targets: List[Union[torch.Tensor, None]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param batch_targets: [[[x, y, w, h, c], ...], ...]
        :return:
        """
        batch_size = len(batch_targets)

        no_obj_mask = torch.ones(batch_size,
                                 self.anchor_mask_num,
                                 self.input_size[0],
                                 self.input_size[1],
                                 requires_grad=False)

        ground_true = torch.zeros(batch_size,
                                  self.anchor_mask_num,
                                  self.input_size[0],
                                  self.input_size[1],
                                  self.body_output_size,
                                  requires_grad=False)

        for b in range(batch_size):
            if batch_targets[b] is None:
                continue
            target = torch.zeros_like(batch_targets[b])
            target_num = target.size()[0]

            target[:, [0, 2]] = batch_targets[b][:, [0, 2]] * self.input_size[0]
            target[:, [1, 3]] = batch_targets[b][:, [1, 3]] * self.input_size[1]
            target[:, 4] = batch_targets[b][:, 4]

            gt_box = torch.cat((torch.zeros((target_num, 2), dtype=torch.float32), target[:, 2:4]), dim=-1)
            scale_anchors = torch.tensor(self.anchors, requires_grad=False, dtype=torch.float32) / self.stride
            anchors = torch.cat((torch.zeros((self.anchor_num, 2), dtype=torch.float32), scale_anchors), dim=-1)
            gt_anchor_iou = torchvision.ops.box_iou(boxes1=gt_box, boxes2=anchors)

            best_anchor_index = torch.argmax(gt_anchor_iou, dim=-1).detach().numpy()

            for t, best_index in enumerate(best_anchor_index):

                if best_index not in self.anchor_mask:
                    continue

                k = best_index % self.anchor_mask_num

                i = torch.floor(target[t, 0]).long()
                j = torch.floor(target[t, 1]).long()

                no_obj_mask[b, k, j, i] = 0

                c = target[t, 4].long()
                ground_true[b, k, j, i, 0] = target[t, 0]
                ground_true[b, k, j, i, 1] = target[t, 1]
                ground_true[b, k, j, i, 2] = target[t, 2]
                ground_true[b, k, j, i, 3] = target[t, 3]
                ground_true[b, k, j, i, 4] = 1
                ground_true[b, k, j, i, c + 5] = 1

        return ground_true, no_obj_mask
