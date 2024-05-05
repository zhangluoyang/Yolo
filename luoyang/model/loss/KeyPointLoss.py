import torch
import torch.nn as nn
from typing import *
import numpy as np


class PoseKeypointLoss(nn.Module):
    """
    ptk loss
    """

    def __init__(self, sigmas: Union[None, List[float]] = None) -> None:
        super(PoseKeypointLoss, self).__init__()
        if sigmas is None:
            self.sigmas = torch.tensor(np.array(
                [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0)
        else:
            self.sigmas = torch.tensor(sigmas)

    def forward(self, p_kpt: torch.Tensor,
                gt_kpt: torch.Tensor,
                kpt_mask: torch.Tensor,
                area: torch.Tensor):
        """

        :param p_kpt: [num_pos, 2 * point_num]
        :param gt_kpt: [num_pos, 2 * point_num]
        :param kpt_mask: [num_pos, point_num]
        :param area: [num_pos, 1]
        :return:
        """
        # [num_pos, point_num]
        d = (p_kpt[..., 0::2] - gt_kpt[..., 0::2]) ** 2 + (p_kpt[..., 1::2] - gt_kpt[..., 1::2]) ** 2
        # [num_pos, ]
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask, dim=-1) + 1e-9)
        # [num_pos, point_num]
        e: torch.Tensor = d / (2 * self.sigmas.to(d.device)) ** 2 / (area + 1e-9) / 2
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()
