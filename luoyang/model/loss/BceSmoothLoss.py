import torch
import torch.nn as nn
from typing import Union


def _clip_by_tensor(t: torch.Tensor, t_min: float, t_max: float) -> torch.Tensor:
    """

    :param t:
    :param t_min:
    :param t_max:
    :return:
    """
    t = t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


class BceSmoothLoss(nn.Module):

    def __init__(self,
                 reduce: Union[str, None] = None,
                 epsilon: float = 1e-7):
        super(BceSmoothLoss, self).__init__()
        self.reduce = reduce
        self.epsilon = epsilon

    def forward(self, predict: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        smooth_predict = _clip_by_tensor(t=predict, t_min=self.epsilon, t_max=1 - self.epsilon)
        loss = - target * torch.log(smooth_predict) - (1.0 - target) * torch.log(1.0 - smooth_predict)
        if self.reduce == "mean":
            return torch.mean(loss)
        elif self.reduce == "sum":
            return torch.sum(loss)
        else:
            return loss
