from luoyang.model.Layer import LossLayer, MetricLayer, TaskLayer
from typing import *
import torch


class LambdaLoss(LossLayer):

    def __init__(self, loss_name: str, lambda_name: str = ""):
        super(LambdaLoss, self).__init__()
        self.loss_name = loss_name
        self.lambda_name = lambda_name

    def loss_feed_forward(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]) \
            -> Dict[str, torch.Tensor]:
        return {self.lambda_name: tensor_dict[self.loss_name]}
