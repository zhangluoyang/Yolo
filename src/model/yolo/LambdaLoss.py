import torch
import torch.nn as nn
from typing import Dict, List, Union
from src.model.Layer import LossLayer


class LambdaLoss(LossLayer):

    def __init__(self, name: str):
        super(LambdaLoss, self).__init__()
        self.name = name

    def loss_feed_forward(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]) \
            -> Dict[str, torch.Tensor]:
        """

        :param tensor_dict:
        :return:
        """
        return {"loss": tensor_dict[self.name]}
