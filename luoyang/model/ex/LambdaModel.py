import torch
from typing import *
import torch.nn as nn
from luoyang.model.Layer import TaskLayer


class LambdaModel(nn.Module):

    def __init__(self, task: TaskLayer):
        super(LambdaModel, self).__init__()
        self.task = task

    def forward(self, tensor_dict: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, torch.Tensor]:
        return self.task.forward(tensor_dict=tensor_dict)
