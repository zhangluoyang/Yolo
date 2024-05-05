import torch
import torch.nn as nn
from typing import List, Dict, Union, Any
import luoyang.utils.torch_utils as torch_utils


class Layer(nn.Module):

    def __init__(self):
        super(Layer, self).__init__()

        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch


class LossLayer(Layer):
    """
    损失函数层
    """

    def __init__(self):
        super(LossLayer, self).__init__()

    def loss_feed_forward(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]) \
            -> Dict[str, torch.Tensor]:
        """

        :param tensor_dict:
        :return:
        """
        raise NotImplemented


class MetricLayer(Layer):
    """
    评估函数层
    """

    def __init__(self, name: str):
        super(MetricLayer, self).__init__()
        self.name: str = name

    def metric_feed_forward(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
                            summary: Dict[str, Any]):
        """

        :param tensor_dict:
        :param summary: 记录中间过程的评估值
        :return:
        """
        raise NotImplemented

    def summary(self, summary: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        生成最终的评估结果
        :param summary:
        :return:
        """
        raise NotImplemented


class TaskLayer(Layer):

    def __init__(self):
        super(TaskLayer, self).__init__()

    def build_loss_layer(self) -> List[LossLayer]:
        """
        损失函数
        :return:
        """
        raise NotImplemented

    def build_metric_layer(self) -> List[MetricLayer]:
        """
        评估函数
        :return:
        """
        raise NotImplemented

    def predict(self, tensor_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]) -> Dict[str, Any]:
        """
        预测函数
        :param tensor_dict:
        :return:
        """
        raise NotImplemented

    def update_fine_tune_param(self):
        raise NotImplemented

    def update_train_param(self):
        raise NotImplemented

    def to_onnx(self, onnx_path: str):
        """

        :param onnx_path:
        :return:
        """
        raise NotImplemented

    def fuse(self):
        raise NotImplemented

    def build_optimizer(self,
                        batch_size: int,
                        epochs: int,
                        epoch: int,
                        init_lr: float = 1e-2,
                        min_lr: float = 1e-4):
        """

        :param batch_size
        :param epochs
        :param epoch
        :param init_lr
        :param min_lr
        :return:
        """
        nbs = 64
        lr_limit_max = 5e-2
        lr_limit_min = 5e-4
        weight_decay = 5e-4
        init_lr_fit = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
        min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        pg0, pg1, pg2 = [], [], []
        for k, v in self.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or "bn" in k or isinstance(v, nn.LayerNorm):
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        optimizer = torch.optim.SGD(pg0, init_lr_fit, momentum=0.937, nesterov=True)
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        lr_scheduler = torch_utils.get_lr_scheduler(lr_decay_type="cos",
                                                    lr=init_lr_fit,
                                                    min_lr=min_lr_fit,
                                                    total_iter=epochs)

        torch_utils.set_optimizer_lr(optimizer=optimizer,
                                     lr_scheduler_func=lr_scheduler,
                                     epoch=epoch)
        return optimizer, None
