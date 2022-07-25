from abc import ABC
import torch
import copy
import math
import time
from tqdm import tqdm
import torch.nn as nn
import src.utils.torch_utils as torch_utils
import src.utils.file_utils as file_utils
from typing import Dict, List, Union, Iterator, Any
from src.model.Layer import MetricLayer, LossLayer, TaskLayer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from src.data.YoloDataSet import BasicDataSet
from torch.optim.optimizer import Optimizer
import src.utils.math_utils as math_utils
from src.distributed.distributed import DistributedDataParallel
import torch.distributed as dist
import src.utils.collate_fn_utils as collate_fn_utils


class ModelEMA(object):

    def __init__(self, task: TaskLayer, decay=0.9999, tau=2000, updates=0):
        super(ModelEMA, self).__init__()
        self.ema: TaskLayer = copy.deepcopy(task).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: TaskLayer):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class Model(nn.Module, ABC):

    def __init__(self,
                 net: Union[TaskLayer, DistributedDataParallel],
                 parallel: bool,
                 local_rank: int,
                 ema: bool = False):
        super(Model, self).__init__()
        self.net = net
        self.parallel = parallel
        self.local_rank = local_rank
        self.ema = ema

        self.loss_layers: List[LossLayer] = NotImplemented
        self.metrics_layers: List[MetricLayer] = NotImplemented

        self.early_stop = NotImplemented
        self.train_epochs = NotImplemented
        self.fine_epochs = NotImplemented
        self.device = NotImplemented
        # 微调的间隔
        self.fine_interval: int = NotImplemented

        self.batch_size: int = NotImplemented

        self.fine_tune_batch_size: int = NotImplemented

        self.num_workers: int = NotImplemented

        self.optimizer: Optimizer = NotImplemented

        self.ema_model: Union[ModelEMA, None] = None

        self.eval_net: Union[TaskLayer, DistributedDataParallel] = NotImplemented

        if self.ema:
            self.ema_model = ModelEMA(net)
            self.eval_net = self.ema_model.ema
            print("ema")
        else:
            self.eval_net = self.net
            self.ema_model = None

    def compile(self, loss_layers: List[LossLayer],
                metrics_layers: List[MetricLayer],
                device: str,
                batch_size: int,
                fine_tune_batch_size: int,
                num_workers: int,
                fine_epochs: int = 0,
                fine_interval: int = 10,
                train_epochs: int = 2,
                early_stop: int = 5):
        self.device = device
        self.loss_layers = loss_layers
        self.fine_interval = fine_interval
        self.fine_epochs = fine_epochs
        self.metrics_layers = metrics_layers
        self.early_stop = early_stop
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.fine_tune_batch_size = fine_tune_batch_size
        self.num_workers = num_workers
        self.net.to(device)
        for layer in self.loss_layers:
            layer.to(device)
        for layer in self.metrics_layers:
            layer.to(device)

    def _train_step(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        训练步骤 (返回损失函数)
        :param tensor_dict:
        :return:
        """
        self.optimizer.zero_grad()
        if isinstance(self.net, DistributedDataParallel):
            tensor_dict = self.net.forward(tensor_dict)
        else:
            tensor_dict = self.net.feed_forward(tensor_dict)
        train_metrics = {}
        loss_list = []
        for loss_layer in self.loss_layers:
            loss_dict = loss_layer.loss_feed_forward(tensor_dict=tensor_dict)
            for name, value in loss_dict.items():
                loss_list.append(value)
                train_metrics[name] = value.item()
        loss = sum(loss_list)
        loss.backward()
        train_metrics["all_loss"] = loss.item()
        self.optimizer.step()
        return train_metrics

    @torch.no_grad()
    def _eval_step(self, tensor_dict: Dict[str, torch.Tensor],
                   epoch: int,
                   summary: Dict[str, Union[float, List[float], Dict[str, Any]]]):
        tensor_dict = self.eval_net.predict(tensor_dict)
        loss_list = []
        for loss_layer in self.loss_layers:
            loss_dict = loss_layer.loss_feed_forward(tensor_dict=tensor_dict)
            for name, value in loss_dict.items():
                loss_list.append(value)
        loss = sum(loss_list)
        summary["all_loss"] = loss.item()
        if epoch >= self.train_epochs:
            for metric_layer in self.metrics_layers:
                if metric_layer.name not in summary:
                    summary[metric_layer.name] = {}
                metric_layer.metric_feed_forward(tensor_dict=tensor_dict, summary=summary[metric_layer.name])

    @torch.no_grad()
    def _eval_summary(self, summary: Dict[str, Union[Dict[str, Any], float]]) \
            -> Dict[str, Union[Dict[str, Dict[str, float]], float]]:
        eval_result_dict = {}
        for metric_layer in self.metrics_layers:
            eval_result = metric_layer.summary(summary=summary[metric_layer.name])
            eval_result_dict[metric_layer.name] = eval_result
        return eval_result_dict

    @staticmethod
    def _display_summary(summary: Dict[str, Union[float, Dict[str, Dict[str, float]]]]):
        for key, value in summary.items():
            if isinstance(value, float):
                print("{0}: {1} \n".format(key, value))
            elif isinstance(value, dict):
                for _key, v in value.items():
                    print("{0}\t{1}\t{2}\n".format(key, _key, v))

    def update_eval_net(self):
        if self.ema:
            self.eval_net = self.ema_model.ema

    def fit(self, epochs: int,
            train_data_set: BasicDataSet,
            val_data_set: Union[BasicDataSet, None] = None,
            train_sample: Union[DistributedSampler, None] = None,
            save_path: str = "./model",
            model_name: str = "model",
            start_step: int = 0):
        """

        :param epochs:
        :param train_data_set:  训练集
        :param val_data_set:  验证集
        :param train_sample:
        :param save_path:  模型存储路径
        :param model_name: 模型名称
        :param start_step
        :return:
        """
        file_utils.mkdir_dirs(path=save_path)
        last_loss = 1e5
        last_index = 0
        for epoch in range(epochs):
            if self.parallel:
                train_sample.set_epoch(epoch=epoch)

            train_data_set.set_epoch(epoch=epoch)
            val_data_set.set_epoch(epoch=epoch)

            if epoch >= start_step:
                is_fine = True if epoch < self.fine_epochs else False
                if is_fine:
                    _batch_size = self.fine_tune_batch_size
                    self.net.update_fine_tune_param()
                else:
                    self.net.update_train_param()
                    _batch_size = self.batch_size

                if isinstance(self.net, DistributedDataParallel):
                    self.optimizer = self.net.module.build_optimizer(batch_size=_batch_size * 2,
                                                                     epochs=epochs,
                                                                     epoch=epoch)
                else:
                    self.optimizer = self.net.build_optimizer(batch_size=_batch_size,
                                                              epochs=epochs,
                                                              epoch=epoch)
                train_data = DataLoader(dataset=train_data_set,
                                        batch_size=_batch_size,
                                        collate_fn=collate_fn_utils.yolo_collate_fn,
                                        num_workers=self.num_workers,
                                        shuffle=not self.parallel,
                                        drop_last=True,
                                        sampler=train_sample)

                eval_data = DataLoader(dataset=val_data_set,
                                       collate_fn=collate_fn_utils.yolo_collate_fn,
                                       num_workers=self.num_workers,
                                       batch_size=_batch_size,
                                       drop_last=False,
                                       shuffle=False)
                self.net.train()
                train_step_num = len(train_data)
                with tqdm(total=train_step_num,
                          desc="train {0}/{1}".format(epoch, epochs),
                          postfix=dict,
                          mininterval=0.3) as bar:
                    train_loss_dict = {}
                    train_loss = 0
                    for step, train_data_tensor in enumerate(train_data):
                        torch_utils.to_device(data_dict=train_data_tensor, device=self.device)
                        train_metrics = self._train_step(tensor_dict=train_data_tensor)
                        math_utils.cal_mean_on_line(on_line_dict=train_loss_dict,
                                                    one_dict={"loss": train_metrics["all_loss"]},
                                                    step=step)
                        train_loss += train_metrics["all_loss"]
                        train_loss_dict["lr"] = torch_utils.get_lr(optimizer=self.optimizer)
                        if self.ema:
                            self.ema_model.update(self.net)
                        if self.local_rank == 0:
                            bar.set_postfix(**train_loss_dict)
                            bar.update(1)

                if epoch >= self.train_epochs and self.local_rank == 0:
                    if self.ema:
                        if isinstance(self.ema_model.ema, DistributedDataParallel):
                            net = self.ema_model.ema.module
                        else:
                            net = self.ema_model.ema
                    else:
                        if isinstance(self.net, DistributedDataParallel):
                            net = self.net.module
                        else:
                            net = self.net
                    state_dict = net.state_dict()
                    torch.save(state_dict, "{0}/{1}_{2}.pth".format(save_path, epoch, model_name))
                if self.local_rank == 0:
                    if eval_data is not None:
                        self.update_eval_net()
                        self.eval_net.eval()
                        eval_summary = {}
                        eval_loss_dict = {}
                        val_step_num = len(eval_data)
                        with tqdm(total=val_step_num,
                                  desc="eval {0}/{1}".format(epoch, epochs),
                                  postfix=dict,
                                  mininterval=0.3) as bar:
                            for step, val_data_tensor in enumerate(eval_data):
                                torch_utils.to_device(data_dict=val_data_tensor, device=self.device)
                                self._eval_step(tensor_dict=val_data_tensor,
                                                epoch=epoch,
                                                summary=eval_summary)
                                math_utils.cal_mean_on_line(on_line_dict=eval_loss_dict,
                                                            one_dict={"loss": eval_summary["all_loss"]},
                                                            step=step)
                                bar.set_postfix(**eval_loss_dict)
                                bar.update(1)
                        if epoch >= self.train_epochs:
                            eval_result = self._eval_summary(summary=eval_summary)
                            self._display_summary(summary=eval_result)
                        if last_loss > eval_loss_dict["loss"]:
                            last_index = epoch
                        elif epoch - last_index >= self.early_stop:
                            print("check early stop ......")
                            break
                        else:
                            file_utils.delta_path_or_file(path="{0}/{1}_{2}.pth".format(save_path,
                                                                                        epoch - self.early_stop,
                                                                                        model_name))
            if self.parallel:
                dist.barrier()
        for epoch in range(epochs):
            if self.local_rank == 0:
                if epoch == last_index:
                    file_utils.rename(src_path="{0}/{1}_{2}.pth".format(save_path, epoch, model_name),
                                      target_path="{0}/{1}.pth".format(save_path, model_name))
                else:
                    file_utils.delta_path_or_file(path="{0}/{1}_{2}.pth".format(save_path, model_name, epoch))
        # onnx
        self.eval_net.eval()
        self.eval_net.to_onnx(onnx_path="{0}/{1}.onnx".format(save_path, model_name))
