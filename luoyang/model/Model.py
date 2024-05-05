import copy
import math
import torch
from abc import ABC
from tqdm import tqdm
import torch.nn as nn
import torch.distributed as dist
from typing import *
from torch.optim.optimizer import Optimizer
import luoyang.utils.file_utils as file_utils
from torch.utils.data.dataloader import DataLoader
import luoyang.utils.math_utils as math_utils
import luoyang.utils.torch_utils as torch_utils
from luoyang.data.YoloDataSet import BasicDataSet
from luoyang.model.ex.LambdaModel import LambdaModel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import StepLR
from luoyang.model.Layer import MetricLayer, LossLayer, TaskLayer
from torch.cuda.amp import autocast, GradScaler


class ModelEMA(object):

    def __init__(self, task: TaskLayer, decay=0.9999, tau=2000, updates=0):
        super(ModelEMA, self).__init__()
        self.ema: TaskLayer = copy.deepcopy(task).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module):
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
        self.early_mode: str = "desc"
        self.early_info: Union[str, Tuple[str, str, str]] = "loss"
        self.early_stop_best_index = 0
        self.early_stop_best_value: float = NotImplemented

        self.train_epochs = NotImplemented
        self.fine_epochs = NotImplemented
        self.device = NotImplemented
        # 微调的间隔
        self.fine_interval: int = NotImplemented

        self.batch_size: int = NotImplemented

        self.fine_tune_batch_size: int = NotImplemented

        self.num_workers: int = NotImplemented

        self.optimizer: Optimizer = NotImplemented

        self.scheduler: Union[StepLR, None] = None

        self.ema_model: Union[ModelEMA, None] = None

        self.eval_net: Union[TaskLayer, DistributedDataParallel] = NotImplemented

        self.gradScaler = GradScaler()

        if self.ema:
            self.ema_model = ModelEMA(net)
            self.eval_net = self.ema_model.ema
        else:
            self.eval_net = self.net
            self.ema_model = None

        self.device_num = 1

    def compile(self, loss_layers: List[LossLayer],
                metrics_layers: List[MetricLayer],
                device: str,
                batch_size: int,
                fine_tune_batch_size: int,
                num_workers: int,
                early_mode: str = "desc",
                early_info: Union[str, Tuple[str, str, str]] = "loss",
                fine_epochs: int = 0,
                fine_interval: int = 10,
                train_epochs: int = 2,
                early_stop: int = 5,
                device_num: int = 1):
        self.device = device
        self.early_mode = early_mode
        self.early_info = early_info
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
        self.device_num = device_num

        self.early_stop_best_value = 1e12 if early_mode == "desc" else -1e10

    def _train_step(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        训练步骤 (返回损失函数)
        :param tensor_dict:
        :return:
        """
        self.optimizer.zero_grad()
        with autocast():
            tensor_dict = self.net.forward(tensor_dict)
            train_metrics = {}
            loss_list = []
            for loss_layer in self.loss_layers:
                loss_dict = loss_layer.loss_feed_forward(tensor_dict=tensor_dict)
                for name, value in loss_dict.items():
                    loss_list.append(value)
                    train_metrics[name] = value.detach().item()

            loss = sum(loss_list)
            self.gradScaler.scale(loss).backward()
            train_metrics["all_loss"] = loss.detach().item()
            # self.optimizer.step()
            self.gradScaler.step(self.optimizer)
            self.gradScaler.update()
            # torch_utils.empty_cache()
        return train_metrics

    @torch.no_grad()
    def _eval_step(self, tensor_dict: Dict[str, torch.Tensor],
                   epoch: int,
                   summary: Dict[str, Union[float, List[float], Dict[str, Any]]]):
        with autocast():
            tensor_dict = self.eval_net.predict(tensor_dict)
            loss_list = []
            for loss_layer in self.loss_layers:
                loss_dict = loss_layer.loss_feed_forward(tensor_dict=tensor_dict)
                for name, value in loss_dict.items():
                    loss_list.append(value)
            loss = sum(loss_list)
            summary["all_loss"] = loss.item()
            for metric_layer in self.metrics_layers:
                if metric_layer.name not in summary:
                    summary[metric_layer.name] = {}
                metric_layer.metric_feed_forward(tensor_dict=tensor_dict, summary=summary[metric_layer.name])
            # torch_utils.empty_cache()

    @torch.no_grad()
    def _eval_summary(self, summary: Dict[str, Union[Dict[str, Any], float]]) \
            -> Dict[str, Union[Dict[str, Dict[str, float]], float]]:
        eval_result_dict = {"loss": summary["all_loss"]}
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

    def update_early_stop(self,
                          current_epoch: int,
                          summary: Dict[str, Union[float, Dict[str, Dict[str, float]]]]) -> bool:
        if isinstance(self.early_info, str):
            value = summary[self.early_info]
        elif isinstance(self.early_info, tuple):
            _key1 = self.early_info[0]
            _key2 = self.early_info[1]
            _key3 = self.early_info[2]
            value = summary[_key1][_key2][_key3]
        else:
            raise NotImplemented
        if self.early_mode == "desc":
            if self.early_stop_best_value > value:
                self.early_stop_best_value = value
                self.early_stop_best_index = current_epoch
        else:
            if self.early_stop_best_value < value:
                self.early_stop_best_value = value
                self.early_stop_best_index = current_epoch

        print("current_epoch:{0}, value:{1}, best_value:{2}, best_index:{3}".format(current_epoch,
                                                                                    value,
                                                                                    self.early_stop_best_value,
                                                                                    self.early_stop_best_index))

        return current_epoch - self.early_stop_best_index >= self.early_stop

    def update_eval_net(self):
        if self.ema:
            self.eval_net = self.ema_model.ema
            if isinstance(self.eval_net, DistributedDataParallel):
                self.eval_net = self.eval_net.module
                if isinstance(self.eval_net, LambdaModel):
                    self.eval_net = self.eval_net.task
        elif isinstance(self.eval_net, DistributedDataParallel):
            self.eval_net = self.eval_net.module
            if isinstance(self.eval_net, LambdaModel):
                self.eval_net = self.eval_net.task

    def fit(self, epochs: int,
            train_data_set: BasicDataSet,
            collate_fn,
            val_data_set: Union[BasicDataSet, None] = None,
            test_data_set: Union[BasicDataSet, None] = None,
            train_sample: Union[DistributedSampler, None] = None,
            save_path: str = "./model",
            model_name: str = "model",
            start_step: int = 0):
        """

        :param epochs:
        :param train_data_set:  训练集
        :param val_data_set:  验证集
        :param test_data_set
        :param train_sample:
        :param collate_fn:
        :param save_path:  模型存储路径
        :param model_name: 模型名称
        :param start_step
        :return:
        """
        file_utils.mkdir_dirs(path=save_path)
        for epoch in range(epochs):
            if self.parallel:
                train_sample.set_epoch(epoch=epoch)
            train_data_set.set_epoch(epoch=epoch)
            for loss_layer in self.loss_layers:
                loss_layer.set_epoch(epoch=epoch)
            if epoch >= start_step:
                is_fine = True if epoch < self.fine_epochs else False
                if is_fine:
                    _batch_size = self.fine_tune_batch_size
                    if isinstance(self.net, DistributedDataParallel):
                        if isinstance(self.net.module, LambdaModel):
                            self.net.module.task.update_fine_tune_param()
                        else:
                            raise NotImplemented
                    else:
                        self.net.update_fine_tune_param()
                else:
                    if isinstance(self.net, DistributedDataParallel):
                        if isinstance(self.net.module, LambdaModel):
                            self.net.module.task.update_train_param()
                        else:
                            raise NotImplemented
                    else:
                        self.net.update_train_param()
                    _batch_size = self.batch_size

                if self.scheduler is None:

                    if isinstance(self.net, DistributedDataParallel):
                        if isinstance(self.net.module, LambdaModel):
                            self.optimizer, self.scheduler = self.net.module.task.build_optimizer(
                                batch_size=_batch_size * self.device_num,
                                epochs=epochs,
                                epoch=epoch)
                        else:
                            raise NotImplemented
                    else:
                        self.optimizer, self.scheduler = self.net.build_optimizer(batch_size=_batch_size,
                                                                                  epochs=epochs,
                                                                                  epoch=epoch)
                else:
                    self.scheduler.step()
                train_data = DataLoader(dataset=train_data_set,
                                        batch_size=_batch_size,
                                        collate_fn=collate_fn,
                                        num_workers=self.num_workers,
                                        shuffle=not self.parallel,
                                        drop_last=True,
                                        sampler=train_sample)
                self.net.train()
                train_step_num = len(train_data)
                with tqdm(total=train_step_num,
                          desc="train {0}/{1}".format(epoch, epochs),
                          postfix=dict,
                          mininterval=0.3) as bar:
                    train_loss_dict = {}
                    train_loss = 0
                    train_loss_dict["lr"] = torch_utils.get_lr(optimizer=self.optimizer)
                    for step, train_data_tensor in enumerate(train_data):
                        torch_utils.to_device(data_dict=train_data_tensor, device=self.device)
                        train_metrics = self._train_step(tensor_dict=train_data_tensor)
                        math_utils.cal_mean_on_line(on_line_dict=train_loss_dict,
                                                    one_dict={"loss": train_metrics["all_loss"]},
                                                    step=step)
                        train_loss += train_metrics["all_loss"]
                        if self.ema:
                            self.ema_model.update(self.net)
                        if self.local_rank == 0:
                            bar.set_postfix(**train_loss_dict)
                            bar.update(1)

                if epoch >= self.train_epochs and self.local_rank == 0:
                    if self.ema:
                        if isinstance(self.ema_model.ema, DistributedDataParallel):
                            net = self.ema_model.ema.module
                            if isinstance(net, LambdaModel):
                                net = net.task
                            else:
                                raise NotImplemented
                        else:
                            net = self.ema_model.ema
                    else:
                        if isinstance(self.net, DistributedDataParallel):
                            if isinstance(self.net.module, LambdaModel):
                                net = self.net.module.task
                            else:
                                raise NotImplemented
                        else:
                            net = self.net
                    state_dict = net.state_dict()
                    torch.save(state_dict, "{0}/{1}_{2}.pth".format(save_path, epoch, model_name))
                if self.local_rank == 0:
                    if epoch >= self.train_epochs:
                        if val_data_set is not None:
                            eval_data = DataLoader(dataset=val_data_set,
                                                   collate_fn=collate_fn,
                                                   num_workers=self.num_workers,
                                                   batch_size=_batch_size,
                                                   drop_last=False,
                                                   shuffle=False)
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
                                print("check early stop......")
                                if self.update_early_stop(summary=eval_result, current_epoch=epoch):
                                    print("early stop......")
                                    break
                                else:
                                    file_utils.delta_path_or_file(path="{0}/{1}_{2}.pth".format(save_path,
                                                                                                epoch - self.early_stop,
                                                                                                model_name))
            if self.parallel:
                dist.barrier()
        for epoch in range(epochs):
            if self.local_rank == 0:
                if epoch == self.early_stop_best_index:
                    file_utils.rename(src_path="{0}/{1}_{2}.pth".format(save_path, epoch, model_name),
                                      target_path="{0}/{1}.pth".format(save_path, model_name))
                else:
                    file_utils.delta_path_or_file(path="{0}/{1}_{2}.pth".format(save_path, epoch, model_name))
        if self.local_rank == 0:
            if test_data_set is not None:
                self.update_eval_net()
                self.eval_net.eval()
                test_summary = {}
                test_loss_dict = {}
                test_data_load = DataLoader(dataset=test_data_set,
                                            collate_fn=collate_fn,
                                            num_workers=self.num_workers,
                                            batch_size=self.batch_size,
                                            drop_last=False,
                                            shuffle=False)
                test_step_num = len(test_data_load)
                with tqdm(total=test_step_num,
                          desc="eval {0}/{1}".format(epochs, epochs),
                          postfix=dict,
                          mininterval=0.3) as bar:
                    for step, val_data_tensor in enumerate(test_data_load):
                        torch_utils.to_device(data_dict=val_data_tensor, device=self.device)
                        self._eval_step(tensor_dict=val_data_tensor,
                                        epoch=epochs,
                                        summary=test_summary)
                        math_utils.cal_mean_on_line(on_line_dict=test_loss_dict,
                                                    one_dict={"loss": test_summary["all_loss"]},
                                                    step=step)
                        bar.set_postfix(**test_loss_dict)
                        bar.update(1)
                eval_result = self._eval_summary(summary=test_summary)
                self._display_summary(summary=eval_result)
        # onnx
        # if self.local_rank == 0:
        #     self.update_eval_net()
        #     self.eval_net.eval()
        #     self.eval_net.to_onnx(onnx_path="{0}/{1}.onnx".format(save_path, model_name))
