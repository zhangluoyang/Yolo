from src.model.Model import Model
from src.task.YoloV3 import YoloV3
from src.task.YoloV4 import YoloV4
from src.task.YoloV5 import YoloV5
from src.param.Param import Yolo3Param, Yolo4Param, Yolo5Param
from torch.utils.data import DataLoader
from src.data.YoloDataSet import YoloDataSet, YoloDataSetWithMosaic, BasicDataSet
import src.utils.transformer_utils as transformer_utils
import argparse
from typing import Union, Tuple
import torch.distributed as dist
from torch.backends import cudnn
import torch.multiprocessing
from src.distributed.distributed import DistributedDataParallel

cudnn.benchmark = True
parser = argparse.ArgumentParser(description="yolo train...")
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--parallel', default=True, type=bool)
parser.add_argument('--device', default="cuda:1", type=str)
parser.add_argument('--save_path', default="yolo_v5", type=str)
parser.add_argument('--train_path', default="./resource/all_voc_train.txt")
parser.add_argument('--eval_path', default="./resource/all_voc_test.txt")
parser.add_argument('--model_type', default="yolo_v5", type=str)
parser.add_argument('--model_path', default=None)
parser.add_argument('--num_workers', default=4)


def get_param(model_type: str):
    """

    :param model_type:
    :return:
    """
    if model_type == "yolo_v3":
        return Yolo3Param()
    elif model_type == "yolo_v4":
        return Yolo4Param()
    elif model_type == "yolo_v5":
        return Yolo5Param()
    else:
        raise NotImplemented


def get_data_set(model_type: str,
                 param: Union[Yolo4Param, Yolo3Param, Yolo5Param]) -> Tuple[BasicDataSet, BasicDataSet]:
    """

    :param model_type:
    :param param:
    :return:
    """
    train_path = args.train_path
    eval_path = args.eval_path
    if model_type == "yolo_v3":
        train_transformers = transformer_utils.yolo_v3_train_transformer(param=param)
        test_transformers = transformer_utils.yolo_v3_test_transformer(param=param)
        yolo_train_data_set = YoloDataSet(path=train_path,
                                          epochs=param.epochs,
                                          transformers=train_transformers)
        yolo_eval_data_set = YoloDataSet(path=eval_path,
                                         epochs=param.epochs,
                                         transformers=test_transformers)
    elif model_type == "yolo_v4":
        mosaic_train_transformers, no_mosaic_train_transformers = transformer_utils.yolo_v4_train_transformer(
            param=param)
        test_transformers = transformer_utils.yolo_v4_test_transformer(param=param)
        yolo_train_data_set = YoloDataSetWithMosaic(path=train_path,
                                                    mosaic_head_transformer=mosaic_train_transformers[0],
                                                    mosaic_transformer=mosaic_train_transformers[1],
                                                    mosaic_tail_transformer=mosaic_train_transformers[2],
                                                    no_mosaic_transformer=no_mosaic_train_transformers,
                                                    epochs=param.epochs)
        yolo_eval_data_set = YoloDataSet(path=eval_path,
                                         epochs=param.epochs,
                                         transformers=test_transformers)
    elif model_type == "yolo_v5":
        mosaic_train_transformers, no_mosaic_train_transformers = transformer_utils.yolo_v5_train_transformer(
            param=param)
        test_transformers = transformer_utils.yolo_v5_test_transformer(param=param)
        yolo_train_data_set = YoloDataSetWithMosaic(path=train_path,
                                                    mosaic_head_transformer=mosaic_train_transformers[0],
                                                    mosaic_transformer=mosaic_train_transformers[1],
                                                    mosaic_tail_transformer=mosaic_train_transformers[2],
                                                    no_mosaic_transformer=no_mosaic_train_transformers,
                                                    epochs=param.epochs)
        yolo_eval_data_set = YoloDataSet(path=eval_path,
                                         epochs=param.epochs,
                                         transformers=test_transformers)
    else:
        raise NotImplemented
    return yolo_train_data_set, yolo_eval_data_set


def get_task(model_type: str, param: Union[Yolo4Param, Yolo3Param, Yolo5Param]):
    if model_type == "yolo_v3":
        return YoloV3(param=param)
    elif model_type == "yolo_v4":
        return YoloV4(param=param)
    elif model_type == "yolo_v5":
        return YoloV5(param=param)
    else:
        raise NotImplemented


def main():
    num_workers = args.num_workers
    # torch_utils.set_seed(1024)
    model_type = args.model_type
    param = get_param(model_type=model_type)
    param.nms_threshold = 0.5
    param.conf_threshold = 0.05
    yolo_train_data_set, yolo_eval_data_set = get_data_set(model_type=model_type, param=param)

    if args.parallel:
        device = "cuda:{0}".format(args.local_rank)
        dist.init_process_group(backend="nccl")
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=yolo_train_data_set,
                                                                        shuffle=True)
    else:
        device = args.device
        train_sampler = None

    yolo_task = get_task(model_type=model_type, param=param)

    if args.model_path:
        model_path = args.model_path
        yolo_task.load_state_dict(torch.load(model_path, map_location="cpu"))

    if args.parallel:
        yolo_task.to(device)
        yolo_task = torch.nn.SyncBatchNorm.convert_sync_batchnorm(yolo_task)
        yolo_task = DistributedDataParallel(yolo_task,
                                            device_ids=[args.local_rank],
                                            find_unused_parameters=True)

    loss_layers = yolo_task.build_loss_layer()
    metrics_layers = yolo_task.build_metric_layer()
    yolo_task.to(device)
    model = Model(net=yolo_task,
                  parallel=args.parallel,
                  local_rank=args.local_rank,
                  ema=isinstance(param, Yolo5Param))

    model.compile(loss_layers=loss_layers,
                  metrics_layers=metrics_layers,
                  device=device,
                  num_workers=num_workers,
                  batch_size=param.batch_size,
                  fine_tune_batch_size=param.fine_tune_batch_size,
                  train_epochs=param.train_epochs,
                  fine_epochs=param.fine_epochs)

    model.fit(epochs=param.epochs,
              train_data_set=yolo_train_data_set,
              val_data_set=yolo_eval_data_set,
              train_sample=train_sampler,
              save_path=args.save_path,
              start_step=0,
              model_name="{0}_model".format(model_type))


if __name__ == "__main__":
    args = parser.parse_args()
    main()
