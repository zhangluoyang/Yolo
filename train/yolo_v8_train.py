import argparse
import torch.multiprocessing
import torch.distributed as dist
from torch.backends import cudnn
from luoyang.model.Model import Model
from torch.utils.data import DataLoader
from luoyang.model.ex.LambdaModel import LambdaModel
import luoyang.utils.transformer_utils as transformer_utils
from luoyang.transformer.transformer import *
from luoyang.yolov8.YoloV8 import YoloV8
from luoyang.param.Param import Yolo8Param
from luoyang.data.YoloDataSet import YoloDataSet, YoloDataSetWithMosaic, BasicDataSet
import luoyang.utils.collate_fn_utils as collate_fn_utils

from torch.nn.parallel import DistributedDataParallel

cudnn.benchmark = True
parser = argparse.ArgumentParser(description="yolo train... ")
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--parallel', default=False, type=bool)
parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--save_path', default="../yolo_model", type=str)
parser.add_argument('--train_path', default="./resource/all_voc_train.txt")
parser.add_argument('--eval_path', default="./resource/all_voc_test.txt")
parser.add_argument('--model_path', default=None)
parser.add_argument('--device_num', default=1)
parser.add_argument('--num_workers', default=0)


def get_data_set(param: Union[Yolo8Param]) -> Tuple[BasicDataSet, BasicDataSet]:
    """
    :param param:
    :return:
    """
    train_path = args.train_path
    eval_path = args.eval_path

    mosaic_train_transformers, no_mosaic_train_transformers = transformer_utils.yolo_v7_train_transformer(param=param)
    test_transformers = transformer_utils.yolo_v7_test_transformer(param=param)
    yolo_train_data_set = YoloDataSetWithMosaic(path=train_path,
                                                mosaic_head_transformer=mosaic_train_transformers[0],
                                                mosaic_transformer=mosaic_train_transformers[1],
                                                mosaic_tail_transformer=mosaic_train_transformers[2],
                                                no_mosaic_transformer=no_mosaic_train_transformers,
                                                epochs=param.epochs,
                                                no_mosaic_radio=0.5)
    yolo_eval_data_set = YoloDataSet(path=eval_path,
                                     epochs=param.epochs,
                                     transformers=test_transformers)
    return yolo_train_data_set, yolo_eval_data_set


def main():
    num_workers = args.num_workers
    # torch_utils.set_seed(1024)
    param = Yolo8Param(m_type="m")
    param.pretrain_path = "/home/zhangluoyang/Desktop/model/yolov8_m_backbone_weights.pth"
    param.nms_threshold = 0.5
    param.conf_threshold = 0.05
    yolo_train_data_set, yolo_eval_data_set = get_data_set(param=param)

    if args.parallel:
        device = "cuda:{0}".format(args.local_rank)
        dist.init_process_group(backend="nccl")
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=yolo_train_data_set,
                                                                        shuffle=True)
    else:
        device = args.device
        train_sampler = None

    yolo_task = YoloV8(param=param)

    if args.model_path:
        model_path = args.model_path
        _state_dict = torch.load(model_path, map_location="cpu")
        _yolo_state_dict = yolo_task.state_dict()
        state_dict = {}
        for (key, value), (yolo_key, yolo_vlue) in zip(_state_dict.items(), _yolo_state_dict.items()):
            state_dict[yolo_key] = value
        yolo_task.load_state_dict(state_dict)

    loss_layers = yolo_task.build_loss_layer()
    metrics_layers = yolo_task.build_metric_layer()

    if args.parallel:
        yolo_task.to(device)
        yolo_task = torch.nn.SyncBatchNorm.convert_sync_batchnorm(yolo_task)
        yolo_task = DistributedDataParallel(LambdaModel(yolo_task),
                                            device_ids=[args.local_rank],
                                            find_unused_parameters=True)
    else:
        yolo_task.to(device)

    model = Model(net=yolo_task,
                  parallel=args.parallel,
                  local_rank=args.local_rank,
                  ema=True)

    model.compile(loss_layers=loss_layers,
                  metrics_layers=metrics_layers,
                  device=device,
                  device_num=args.device_num,
                  num_workers=num_workers,
                  batch_size=param.batch_size,
                  fine_tune_batch_size=param.fine_tune_batch_size,
                  train_epochs=param.train_epochs,
                  fine_epochs=param.fine_epochs,
                  early_stop=50)

    model.fit(epochs=param.epochs,
              train_data_set=yolo_train_data_set,
              val_data_set=yolo_eval_data_set,
              train_sample=train_sampler,
              save_path=args.save_path,
              start_step=0,
              collate_fn=collate_fn_utils.yolo_v8_collate_fn(img_size=(param.img_size, param.img_size)),
              model_name="yolo_v8_m")


if __name__ == "__main__":
    args = parser.parse_args()
    main()
