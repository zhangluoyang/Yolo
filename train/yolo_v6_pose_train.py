import argparse
import multiprocessing
import torch.multiprocessing
import torch.distributed as dist
from torch.backends import cudnn
from luoyang.model.Model import Model
from luoyang.yolov6_pose.YoloV6Pose import YoloV6Pose
from torch.utils.data import DataLoader
from luoyang.model.ex.LambdaModel import LambdaModel
import luoyang.utils.transformer_utils as transformer_utils
from luoyang.transformer.transformer import *
from luoyang.param.Param import Yolo6PoseParam
import luoyang.utils.torch_utils as torch_utils
from luoyang.data.YoloDataSet import YoloDataSet, BasicDataSet
import luoyang.utils.collate_fn_utils as collate_fn_utils

from torch.nn.parallel import DistributedDataParallel

cudnn.benchmark = True
parser = argparse.ArgumentParser(description="yolo train... ")
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--parallel', default=False, type=int)
parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--save_path', default="../yolo_model", type=str)
parser.add_argument('--train_path', default="./resource/person_key_train.txt")
parser.add_argument('--eval_path', default="./resource/person_key_val.txt")
parser.add_argument('--model_path', default=None)
parser.add_argument('--device_num', default=1)
parser.add_argument('--num_workers', default=0)


def get_data_set(param: Union[Yolo6PoseParam]) -> Tuple[BasicDataSet, BasicDataSet]:
    """
    :param param:
    :return:
    """
    train_path = args.train_path
    eval_path = args.eval_path

    train_transformers = transformer_utils.get_yolo_v6_pose_train_transformer(param=param)
    test_transformers = transformer_utils.get_yolo_v6_pose_test_transformer(param=param)
    yolo_train_data_set = YoloDataSet(path=train_path,
                                      epochs=param.epochs,
                                      transformers=train_transformers)
    yolo_eval_data_set = YoloDataSet(path=eval_path,
                                     epochs=param.epochs,
                                     transformers=test_transformers)
    return yolo_train_data_set, yolo_eval_data_set


def main():
    num_workers = args.num_workers
    torch_utils.set_seed(1024)
    param = Yolo6PoseParam(m_type="s")
    param.pretrain_path = "/home/zhangluoyang/Desktop/model/yolov6s6.pth"
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

    yolo_task = YoloV6Pose(param=param)

    if args.model_path:
        model_path = args.model_path
        yolo_task.load_state_dict(torch.load(model_path, map_location="cpu"))

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
                  num_workers=num_workers,
                  batch_size=param.batch_size,
                  fine_tune_batch_size=param.fine_tune_batch_size,
                  train_epochs=param.train_epochs,
                  fine_epochs=param.fine_epochs,
                  device_num=args.device_num,
                  early_stop=1000)

    model.fit(epochs=param.epochs,
              train_data_set=yolo_train_data_set,
              val_data_set=yolo_eval_data_set,
              train_sample=train_sampler,
              save_path=args.save_path,
              start_step=0,
              collate_fn=collate_fn_utils.yolo_v6_pose_collate_fn(param.img_size),
              model_name="yolo_v6_s_pose")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    args = parser.parse_args()
    main()
