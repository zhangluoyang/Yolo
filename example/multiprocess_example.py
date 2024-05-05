import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import torch
from queue import Empty
from luoyang.param.Param import Yolo6Param
import luoyang.utils.draw_utils as draw_utils
from luoyang.predict.YoloPredict import YoloPredict
from luoyang.transformer.inference_transformer import *
from multiprocessing import Process, Condition, Lock, Queue
import luoyang.utils.torch_utils as torch_utils


def single_process():
    start_time = time.time()
    param = Yolo6Param(m_type="s")
    object_process = ObjectDetectProcess(target_size=(param.img_size, param.img_size))
    cap = cv2.VideoCapture("../resource/caiji.mp4")
    onnx_path = "../../yolo_model/yolo_v6_s.onnx"
    device = "cuda:0"
    yolo = YoloPredict(onnx_path=onnx_path,
                       device=device,
                       input_size=(param.img_size, param.img_size),
                       output_size=25,
                       conf_threshold=0.5)
    while True:
        ret, ori_image = cap.read()
        if ori_image is None:
            break
        warped_image, inv_m = object_process.process(image=ori_image)
        # print(warped_image[0], warped_image.shape)
        predicts = yolo.predict(feed_dict={"images": np.array([warped_image])})
        predict = predicts[0]
        predict = object_process.restore(predict=predict, inv_m=inv_m)
        if predict is not None:
            boxes = predict[:, :4].astype(np.int32)
            class_ids = predict[:, -1].astype(np.int32)
            class_names = [param.class_names[int(_id)] for _id in class_ids]
            img = draw_utils.draw_bbox_labels(ori_image, boxes=boxes, labels=class_names)
            cv2.imshow("yolo_v6_video", img)
            cv2.waitKey(1)
        # break
    end_time = time.time()
    print("time consume:{0}".format(end_time - start_time))


class YoloDataProcess(Process):
    """
    生产者 用于采集数据预处理
    """

    def __init__(self,
                 data_queue: Queue,
                 data_condition: Condition,
                 target_size: Tuple[int, int] = (640, 640),
                 limit_size: int = 10):
        super(YoloDataProcess, self).__init__()
        self.data_condition = data_condition
        self.data_queue = data_queue
        self.object_process = ObjectDetectProcess(target_size=target_size)
        self.limit_size = limit_size

    def run(self) -> None:
        cap = cv2.VideoCapture("../resource/pretty.mp4")
        while True:
            ret, ori_image = cap.read()
            if ori_image is None:
                self.data_queue.put((None, None, None))
                break
            warped_image, inv_m = self.object_process.process(image=ori_image)
            with self.data_condition:
                if self.data_queue.qsize() > self.limit_size:
                    self.data_condition.wait()
            self.data_queue.put((warped_image, inv_m, ori_image))


class YoloInference(Process):
    """
    消费者 模型预测
    """

    def __init__(self, data_queue: Queue,
                 data_condition: Condition,
                 result_queue: Queue,
                 result_condition: Condition,
                 onnx_path: str,
                 output_size: int,
                 conf_threshold: float,
                 input_size: Tuple[int, int] = (640, 640),
                 limit_size: int = 10):
        super(YoloInference, self).__init__()
        self.data_queue = data_queue
        self.data_condition = data_condition
        self.output_size = output_size
        self.conf_threshold = conf_threshold
        self.result_queue = result_queue
        self.result_condition = result_condition
        self.onnx_path = onnx_path
        self.input_size = input_size

        self.limit_size = limit_size

    def run(self) -> None:
        yolo = YoloPredict(onnx_path=self.onnx_path,
                           device="cuda:0",
                           input_size=self.input_size,
                           output_size=self.output_size,
                           conf_threshold=self.conf_threshold)

        while True:
            ori_image, predicts, inv_m = None, None, None
            with self.data_condition:
                if self.data_queue.qsize() <= self.limit_size / 2:
                    self.data_condition.notify_all()
            try:
                warped_image, inv_m, ori_image = self.data_queue.get()
                if warped_image is None and inv_m is None and ori_image is None:
                    self.result_queue.put((None, None, None))
                    break
                predicts = yolo.sess_predict(feed_dict={"images": np.array([warped_image])})
            except Empty as e:
                pass
            with self.result_condition:
                if self.result_queue.qsize() > self.limit_size:
                    self.result_condition.wait()
            if ori_image is not None:
                self.result_queue.put((ori_image, predicts, inv_m))


def run_forever(result_condition,
                result_queue: Queue,
                param: Yolo6Param,
                target_size: Tuple[int, int] = (640, 640),
                limit_size: int = 10):
    object_process = ObjectDetectProcess(target_size=target_size)
    out = cv2.VideoWriter("./caiji.mp4", )
    while True:
        with result_condition:
            if result_queue.qsize() <= limit_size / 2:
                result_condition.notify_all()

        try:
            ori_image, predicts, inv_m = result_queue.get()
            if ori_image is None and predicts is None and inv_m is None:
                break
            batch_nms_predicts = torch_utils.non_max_suppression(prediction=torch.tensor(predicts).cuda(),
                                                                 conf_threshold=param.conf_threshold,
                                                                 nms_threshold=param.nms_threshold,
                                                                 img_size=param.img_size)
            predict = batch_nms_predicts[0]
            if predict is not None:
                predict = predict.detach().cpu().numpy()
                predict = object_process.restore(predict=predict, inv_m=inv_m)
                boxes = predict[:, :4].astype(np.int32)
                class_ids = predict[:, -1].astype(np.int32)
                class_names = [param.class_names[int(_id)] for _id in class_ids]
                img = draw_utils.draw_bbox_labels(ori_image, boxes=boxes, labels=class_names)
                cv2.imshow("yolo_v6_video", img)
                cv2.waitKey(1)
        except Empty as e:
            pass


if __name__ == "__main__":
    # single_process()
    start = time.time()
    _onnx_path = "../../yolo_model/yolo_v6_s.onnx"
    _param = Yolo6Param(m_type="s")
    _param.conf_threshold = 0.5

    _data_queue = Queue()
    _result_queue = Queue()
    _data_condition = Condition(Lock())
    _result_condition = Condition(Lock())

    yolo_process = YoloDataProcess(data_queue=_data_queue,
                                   data_condition=_data_condition)

    yolo_inference = YoloInference(data_queue=_data_queue,
                                   data_condition=_data_condition,
                                   result_queue=_result_queue,
                                   onnx_path=_onnx_path,
                                   result_condition=_result_condition,
                                   output_size=25,
                                   conf_threshold=_param.conf_threshold)

    yolo_process.start()
    yolo_inference.start()

    run_forever(result_condition=_result_condition,
                result_queue=_result_queue,
                param=_param)

    yolo_process.join()
    yolo_inference.join()
    end = time.time()
    print(end - start)
