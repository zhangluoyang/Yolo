"""
多进程实现的生产者 消费者
"""
import time
import random
from queue import Empty
from multiprocessing import Process, Condition, Lock, Queue


class YoloDataProcess(Process):
    """
    生产者 用于采集数据预处理
    """

    def __init__(self,
                 data_queue: Queue,
                 data_condition: Condition):
        super(YoloDataProcess, self).__init__()
        self.data_condition = data_condition
        self.data_queue = data_queue

    def run(self) -> None:
        _id = 0
        while True:
            _id += 1
            with self.data_condition:
                if self.data_queue.qsize() > 5:
                    # 等待
                    print("----------------堆积太多 生产者等待 消费者通知继续生产-------------------")
                    self.data_condition.wait()
            self.data_queue.put(_id)
            print("##生产者生产数据:{0}##, 此时队列:{1}".format(_id, self.data_queue.qsize()))
            time.sleep(random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))


class YoloInference(Process):
    """
    消费者 模型预测
    """

    def __init__(self, data_queue: Queue,
                 data_condition: Condition,
                 result_queue: Queue,
                 result_condition: Condition):
        super(YoloInference, self).__init__()
        self.data_queue = data_queue
        self.data_condition = data_condition

        self.result_queue = result_queue
        self.result_condition = result_condition

    def run(self) -> None:
        while True:
            _id = None
            with self.data_condition:
                if self.data_queue.qsize() == 2:
                    self.data_condition.notify_all()
            try:
                _id = self.data_queue.get()
                print("@@消费者消费数据:{0}@@, 此时队列:{1}".format(_id, self.data_queue.qsize()))
                time.sleep(random.choice([1.1, 1.2, 1.3, 1.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
            except Empty as e:
                pass

            with self.result_condition:
                if self.result_queue.qsize() > 5:
                    print("后处理程序积累太多 等待")
                    self.result_condition.wait()

            if _id is not None:
                self.result_queue.put(_id)


class YoloResult(Process):
    """

    """

    def __init__(self, result_queue: Queue,
                 result_condition: Condition):
        super(YoloResult, self).__init__()
        self.result_queue = result_queue
        self.result_condition = result_condition

    def run(self) -> None:
        while True:
            with self.result_condition:
                if self.result_queue.qsize() == 2:
                    self.result_condition.notify_all()

            try:
                _id = self.result_queue.get()
                print("&&后处理处理数据:{0}&&, 此时后处理队列:{1}".format(_id, self.result_queue.qsize()))
                time.sleep(random.choice([2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.0]))

            except Empty as e:
                pass


def run_forever(result_condition, result_queue: Queue):
    while True:
        with result_condition:
            if result_queue.qsize() == 2:
                result_condition.notify_all()

        try:
            _id = result_queue.get()
            print("&&后处理处理数据:{0}&&, 此时后处理队列:{1}".format(_id, result_queue.qsize()))
            time.sleep(random.choice([2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.0]))

        except Empty as e:
            pass


if __name__ == "__main__":
    _data_queue = Queue()
    _result_queue = Queue()
    _data_condition = Condition(Lock())
    _result_condition = Condition(Lock())

    yolo_process = YoloDataProcess(data_queue=_data_queue,
                                   data_condition=_data_condition)

    yolo_inference = YoloInference(data_queue=_data_queue,
                                   data_condition=_data_condition,
                                   result_queue=_result_queue,
                                   result_condition=_result_condition)

    yolo_process.start()
    yolo_inference.start()

    run_forever(result_condition=_result_condition,
                result_queue=_result_queue)

    yolo_process.join()
    yolo_inference.join()
