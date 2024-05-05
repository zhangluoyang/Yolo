from abc import ABC
import json
import copy
import random
import numpy as np
import luoyang.utils.file_utils as file_utils
from luoyang.transformer.transformer import Transformer
from typing import *
from luoyang.data.BasicDataSet import BasicDataSet


class YoloDataSet(BasicDataSet, ABC):
    """
    yolo data set
    """

    def __init__(self, path: str,
                 transformers: List[Transformer],
                 epochs: int):
        super(YoloDataSet, self).__init__(epochs=epochs)
        self.path = path
        self.lines = file_utils.read_lines(path=path)
        self.data_dict_list = [json.loads(line) for line in self.lines]
        self.transformers = transformers

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """

        :param item:
        :return:
        """
        data_dict: Dict[str, np.ndarray] = json.loads(self.lines[item])
        for transformer in self.transformers:
            transformer.transformer(data_dict=data_dict)
        return data_dict


class YoloDataSetWithMosaic(BasicDataSet, ABC):
    """
    """

    def __init__(self, path: str,
                 mosaic_head_transformer: List[Transformer],
                 mosaic_transformer: Transformer,
                 mosaic_tail_transformer: List[Transformer],
                 no_mosaic_transformer: List[Transformer],
                 epochs: int,
                 no_mosaic_radio: float = 0.3):
        super(YoloDataSetWithMosaic, self).__init__(epochs=epochs)
        self.path = path
        self.lines = file_utils.read_lines(path=path)
        self.mosaic_head_transformer = mosaic_head_transformer
        self.mosaic_transformer = mosaic_transformer
        self.mosaic_tail_transformer = mosaic_tail_transformer
        self.no_mosaic_radio = no_mosaic_radio
        self.no_mosaic_transformer = no_mosaic_transformer

        self.all_sample_index = NotImplemented

        self.item_mosaic = NotImplemented

    def __len__(self):
        return len(self.lines)

    def _sample_mosaic_index(self, item: int) -> np.ndarray:
        """
        随机采样选择3张图片
        :param item:
        :return:
        """
        index_np = np.random.randint(low=0, high=self.__len__(), size=100)
        index_np = index_np[index_np != item]
        if len(index_np) < 3:
            return self._sample_mosaic_index(item=item)
        return index_np[: 3]

    def _set_sample_mosaic_index(self):
        """
        随机采样选择3张图片
        :return:
        """
        self.all_sample_index = np.random.randint(low=0, high=self.__len__(), size=(len(self.lines), 100))

        self.item_mosaic = np.random.randint(0, 1, size=len(self.lines))

    def set_epoch(self, epoch):
        super(YoloDataSetWithMosaic, self).set_epoch(epoch=epoch)
        # 重置采样
        self._set_sample_mosaic_index()

    def _get_sample_mosaic_index(self, item: int) -> np.ndarray:
        index_np = self.all_sample_index[item]
        index_np = index_np[index_np != item]
        if len(index_np) < 3:
            # 不成功 重新采样
            return self._sample_mosaic_index(item=item)
        return index_np[: 3]

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """

        :param item:
        :return:
        """
        data: Dict[str, np.ndarray] = json.loads(self.lines[item])
        if self.epoch <= self.epochs * (1 - self.no_mosaic_radio) and self.item_mosaic[item]:
            mosaic_index = self._get_sample_mosaic_index(item=item)
            sample_data_list = [json.loads(self.lines[index]) for index in mosaic_index]
            sample_data_list.append(data)
            for data_dict in sample_data_list:
                for transformer in self.mosaic_head_transformer:
                    transformer.transformer(data_dict=data_dict)
            mosaic_data_dict = self.mosaic_transformer.batch_transformer(data_dict_list=sample_data_list)
            for transformer in self.mosaic_tail_transformer:
                transformer.transformer(data_dict=mosaic_data_dict)
            data = mosaic_data_dict
        else:
            for transformer in self.no_mosaic_transformer:
                transformer.transformer(data_dict=data)
        return data
