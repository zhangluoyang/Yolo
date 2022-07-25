from abc import ABC
import json
import random
import numpy as np
import torch
from numpy import ndarray
from torch.utils.data.dataset import Dataset
import src.utils.file_utils as file_utils
from src.transformer.transformer import Transformer
from typing import *


class BasicDataSet(Dataset, ABC):

    def __init__(self, epochs):
        self.epochs = epochs
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch


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
        if len(index_np) < 4:
            return self._sample_mosaic_index(item=item)
        return index_np[: 3]

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """

        :param item:
        :return:
        """
        data: Dict[str, np.ndarray] = json.loads(self.lines[item])
        if self.epoch <= self.epochs * (1 - self.no_mosaic_radio) and random.randint(0, 1):
            mosaic_index = self._sample_mosaic_index(item=item)
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
