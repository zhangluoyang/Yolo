from abc import ABC
import copy
import numpy as np
from luoyang.transformer.transformer import Transformer
from typing import *
from luoyang.data.BasicDataSet import BasicDataSet


class DataInMemory(BasicDataSet, ABC):

    def __init__(self,
                 data_list: List[Any],
                 transformers: List[Transformer]):
        super(DataInMemory, self).__init__(epochs=-1)
        self.data_list = data_list
        self.transformers: List[Transformer] = transformers

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """

        :param item:
        :return:
        """
        data_dict: Dict[str, np.ndarray] = copy.deepcopy(self.data_list[item])
        for transformer in self.transformers:
            transformer.transformer(data_dict=data_dict)
        return data_dict
