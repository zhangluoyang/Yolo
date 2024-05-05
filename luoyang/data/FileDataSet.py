import json
from abc import ABC
from typing import *
import luoyang.utils.file_utils as file_utils
from luoyang.data.BasicDataSet import BasicDataSet
from luoyang.transformer.transformer import Transformer


class FileDataSet(BasicDataSet, ABC):

    def __init__(self,
                 path: str,
                 transformers: List[Transformer]):
        super(FileDataSet, self).__init__(epochs=-1)
        self.path = path
        self.transformers = transformers

        self.lines = file_utils.read_lines(path=path)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """
        :param item:
        :return:
        """
        data_dict = json.loads(self.lines[item])
        for transformer in self.transformers:
            transformer.transformer(data_dict=data_dict)
        return data_dict
