from torch.utils.data.dataset import Dataset
from abc import ABC


class BasicDataSet(Dataset, ABC):

    def __init__(self, epochs):
        self.epochs = epochs
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
