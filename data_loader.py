import os
import cv2
import sys
import torch
import torchvision
import itertools
import numpy as np

from utils import read_pickle, dump_pickle
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class UAVDatasetTuple(Dataset):
    def __init__(self, image_path, mode):
        self.image_path = image_path
        self.label_lstm_path = image_path.replace("_data_trajectory", "_label_trajectory")
        self.label_sum_path = image_path.replace("_data_trajectory", "_label_density")
        self.mode = mode
        self.image_md = list()
        self.label_lstm_md = list()
        self.label_sum_md = list()
        self._get_tuple()

    def _get_tuple(self):
        image_collection = np.load(self.image_path)
        label_lstm_collection = np.load(self.label_lstm_path)
        label_sum_collection = np.load(self.label_sum_path)

        assert len(image_collection) == len(label_lstm_collection) == len(label_sum_collection), "image size and label size is not identical"

        for idx, _ in enumerate(image_collection):
            self.image_md.append(image_collection[idx])
            self.label_lstm_md.append(label_lstm_collection[idx])
            self.label_sum_md.append(label_sum_collection[idx])

    def _prepare_data(self, idx):
        image_md = self.image_md[idx]
        return  image_md

    def _get_label(self, idx):
        label_lstm_md = self.label_lstm_md[idx]
        label_sum_md = self.label_sum_md[idx]
        return label_lstm_md, label_sum_md

    def __len__(self):
        assert len(self.image_md) == len(self.label_lstm_md) == len(self.label_sum_md), "image size and label size is not identical"
        return len(self.image_md)

    def __getitem__(self, idx):

        try:
            image = self._prepare_data(idx)
            label_lstm, label_cnn = self._get_label(idx)
        except Exception as e:
            print('error encountered while loading {}'.format(idx))
            print("Unexpected error:", sys.exc_info()[0])
            print(e)
            raise

        sample = {'data': image, 'label_lstm': label_lstm, 'label_sum': label_cnn}

        return sample