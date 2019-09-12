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
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode
        self.image_md = list()
        self.label_md = list()
        pass

    def _get_tuple(self):
        image_path = ""
        label_path = ""

        image_collection = read_pickle(image_path)
        label_collection = read_pickle(label_path)

        assert len(image_collection) == len(label_collection), "image size and label size is not identical"

        for idx, _ in enumerate(image_collection):
            self.image_md.append(image_collection[idx])
            self.label_md.append(label_collection[idx])

    def _prepare_data(self, idx):
        image_md = self.image_md[idx]
        return  image_md

    def _get_label(self, idx):
        label_md = self.label_md[idx]
        return label_md

    def __len__(self):
        assert len(self.image_md) == len(self.label_md), "image size and label size is not identical"
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

        sample = {'image': image, 'label_lstm': label_lstm, 'label_cnn': label_cnn}

        return sample