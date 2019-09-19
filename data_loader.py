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
    def __init__(self, image_path, mode, structure):
        self.image_path = image_path
        self.label_lstm_path = image_path.replace("_data_trajectory", "_label_trajectory")
        self.label_sum_path = image_path.replace("_data_trajectory", "_label_density")
        self.mode = mode
        self.structure = structure
        self.image_md = list()
        self.label_lstm_md = list()
        self.label_sum_md = list()
        self._get_tuple()

    def _get_tuple(self):

        image_collection = np.load(self.image_path)
        if self.structure == 'basic_cnn' or self.structure == 'pnet':
            label_lstm_collection = np.load(self.label_lstm_path)
            label_sum_collection = np.load(self.label_sum_path)
        elif self.structure == 'rnet':
            label_sum_collection = np.load(self.label_sum_path)
        # assert len(image_collection) == len(label_lstm_collection) == len(label_sum_collection), "image size and label size is not identical"

        for idx, _ in enumerate(image_collection):
            if self.structure == 'basic_cnn' or self.structure == 'pnet':
                self.image_md.append(image_collection[idx])
                self.label_lstm_md.append(label_lstm_collection[idx])
                self.label_sum_md.append(label_sum_collection[idx])
            elif self.structure == 'rnet':
                self.image_md.append(image_collection[idx])
                self.label_sum_md.append(label_sum_collection[idx])
    def _prepare_data(self, idx):
        image_md = self.image_md[idx]
        return  image_md

    def _get_label(self, idx):
        if self.structure == 'basic_cnn' or self.structure == 'pnet':
            label_lstm_md = self.label_lstm_md[idx]
            label_sum_md = self.label_sum_md[idx]
            return label_lstm_md, label_sum_md
        elif self.structure == 'rnet':
            label_sum_md = self.label_sum_md[idx]
            return label_sum_md

    def __len__(self):
        # assert len(self.image_md) == len(self.label_lstm_md) == len(self.label_sum_md), "image size and label size is not identical"
        return len(self.image_md)

    def __getitem__(self, idx):
        sample = dict()
        try:
            image = self._prepare_data(idx)
            if self.structure == 'basic_cnn' or self.structure == 'pnet':
                label_lstm, label_sum = self._get_label(idx)
            elif self.structure == 'rnet':
                label_sum = self._get_label(idx)
        except Exception as e:
            print('error encountered while loading {}'.format(idx))
            print("Unexpected error:", sys.exc_info()[0])
            print(e)
            raise

        if self.structure == 'basic_cnn' or self.structure == 'pnet':
            sample = {'data': image, 'label_lstm': label_lstm, 'label_sum': label_sum}
        elif self.structure == 'rnet':
            sample = {'data': image, 'label_sum': label_sum}
        return sample

    def get_class_count(self):
        if self.structure == 'basic_cnn' or self.structure == 'pnet':
            total = len(self.label_lstm_md) * self.label_lstm_md[0].shape[0] * self.label_lstm_md[0].shape[1] * \
                    self.label_lstm_md[0].shape[2]
            label = self.label_lstm_md
            positive_class = 0
            for label_lstm_md in label:
                positive_class += np.sum(label_lstm_md)
            print("The number of positive image pair is:", positive_class)
            print("The number of negative image pair is:", total - positive_class)
            positive_ratio = positive_class / total
            negative_ratio = (total - positive_class) / total

            return positive_ratio, negative_ratio
        elif self.structure == 'rnet':
            return 0, 0
