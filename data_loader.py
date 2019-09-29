import sys
import numpy as np

from torch.utils.data import Dataset, DataLoader

class UAVDatasetTuple(Dataset):
    def __init__(self, image_path, mode, structure):
        self.image_path = image_path
        if structure == 'pnet':
            self.density_path = image_path.replace("_tasks", "_density")
            self.label_path = self.density_path.replace("data_", "label_")
        elif structure == 'rnet':
            self.label_path = image_path.replace("_data_trajectory", "_label_density")
        self.mode = mode
        self.structure = structure
        self.image_md = list()
        self.density_md = list()
        self.label_md = list()
        self._get_tuple()

    def _get_tuple(self):

        image_collection = np.load(self.image_path)
        if self.structure == 'pnet':
            density_collection = np.load(self.density_path)
            label_collection = np.load(self.label_path)
        elif self.structure == 'rnet':
            label_collection = np.load(self.label_path)

        assert len(image_collection) == len(label_collection), "image size and label size is not identical"

        for idx, _ in enumerate(image_collection):
            if self.structure == 'pnet':
                self.image_md.append(image_collection[idx].reshape(image_collection[idx].shape[0] * image_collection[idx].shape[1], image_collection[idx].shape[2]))
                self.density_md.append(density_collection[idx].reshape(1, density_collection[idx].shape[0], density_collection[idx].shape[1]))
                self.label_md.append(label_collection[idx])
            elif self.structure == 'rnet':
                self.image_md.append(image_collection[idx])
                self.label_md.append(label_collection[idx])

    def _prepare_data(self, idx):
        if self.structure == 'pnet':
            image_md = self.image_md[idx]
            density_md = self.density_md[idx]
            return image_md, density_md
        elif self.structure == 'rnet':
            image_md = self.image_md[idx]
            return  image_md

    def _get_label(self, idx):
        label_md = self.label_md[idx]
        return label_md

    def __len__(self):
        assert len(self.image_md) == len(self.label_md), "image size and label size is not identical"
        return len(self.image_md)

    def __getitem__(self, idx):
        sample = dict()
        try:
            if self.structure == 'pnet':
                image, density = self._prepare_data(idx)
                label = self._get_label(idx)
            elif self.structure == 'rnet':
                image = self._prepare_data(idx)
                label = self._get_label(idx)
        except Exception as e:
            print('error encountered while loading {}'.format(idx))
            print("Unexpected error:", sys.exc_info()[0])
            print(e)
            raise

        if self.structure == 'pnet':
            sample = {'data': image, 'density': density, 'label': label}
        elif self.structure == 'rnet':
            sample = {'data': image, 'label': label}
        return sample

    def get_class_count(self):
        if self.structure == 'pnet':
            total = len(self.label_md) * self.label_md[0].shape[0] * self.label_md[0].shape[1]
            label = self.label_md
            positive_class = 0
            for label_md in label:
                positive_class += np.sum(label_md)
            print("The number of positive image pair is:", positive_class)
            print("The number of negative image pair is:", total - positive_class)
            positive_ratio = positive_class / total
            negative_ratio = (total - positive_class) / total

            return positive_ratio, negative_ratio
        elif self.structure == 'rnet':
            total = len(self.label_md) * self.label_md[0].shape[0] * self.label_md[0].shape[1]
            label = self.label_md
            positive_class = 0
            for label_md in label:
                positive_class += np.sum(label_md)
            print("The number of positive image pair is:", positive_class)
            print("The number of negative image pair is:", total - positive_class)
            positive_ratio = positive_class / total
            negative_ratio = (total - positive_class) / total

            return positive_ratio, negative_ratio
