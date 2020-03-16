from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import torch


class UAVDatasetTuple(Dataset):
    def __init__(self, task_label_path, init_path, last_label_path, avg_label_path):
        self.task_label_path = task_label_path
        self.init_path = init_path
        self.last_label_path = last_label_path
        self.avg_label_path = avg_label_path

        self.last_label_md = []
        self.avg_label_md = []
        self.init_md = []
        self.task_label_md = []

        self._get_tuple()

    def __len__(self):
        return len(self.last_label_md)

    def _get_tuple(self):
        self.task_label_md = np.load(self.task_label_path).astype(float)
        self.init_md = np.load(self.init_path).astype(float)
        self.last_label_md = np.load(self.last_label_path).astype(float)
        self.avg_label_md = np.load(self.avg_label_path).astype(float)

    def __getitem__(self, idx):
        try:
            task_label = self._prepare_task_label(idx)
            init = self._prepare_init(idx)
            last_label = self._get_last_label(idx)
            avg_label = self._get_avg_label(idx)

            # normal evaluation
            # init = np.expand_dims(init, axis=0)

            # continuous evaluation
            init = np.expand_dims(init, axis=1)
        except Exception as e:
            print('error encountered while loading {}'.format(idx))
            print("Unexpected error:", sys.exc_info()[0])
            print(e)
            raise

        return {'task_label': task_label,'init':init, 'last_label': last_label, 'avg_label': avg_label}

    def _prepare_init(self, idx):
        init_md = self.init_md[idx]
        return init_md

    def _prepare_task_label(self, idx):
        task_label_md = self.task_label_md[idx]
        return task_label_md

    def _get_last_label(self, idx):
        last_label_md = self.last_label_md[idx]
        return last_label_md

    def _get_avg_label(self, idx):
        avg_label_md = self.avg_label_md[idx]
        return avg_label_md

if __name__ == '__main__':
    data_path ='/data/zzhao/uav_regression/main_test/data_tasks.npy'
    init_path = '/data/zzhao/uav_regression/main_test/data_init_density.npy'
    last_label_path = '/data/zzhao/uav_regression/main_test/training_label_density.npy'

    all_dataset = UAVDatasetTuple(task_path=data_path, init_path=init_path, last_label_path=last_label_path)
    sample = all_dataset[0]
    print(sample['task'].shape)
    count = 0
