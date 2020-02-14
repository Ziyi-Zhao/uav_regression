from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import torch


class UAVDatasetTuple(Dataset):
    def __init__(self, task_path, task_label_path, init_path, label_path):
        self.task_path = task_path
        self.task_label_path = task_label_path
        self.init_path = init_path
        self.label_path = label_path
        self.label_md = []
        self.init_md = []
        self.task_md = []
        self._get_tuple()

    def __len__(self):
        return len(self.label_md)

    def _get_tuple(self):
        self.task_md = np.load(self.task_path).astype(float)
        self.task_label_md = np.load(self.task_label_path).astype(float)
        self.init_md = np.load(self.init_path).astype(float)
        self.label_md = np.load(self.label_path).astype(float)
        #assert len(self.task_md) == len(self.label_md), "not identical"

    def __getitem__(self, idx):
        try:
            task = self._prepare_task(idx)
            task_label = self._prepare_task_label(idx)
            init = self._prepare_init(idx)
            label = self._get_label(idx)

            init = np.expand_dims(init, axis=0)
        except Exception as e:
            print('error encountered while loading {}'.format(idx))
            print("Unexpected error:", sys.exc_info()[0])
            print(e)
            raise

        return {'task': task, 'task_label': task_label,'init':init, 'label': label}

    def _prepare_init(self, idx):
        init_md = self.init_md[idx]
        return init_md

    def _prepare_task_label(self, idx):
        task_label_md = self.task_label_md[idx]
        return task_label_md


    def _prepare_task(self, idx):
        #task_coordinate = self.task_md[idx]
        input = self.task_md[idx]
        #print("input shape", input.shape)
        task_md = torch.zeros([input.shape[0],15, 100, 100])
        for i in range(input.shape[0]):
            # if i < 30:
            #     continue
            for j in range(15):
                x1 = int(input[i][j][0])
                y1 = int(input[i][j][1])
                x2 = int(input[i][j][2])
                y2 = int(input[i][j][3])
                if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                    continue
                else:
                    task_md[i][j][x1][y1] = 1.00
                    task_md[i][j][x2][y2] = 1.00
            # if i > 30:
            #     return task_md.reshape(input.shape[0],15,10000)
        return task_md.reshape(input.shape[0],15,10000)

    def _get_label(self, idx):
        label_md = self.label_md[idx].reshape(100,100)
        return label_md

    def get_class_count(self):
        total = len(self.label_md) * self.label_md[0].shape[0] * self.label_md[0].shape[1]
        positive_class = 0
        for label in self.label_md:
            positive_class += np.sum(label)
        print("The number of positive image pair is:", positive_class)
        print("The number of negative image pair is:", total - positive_class)
        positive_ratio = positive_class / total
        negative_ratio = (total - positive_class) / total

        return positive_ratio, negative_ratio

if __name__ == '__main__':
    data_path ='/data/zzhao/uav_regression/main_test/data_tasks.npy'
    init_path = '/data/zzhao/uav_regression/main_test/data_init_density.npy'
    label_path = '/data/zzhao/uav_regression/main_test/training_label_density.npy'

    all_dataset = UAVDatasetTuple(task_path=data_path, init_path=init_path, label_path=label_path)
    sample = all_dataset[0]
    print(sample['task'].shape)
    count = 0
