import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.metrics import confusion_matrix

thresholds = np.arange(0.0, 1.0, 0.01)


def rates(label, prediction, percentage):
    # calculate true positive rates and false positive rates
    tpr = []
    fpr = []

    y_true = deepcopy(label)

    deleteIndex = np.where(y_true == 0)
    y_true = np.delete(y_true, deleteIndex, None)
    prediction = np.delete(prediction, deleteIndex, None)

    # print("label max: {}".format(np.max(y_true[y_true>0])))
    # print("label min: {}".format(np.min(y_true[y_true>0])))
    y_sort = np.sort(y_true[y_true > 0])
    th = np.array_split(y_sort, percentage)[-1][0]
    y_true[y_true < th] = 0
    y_true[y_true >= th] = 1
    print("label with percentage {0} threshold: {1}".format(1 - 1 / percentage, th))

    for th in thresholds:
        y_pred = deepcopy(prediction)
        y_pred[y_pred < th] = 0
        y_pred[y_pred >= th] = 1

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        # print(tn, fp, fn, tp)
        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))

    return tpr, fpr


def auc(types, percentages, data, auc_path, epoch, segment = 0):

    for p in percentages:
        tprs = []
        fprs = []
        for label_path, pred_path in data:
            label = label_path
            prediction = pred_path
            tpr, fpr = rates(label.reshape(-1), prediction.reshape(-1), p)
            tprs.append(tpr)
            fprs.append(fpr)

        plt.figure(figsize=(5, 5))
        palette = plt.get_cmap('Set1')
        for i, t in zip(range(len(types)), types):
            roc_auc = np.trapz(fprs[i], tprs[i])
            plt.plot(fprs[i], tprs[i], color=palette(i + 1), label="auc: {0}".format(round(1 + roc_auc, 3)))
            print("auc: {0}".format(1 + roc_auc))
        print()
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("AUROC")
        plt.legend()
        plt.savefig(auc_path + "/auroc_{0}_{1}_p_{2}.png".format(epoch, segment, (1 - 1 / p)))


if __name__ == "__main__":
    types = ['flow', 'ones', 'probability']
    percentages = [2, 4, 10, 100]  # 50% 75% 90% 99%
    data = [
        ["data/sixSeg/label_flow.npy", "data/sixSeg/pred_flow.npy"],
        ["data/sixSeg/label_probability.npy", "data/sixSeg/pred_probability.npy"],
        ["data/sixSeg/label_ones.npy", "data/sixSeg/pred_ones.npy"],
    ]
    auc(types, percentages, data)