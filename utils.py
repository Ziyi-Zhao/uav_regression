import os
import torch
import pickle
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from inspect import signature
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score


def read_pickle(filename):
    infile = open(filename, 'rb')
    data = pickle.load(infile)
    infile.close()
    return data


def dump_pickle(filename, data):
    outfile = open(filename, "wb")
    pickle.dump(data, filename)
    outfile.close()


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# precision = TP / (TP + FP)
# recall = TP / (TP + FN)
def calculate_precision_recall(prediction, label, mode, batch_idx, epoch):
    prediction = prediction.reshape(prediction.shape[0] * prediction.shape[1] * prediction.shape[2] * prediction.shape[3])
    label = label.reshape(label.shape[0] * label.shape[1] * label.shape[2] * label.shape[3])
    average_precision = average_precision_score(label, prediction)
    precision, recall, thresholds = precision_recall_curve(label, prediction)
    idx = find_nearest(thresholds, 0.5)

    step_kwargs = ({'step': 'post'}if 'step' in signature(plt.fill_between).parameters else {})
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig('/home/zzhao/data/uav_regression/regression_process/precision_recall_curve/' + str(mode) + '/precision_recall_curve_epoch_' + str(epoch) + 'batch_' + str (batch_idx) + '.png')
    plt.close()

    return precision[idx], recall[idx]

# TPR = TP / (TP + FN)
# FPR = FP / (FP + TN)
def draw_roc_curve(prediction, label, mode, epoch, batch_idx):
    prediction = prediction.reshape(prediction.shape[0] * prediction.shape[1] *  prediction.shape[2] *  prediction.shape[3])
    label = label.reshape(label.shape[0] * label.shape[1] * label.shape[2] * label.shape[3])

    fpr, tpr, thresholds = roc_curve(label, prediction, pos_label=1)
    auroc = roc_auc_score(label, prediction)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auroc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('/home/zzhao/data/uav_regression/regression_process/roc_curve/' + str(mode) + '/roc_curve_epoch_' + str(epoch) + '_batch_' + str (batch_idx) + '.png')
    plt.close()

    return auroc

def visualize_lstm_testing_result(prediction, label, batch_id, epoch):
    assert prediction.shape[0] == label.shape[0], "prediction size and label size is not identical"
    if not os.path.exists("/home/zzhao/data/uav_regression/testing_result/epoch_" + str(epoch)):
        os.mkdir("/home/zzhao/data/uav_regression/testing_result/epoch_" + str(epoch))
    if not os.path.exists("/home/zzhao/data/uav_regression/testing_result/epoch_" + str(epoch) + "/lstm"):
        os.mkdir("/home/zzhao/data/uav_regression/testing_result/epoch_" + str(epoch) + "/lstm")
    for idx, _ in enumerate(prediction):
        prediction_lstm = torch.sum(prediction[idx], dim=0)
        label_lstm = torch.sum(label[idx], dim=0)
        prediction_lstm = (prediction_lstm - torch.min(prediction_lstm)) / (torch.max(prediction_lstm) - torch.min(prediction_lstm))
        # prediction_lstm[prediction_lstm < 0.5] = 0
        label_lstm = (label_lstm - torch.min(label_lstm)) / (torch.max(label_lstm) - torch.min(label_lstm))
        torchvision.utils.save_image(prediction_lstm, "/home/zzhao/data/uav_regression/testing_result/epoch_" + str(epoch) + "/lstm/" + str(idx + batch_id * 32) +  "_prediction.png")
        torchvision.utils.save_image(label_lstm, "/home/zzhao/data/uav_regression/testing_result/epoch_" + str(epoch) + "/lstm/" + str(idx + batch_id * 32) + "_label.png")


def visualize_sum_testing_result(prediction, label, batch_id, epoch):
    assert prediction.shape[0] == label.shape[0], "prediction size and label size is not identical"
    if not os.path.exists("/home/zzhao/data/uav_regression/testing_result/epoch_" + str(epoch)):
        os.mkdir("/home/zzhao/data/uav_regression/testing_result/epoch_" + str(epoch))
    if not os.path.exists("/home/zzhao/data/uav_regression/testing_result/epoch_" + str(epoch) + "/sum"):
        os.mkdir("/home/zzhao/data/uav_regression/testing_result/epoch_" + str(epoch) + "/sum")
    for idx, _ in enumerate(prediction):
        prediction_output = prediction[idx]#.cpu().detach().numpy()
        label_output = label[idx]#.cpu().detach().numpy()
        prediction_output[prediction_output < 0.30] = 0
        # output[output >= 0.50] = 1
        # plt.imshow(prediction_output)
        # plt.savefig("/home/zzhao/data/uav_regression/testing_result/epoch_" + str(epoch) + "/sum" + "/" + str(idx + batch_id * 32) +  "_prediction.png")
        # plt.imshow(label_output)
        # plt.savefig("/home/zzhao/data/uav_regression/testing_result/epoch_" + str(epoch) + "/sum" + "/" + str(idx + batch_id * 32) +  "_label.png")
        # plt.close()
        torchvision.utils.save_image(prediction_output, "/home/zzhao/data/uav_regression/testing_result/epoch_" + str(epoch) + "/sum" + "/" + str(idx + batch_id * 32) +  "_prediction.png")
        torchvision.utils.save_image(label_output, "/home/zzhao/data/uav_regression/testing_result/epoch_" + str(epoch) + "/sum" + "/" + str(idx + batch_id * 32) +  "_label.png")

