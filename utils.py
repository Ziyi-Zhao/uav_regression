import pickle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve


def read_pickle(filename):
    infile = open(filename, 'rb')
    data = pickle.load(infile)
    infile.close()
    return data


def dump_pickle(filename, data):
    outfile = open(filename, "wb")
    pickle.dump(data, filename)
    outfile.close()

# precision = TP / (TP + FP)
# recall = RP / (TP + FN)
def calculate_precision_recall(prediction, label):
    prediction = prediction.reshape(prediction.shape[0] * prediction.shape[1], prediction.shape[2], prediction.shape[3])
    label = label.reshape(label.shape[0] * label.shape[1], label.shape[2], label.shape[3])
    precision, recall, thresholds = precision_recall_curve(label, prediction)
    # ToDo: 1. draw the precision-recall curve. 2. sample recall @ threshold 0.5
    return precision, recall

# TPR = TP / (TP + FN)
# FPR = FP / (FP + TN)
def draw_roc_curve(prediction, label, mode, epoch, batch):
    prediction = prediction.reshape(prediction.shape[0] * prediction.shape[1], prediction.shape[2], prediction.shape[3])
    label = label.reshape(label.shape[0] * label.shape[1], label.shape[2], label.shape[3])
    # ToDo: 1. draw the roc curve
    fpr, tpr, thresholds = roc_curve(label, prediction, pos_label=2)
    auroc = roc_auc_score(label, prediction)
    return auroc