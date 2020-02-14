import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Correlation:
    def __init__(self):
        # save correlation in data file
        # path = os.path.split(path)[0]
        self.best = 0.0
    
    def bn(self, a, maxVal, minVal):
        for i in range(len(a)):
            a[i] = (a[i] - minVal) / (maxVal - minVal)
        return a

    def corrcoef(self, prediction, label, path = '.', name = "correlation.png"):
        self.path = os.path.join(path, name)

        x = prediction.reshape((-1))
        y_true = label.reshape((-1))
        r = np.corrcoef(x, y_true)
        r = r[0,1]
        # print('correlation coefficient : \n', r)
        if r > self.best:
            self.best = r
        r = self.best
        y_pre = x*r
        
        maxVal = 0.0
        maxVal = max(maxVal, np.max(x))
        maxVal = max(maxVal, np.max(y_pre))
        maxVal = max(maxVal, np.max(y_true))
        minVal = 1.0
        minVal = min(minVal, np.min(x))
        minVal = min(minVal, np.min(y_pre))
        minVal = min(minVal, np.min(y_true))
        self.bn(x, maxVal, minVal)
        self.bn(y_pre, maxVal, minVal)
        self.bn(y_true, maxVal, minVal)

        plt.figure()
        palette = plt.get_cmap('Set1')
        plt.plot(x, y_pre, color='blue',label="correlation", linewidth=1, zorder=1)
        plt.scatter(x, y_true, color=palette(0), label="actual", s=0.1, zorder=-1)
        plt.xlabel("prediction")
        plt.ylabel("label")
        plt.title('Correlation Coefficient = {0}'.format(np.round(r,3)))
        plt.legend()
        
        plt.savefig(self.path)
        plt.close()
        return r


# if __name__ == '__main__':
#     prediction = np.load('../UAV_POSTPROCESS/tmpdata/lstmdata.npy')
#     label = np.load('../UAV_POSTPROCESS/tmpdata/y_test.npy')
#     c = Correlation();
#     c.corrcoef(prediction, label);