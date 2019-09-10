import torch
import torch.nn as nn
import torchvision

class UAVModel(nn.Module):
    def __init__(self):
        super(UAVModel, self).__init__()

        # Common model declaration
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=1)

        # CNN embedding model declaration
        self.cnn_embedding_conv1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=1)
        self.cnn_embedding_conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1)
        self.cnn_embedding_conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=1)
        self.cnn_embedding_conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=1)
        self.cnn_embedding_conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1)
        self.cnn_embedding_conv6 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1)

        self.cnn_embedding_bn1 = nn.BatchNorm1d(8)
        self.cnn_embedding_bn2 = nn.BatchNorm1d(16)
        self.cnn_embedding_bn3 = nn.BatchNorm1d(32)
        self.cnn_embedding_bn4 = nn.BatchNorm1d(64)

        # Model declaration for the sumNet
        self.sum_conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1)
        self.sum_conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1)

        pass

    def init_parameters(self):
        pass

    # PointNet for the feature extraction
    # Reference: https://arxiv.org/pdf/1612.00593.pdf
    def _pNet_forward(self, x):
        pass

    # RouteNet for the feature extraction
    # Reference: https://research.nvidia.com/sites/default/files/pubs/2018-11_RouteNet%3A-routability-prediction/a80-xie.pdf
    def _rNet_froward(self, x):
        pass

    # Naive CNN for the feature extraction
    def _cnn_forward(self, x):
        # First cnn block
        x = self.cnn_embedding_conv1(x)
        x = self.cnn_embedding_conv2(x)
        x = self.cnn_embedding_bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        # Second cnn block
        x = self.cnn_embedding_conv3(x)
        x = self.cnn_embedding_conv4(x)
        x = self.cnn_embedding_bn2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        # Third cnn block
        x = self.cnn_embedding_conv5(x)
        x = self.cnn_embedding_bn3(x)
        x = self.relu(x)
        x = self.max_pool(x)

        # Fourth cnn block
        x = self.cnn_embedding_conv6(x)
        x = self.cnn_embedding_bn4(x)
        x = self.relu(x)

        # Flatten
        x = torch.flatten(x, 1)
        return x

    # LSTM for the trajectory sequence prediction
    def _lstm_froward(self, x):
        pass

    # Summarize the trajectory sequence for the final density prediction
    def _sumNet_forward(self, x):
        x = self.sum_conv1(x)
        x = self.sum_conv2(x)
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        #ToDo: add 2D transpose layer to upsample the output
        pass

    def forward(self):
        pass