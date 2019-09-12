import numpy as np
import torch
import torch.nn as nn
import torchvision

from torch.autograd import Variable


class UAVModel(nn.Module):
    def __init__(self, structure="basic_cnn"):
        super(UAVModel, self).__init__()

        # Common model declaration
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=1)

        if self.structure == "basic_cnn":
            # CNN embedding model declaration
            self.cnn_embedding_conv1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=1)
            self.cnn_embedding_conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1)
            self.cnn_embedding_conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=1)
            self.cnn_embedding_conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=1)
            self.cnn_embedding_conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1)
            self.cnn_embedding_conv6 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1)

            self.cnn_embedding_bn1 = nn.BatchNorm2d(8)
            self.cnn_embedding_bn2 = nn.BatchNorm2d(16)
            self.cnn_embedding_bn3 = nn.BatchNorm2d(32)
            self.cnn_embedding_bn4 = nn.BatchNorm2d(64)
        elif self.structure == "pnet":
            # stn to support the pNet
            self.stn_conv1 = nn.Conv1d(4, 64, 1)
            self.stn_conv2 = nn.Conv1d(64, 128, 1)
            self.stn_conv3 = nn.Conv1d(128, 1024, 1)

            self.stn_fc1 = nn.Linear(1024, 512)
            self.stn_fc2 = nn.Linear(512, 256)
            self.stn_fc3 = nn.Linear(256, 16)

            self.stn_bn1 = nn.BatchNorm1d(64)
            self.stn_bn2 = nn.BatchNorm1d(128)
            self.stn_bn3 = nn.BatchNorm1d(1024)
            self.stn_bn4 = nn.BatchNorm1d(512)
            self.stn_bn5 = nn.BatchNorm1d(256)

            # pNet embedding model declaration
            self.pNet_conv1 = torch.nn.Conv1d(4, 64, 1)
            self.pNet_conv2 = torch.nn.Conv1d(64, 128, 1)
            self.pNet_conv3 = torch.nn.Conv1d(128, 1024, 1)
            self.pNet_bn1 = nn.BatchNorm1d(64)
            self.pNet_bn2 = nn.BatchNorm1d(128)
            self.pNet_bn3 = nn.BatchNorm1d(1024)
        elif self.structure == "rnet":
            pass

        # lstm model declaration
        # Note: the order is (seq, batch, feature) in pytorch
        self.lstm = nn.LSTM(input_size=1600, hidden_size=512, num_layers=2)

        self.lstm_fc1 = nn.Linear(in_features=512, out_features=1024)

        self.lstm_bn1 = nn.BatchNorm1d(1024)

        # sumNet model declaration
        self.sum_conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1)
        self.sum_conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1)
        self.sum_transpose1 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=1)
        self.sum_transpose2 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=1)

        self.sum_fc1 = nn.Linear(in_features=2048, out_features=1024)

        self.sum_bn1 = nn.BatchNorm2d(8)
        self.sum_bn2 = nn.BatchNorm2d(16)
        self.sum_bn3 = nn.BatchNorm2d(8)
        self.sum_bn4 = nn.BatchNorm2d(1)

        self.structure = structure

        self.init_parameters()

    def init_parameters(self):
        if self.structure == "basic_cnn":
            # initialize the parameters within the CNN embedding model
            torch.nn.init.normal_(self.cnn_embedding_conv1.weight, std=0.1)
            torch.nn.init.constant_(self.cnn_embedding_conv1.bias, val=0.0)
            torch.nn.init.normal_(self.cnn_embedding_conv2.weight, std=0.1)
            torch.nn.init.constant_(self.cnn_embedding_conv2.bias, val=0.0)
            torch.nn.init.normal_(self.cnn_embedding_conv3.weight, std=0.1)
            torch.nn.init.constant_(self.cnn_embedding_conv3.bias, val=0.0)
            torch.nn.init.normal_(self.cnn_embedding_conv4.weight, std=0.1)
            torch.nn.init.constant_(self.cnn_embedding_conv4.bias, val=0.0)
            torch.nn.init.normal_(self.cnn_embedding_conv5.weight, std=0.1)
            torch.nn.init.constant_(self.cnn_embedding_conv5.bias, val=0.0)
            torch.nn.init.normal_(self.cnn_embedding_conv6.weight, std=0.1)
            torch.nn.init.constant_(self.cnn_embedding_conv6.bias, val=0.0)
        elif self.structure == "pnet":
            # initialize the parameters within the STN model
            torch.nn.init.normal_(self.stn_conv1.weight, std=0.1)
            torch.nn.init.constant_(self.stn_conv1.bias, val=0.0)
            torch.nn.init.normal_(self.stn_conv2.weight, std=0.1)
            torch.nn.init.constant_(self.stn_conv2.bias, val=0.0)
            torch.nn.init.normal_(self.stn_conv3.weight, std=0.1)
            torch.nn.init.constant_(self.stn_conv3.bias, val=0.0)

            torch.nn.init.normal_(self.stn_fc1.weight, std=0.1)
            torch.nn.init.constant_(self.stn_fc1.bias, val=0.0)
            torch.nn.init.normal_(self.stn_fc2.weight, std=0.1)
            torch.nn.init.constant_(self.stn_fc2.bias, val=0.0)
            torch.nn.init.normal_(self.stn_fc3.weight, std=0.1)
            torch.nn.init.constant_(self.stn_fc3.bias, val=0.0)

            # initialize the parameters within the PointNet model
            torch.nn.init.normal_(self.pNet_conv1.weight, std=0.1)
            torch.nn.init.constant_(self.pNet_conv1.bias, val=0.0)
            torch.nn.init.normal_(self.pNet_conv2.weight, std=0.1)
            torch.nn.init.constant_(self.pNet_conv2.bias, val=0.0)
            torch.nn.init.normal_(self.pNet_conv3.weight, std=0.1)
            torch.nn.init.constant_(self.pNet_conv3.bias, val=0.0)
        elif self.structure == "rnet":
            pass

        # initialize the parameters within the lstm model
        torch.nn.init.normal_(self.lstm.weight, std=0.1)
        torch.nn.init.constant_(self.lstm.bias, val=0.0)
        torch.nn.init.normal_(self.lstm_fc1.weight, std=0.1)
        torch.nn.init.constant_(self.lstm_fc1.bias, val=0.0)

        # initialize the parameters within the sumNet model
        torch.nn.init.normal_(self.sum_conv1.weight, std=0.1)
        torch.nn.init.constant_(self.sum_conv1.bias, val=0.0)
        torch.nn.init.normal_(self.sum_conv2.weight, std=0.1)
        torch.nn.init.constant_(self.sum_conv2.bias, val=0.0)

        torch.nn.init.normal_(self.sum_transpose1.weight, std=0.1)
        torch.nn.init.constant_(self.sum_transpose1.bias, val=0.0)
        torch.nn.init.normal_(self.sum_transpose2.weight, std=0.1)
        torch.nn.init.constant_(self.sum_transpose2.bias, val=0.0)

        torch.nn.init.normal_(self.sum_fc1.weight, std=0.1)
        torch.nn.init.constant_(self.sum_fc1.bias, val=0.0)

    # PointNet for the feature extraction
    # Reference: https://arxiv.org/pdf/1612.00593.pdf
    def _pNet_forward(self, x):
        # ToDo: handle the input data variance
        trans = self._stn_forward(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = self.relu(self.pNet_bn1(self.pNet_conv1(x)))

        x = self.relu(self.pNet_bn2(self.pNet_conv2(x)))
        x = self.pNet_bn3(self.pNet_conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x

    # STN to support the pNet
    def _stn_forward(self, x):
        batch_size = x.size()[0]
        x = self.relu(self.stn_bn1(self.stn_conv1(x)))
        x = self.relu(self.stn_bn2(self.stn_conv2(x)))
        x = self.relu(self.stn_bn3(self.stn_conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.relu(self.stn_bn4(self.stn_fc1(x)))
        x = self.relu(self.stn_bn5(self.stn_fc2(x)))
        x = self.stn_fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).astype(np.float32))).view(1, 16).repeat(batch_size, 1)

        if x.is_cuda:
            iden = iden.cuda()

        x = x + iden
        x = x.view(-1, 4, 4)
        return x

    # RouteNet for the feature extraction
    # Reference: https://research.nvidia.com/sites/default/files/pubs/2018-11_RouteNet%3A-routability-prediction/a80-xie.pdf
    def _rNet_froward(self, x):
        pass

    # Basic CNN model for the feature extraction
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
        x = self.lstm(x)

        trajectory_list = list()
        for time_sample in x:
            time_sample = self.lstm_fc1(time_sample)
            x = self.lstm_bn1(time_sample)
            time_sample = torch.sigmoid(time_sample)
            trajectory_list.append(time_sample)
        x = torch.stack(trajectory_list, dim=0)

        x = x = x.view(x.shape[0], x.shape[1], 32, 32)
        return x

    # Summarize the trajectory sequence to predict the final density
    def _sumNet_forward(self, x):
        # Extract features from the lstm outputs
        x = self.sum_conv1(x)
        x = self.sum_bn1(x)
        x = self.relu(x)
        x = self.sum_conv2(x)
        x = self.sum_bn2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = self.sum_fc1(x)
        x = x.view(-1, 8*8*8)

        # First 2d transpose block
        x = self.sum_transpose1(x)
        x = self.sum_bn3(x)
        x = self.relu(x)

        # Second 2d transpose block
        x = self.sum_transpose2(x)
        x = self.sum_bn4(x)

        # Pixel-wise regression
        x = torch.sigmoid(x)
        return  x

    def forward(self, x):
        # Note: the order is (seq, batch, feature) in pytorch
        # (batch, seq, w, w) -> (seq, batch, w, w)
        x = x.permute(1, 0, 2, 3)

        embedding_list = list()
        for time_sample in x:
            if self.structure == "basic_cnn":
                x_embedding = self._cnn_forward(time_sample)
            elif self.structure == "pnet":
                x_embedding = self._pNet_forward(time_sample)
            elif self.structure == "rnet":
                pass

            embedding_list.append(x_embedding)
        x_embedding = torch.stack(embedding_list, dim=0)

        x_lstm = self._lstm_froward(x_embedding)

        # (seq, batch, w, w) -> (batch, seq, w, w)
        x_lstm = x_lstm.permute(1, 0, 2, 3)

        x_sum = self._sumNet_forward(x_lstm)
        return x_lstm, x_sum