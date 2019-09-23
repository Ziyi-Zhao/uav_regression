import numpy as np
import torch
import torch.nn as nn
import torchvision

from torch.autograd import Variable


class UAVModel(nn.Module):
    def __init__(self, structure="basic_cnn"):
        super(UAVModel, self).__init__()

        self.structure = structure

        # Common model declaration
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if self.structure == "basic_cnn":
            # CNN embedding model declaration
            self.cnn_embedding_conv1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1)
            self.cnn_embedding_conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1)
            self.cnn_embedding_conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
            self.cnn_embedding_conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)

            self.cnn_embedding_bn1 = nn.BatchNorm2d(8)
            self.cnn_embedding_bn2 = nn.BatchNorm2d(16)
            self.cnn_embedding_bn3 = nn.BatchNorm2d(32)
            self.cnn_embedding_bn4 = nn.BatchNorm2d(64)

            # lstm model declaration
            # Note: the order is (seq, batch, feature) in pytorch
            self.lstm = nn.LSTM(input_size=576, hidden_size=512, num_layers=2)

            self.lstm_fc1 = nn.Linear(in_features=512, out_features=1024)

            self.lstm_bn1 = nn.BatchNorm1d(1024)
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

            # lstm model declaration
            # Note: the order is (seq, batch, feature) in pytorch
            self.lstm = nn.LSTM(input_size=1024, hidden_size=512, num_layers=2)

            self.lstm_fc1 = nn.Linear(in_features=512, out_features=1024)

            self.lstm_bn1 = nn.BatchNorm1d(1024)
        elif self.structure == "rnet":
            # conv
            self.rnet_conv1 = torch.nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, padding=2)
            self.rnet_conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.rnet_conv3 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
            self.rnet_conv4 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
            self.rnet_conv5 = torch.nn.Conv2d(in_channels=48, out_channels=16, kernel_size=1)
            self.rnet_conv6 = torch.nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1)

            # deconv
            self.rnet_transpose1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
            self.rnet_transpose2 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=2, stride=2)

            # bn
            self.rnet_bn1 = nn.BatchNorm2d(32)
            self.rnet_bn2 = nn.BatchNorm2d(64)
            self.rnet_bn3 = nn.BatchNorm2d(32)
            self.rnet_bn4 = nn.BatchNorm2d(32)
            self.rnet_bn5 = nn.BatchNorm2d(16)
            self.rnet_bn6 = nn.BatchNorm2d(16)
            self.rnet_bn7 = nn.BatchNorm2d(4)
            self.rnet_bn8 = nn.BatchNorm2d(1)

        # sumNet model declaration
        self.sum_conv1 = nn.Conv2d(in_channels=60, out_channels=32, kernel_size=3, stride=1)
        self.sum_conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1)
        self.sum_transpose1 = nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=3, stride=1)
        self.sum_transpose2 = nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=3, stride=1)

        self.sum_fc1 = nn.Linear(in_features=576, out_features=3136)

        self.sum_bn1 = nn.BatchNorm2d(32)
        self.sum_bn2 = nn.BatchNorm2d(16)
        self.sum_bn3 = nn.BatchNorm2d(2)
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
            # initialize the parameters within the rNet model
            torch.nn.init.normal_(self.rnet_conv1.weight, std=0.1)
            torch.nn.init.constant_(self.rnet_conv1.bias, val=0.0)
            torch.nn.init.normal_(self.rnet_conv2.weight, std=0.1)
            torch.nn.init.constant_(self.rnet_conv2.bias, val=0.0)
            torch.nn.init.normal_(self.rnet_conv3.weight, std=0.1)
            torch.nn.init.constant_(self.rnet_conv3.bias, val=0.0)
            torch.nn.init.normal_(self.rnet_conv4.weight, std=0.1)
            torch.nn.init.constant_(self.rnet_conv4.bias, val=0.0)
            torch.nn.init.normal_(self.rnet_conv5.weight, std=0.1)
            torch.nn.init.constant_(self.rnet_conv5.bias, val=0.0)
            torch.nn.init.normal_(self.rnet_conv6.weight, std=0.1)
            torch.nn.init.constant_(self.rnet_conv6.bias, val=0.0)

            torch.nn.init.normal_(self.rnet_transpose1.weight, std=0.1)
            torch.nn.init.constant_(self.rnet_transpose1.bias, val=0.0)
            torch.nn.init.normal_(self.rnet_transpose2.weight, std=0.1)
            torch.nn.init.constant_(self.rnet_transpose2.bias, val=0.0)

        if self.structure == 'basic_cnn' or self.structure == 'pnet':
            # initialize the parameters within the lstm model
            torch.nn.init.normal_(self.lstm.weight_hh_l0, std=0.1)
            torch.nn.init.constant_(self.lstm.bias_hh_l0, val=0.0)
            torch.nn.init.normal_(self.lstm.weight_ih_l0, std=0.1)
            torch.nn.init.constant_(self.lstm.bias_ih_l0, val=0.0)
            torch.nn.init.normal_(self.lstm.weight_hh_l1, std=0.1)
            torch.nn.init.constant_(self.lstm.bias_hh_l1, val=0.0)
            torch.nn.init.normal_(self.lstm.weight_ih_l1, std=0.1)
            torch.nn.init.constant_(self.lstm.bias_ih_l1, val=0.0)
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
        # 1->32
        x = self.rnet_conv1(x)
        x = self.rnet_bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x_shortcut = x
        # 32->64
        x = self.rnet_conv2(x)
        x = self.rnet_bn2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        # 64->32
        x = self.rnet_conv3(x)
        x = self.rnet_bn3(x)
        x = self.relu(x)
        # 32->32
        x = self.rnet_conv4(x)
        x = self.rnet_bn4(x)
        x = self.relu(x)
        # 32->16
        x = self.rnet_transpose1(x)
        x = self.rnet_bn5(x)
        x = self.relu(x)

        # shortcut path
        x = torch.cat([x, x_shortcut], dim=1)

        # 16 + 32->16
        x = self.rnet_conv5(x)
        x = self.rnet_bn6(x)
        x = self.relu(x)
        # 16->4
        x = self.rnet_transpose2(x)
        x = self.rnet_bn7(x)
        x = self.relu(x)
        # 4->1
        x = self.rnet_conv6(x)
        x = self.rnet_bn8(x)
        x = self.relu(x)
        # x = torch.sigmoid(x)
        return x

    # Basic CNN model for the feature extraction
    def _cnn_forward(self, x):
        # First cnn block
        x = self.cnn_embedding_conv1(x)
        x = self.cnn_embedding_bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        # Second cnn block
        x = self.cnn_embedding_conv2(x)
        x = self.cnn_embedding_bn2(x)
        x = self.relu(x)
        # x = self.max_pool(x)

        # Third cnn block
        x = self.cnn_embedding_conv3(x)
        x = self.cnn_embedding_bn3(x)
        x = self.relu(x)
        x = self.max_pool(x)

        # Fourth cnn block
        x = self.cnn_embedding_conv4(x)
        x = self.cnn_embedding_bn4(x)
        x = self.relu(x)

        # Flatten
        x = torch.flatten(x, 1)
        return x

    # LSTM for the trajectory sequence prediction
    def _lstm_froward(self, x):
        x_out, x_hidden = self.lstm(x)

        trajectory_list = list()
        for time_sample in x_out:
            time_sample = self.lstm_fc1(time_sample)
            time_sample = self.lstm_bn1(time_sample)
            time_sample = torch.sigmoid(time_sample)
            trajectory_list.append(time_sample)
        x = torch.stack(trajectory_list, dim=0)

        x = x.view(x.shape[0], x.shape[1], 32, 32)
        return x

    # Summarize the trajectory sequence to predict the final density
    def _sumNet_forward(self, x):
        # Extract features from the lstm outputs
        x = self.sum_conv1(x)
        x = self.sum_bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.sum_conv2(x)
        x = self.sum_bn2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = self.sum_fc1(x)
        x = x.view(-1, 4, 28, 28)

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
        if self.structure == "basic_cnn":
            # Note: the order is (seq, batch, feature) in pytorch
            # (batch, seq, w, w, c) -> (seq, batch, c, w, w)
            x = x.permute(1, 0, 4, 2, 3)

            embedding_list = list()
            for time_sample in x:
                x_embedding = self._cnn_forward(time_sample)

                embedding_list.append(x_embedding)
            x_embedding = torch.stack(embedding_list, dim=0)

            x_lstm = self._lstm_froward(x_embedding)

            # (seq, batch, w, w) -> (batch, seq, w, w)
            x_lstm = x_lstm.permute(1, 0, 2, 3)

            # x_sum = self._sumNet_forward(x_lstm)
            # x_sum = torch.squeeze(x_sum)
            return x_lstm  # , x_sum
        elif self.structure == "pnet":
            # (batch, seq, w, c) -> (seq, batch, c, w)
            x = x.permute(1, 0, 3, 2)

            embedding_list = list()
            for time_sample in x:
                x_embedding = self._pNet_forward(time_sample)

                embedding_list.append(x_embedding)
            x_embedding = torch.stack(embedding_list, dim=0)

            x_lstm = self._lstm_froward(x_embedding)

            # (seq, batch, w, w) -> (batch, seq, w, w)
            x_lstm = x_lstm.permute(1, 0, 2, 3)

            # x_sum = self._sumNet_forward(x_lstm)
            # x_sum = torch.squeeze(x_sum)
            return x_lstm  # , x_sum
        elif self.structure == "rnet":
            x = x.permute(0, 3, 1, 2)

            x = self._rNet_froward(x)
            return x.squeeze()
