import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class STN4d(nn.Module):
    def __init__(self):
        super(STN4d, self).__init__()
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 16)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]).astype(np.float32))).view(1,16).repeat(batchsize,1)
        # print(iden)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 4, 4)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN4d()
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


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

        self.cnn_embedding_bn1 = nn.BatchNorm2d(8)
        self.cnn_embedding_bn2 = nn.BatchNorm2d(16)
        self.cnn_embedding_bn3 = nn.BatchNorm2d(32)
        self.cnn_embedding_bn4 = nn.BatchNorm2d(64)

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

    def init_parameters(self):
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
        pointfeat = PointNetfeat(global_feat=True)
        out, _, _ = pointfeat(x)
        # print('global feat', out.size())
        return out

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

    def forward(self, x, structure = "org"):
        # Note: the order is (seq, batch, feature) in pytorch
        # (batch, seq, w, w) -> (seq, batch, w, w)
        x = x.permute(1, 0, 2, 3)
        embedding_list = list()
        for time_sample in x:
            x_embedding = self._cnn_forward(time_sample)
            embedding_list.append(x_embedding)
        x_embedding = torch.stack(embedding_list, dim=0)

        x_lstm = self._lstm_froward(x_embedding)

        # (seq, batch, w, w) -> (batch, seq, w, w)
        x_lstm = x_lstm.permute(1, 0, 2, 3)

        x_sum = self._sumNet_forward(x_lstm)
        return x_lstm, x_sum

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    # Note: (batch, channels, amount)
    print("spatial transformer testing")
    sim_data = torch.rand(32, 4, 2500)
    trans = STN4d()
    out = trans(sim_data)
    print("trans output", out.shape)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))
    print("spatial transformer finished")
#######################################################
    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())




