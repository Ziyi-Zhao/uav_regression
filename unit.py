import torch
import torch.nn as nn

class Unit(nn.Module):
    def __init__(self, channel=10):
        super(Unit, self).__init__()

        self.out1 = 64
        self.out2 = 128
        self.out3 = 256

        self.relu = nn.ReLU(inplace=True)

        # 3D Conv Operation
        # self.sub_conv1 = nn.Conv3d(in_channels=1, out_channels=self.out1, kernel_size=(4, 4, 2), stride=(2, 2, 2))
        # self.sub_conv2 = nn.Conv3d(in_channels=self.out1, out_channels=self.out2, kernel_size=(2,2,2), stride=(1, 1, 1))
        # self.sub_conv3 = nn.Conv3d(in_channels=self.out2, out_channels=self.out3, kernel_size=(2,2,2), stride=(1, 1, 1))

        # 2D Conv Operation
        self.sub_conv1 = nn.Conv2d(in_channels=channel, out_channels=self.out1, kernel_size=(4, 4), stride=(2, 2))
        self.sub_conv2 = nn.Conv2d(in_channels=self.out1, out_channels=self.out2, kernel_size=(2,2), stride=(1, 1))
        self.sub_conv3 = nn.Conv2d(in_channels=self.out2, out_channels=self.out3, kernel_size=(2,2), stride=(1, 1))

        # 3D Conv Operation
        # self.bn3d_sub_1 = nn.BatchNorm3d(self.out1)
        # self.bn3d_sub_2 = nn.BatchNorm3d(self.out2)
        # self.bn3d_sub_3 = nn.BatchNorm3d(self.out3)

        # 2D Conv Operation
        self.bn3d_sub_1 = nn.BatchNorm2d(self.out1)
        self.bn3d_sub_2 = nn.BatchNorm2d(self.out2)
        self.bn3d_sub_3 = nn.BatchNorm2d(self.out3)


        # 3D Conv Operation
        # self.max_pool_3d1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # 2D Conv Operation
        self.max_pool_3d1 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        subx = x

        subx = self.sub_conv1(subx)
        subx = self.bn3d_sub_1(subx)
        subx = self.relu(subx)


        subx = self.sub_conv2(subx)
        subx = self.bn3d_sub_2(subx)
        subx = self.relu(subx)

        subx = self.max_pool_3d1(subx)

        subx = self.sub_conv3(subx)
        subx = self.bn3d_sub_3(subx)
        subx = self.relu(subx)

        return subx

if __name__ == "__main__":
    x = torch.rand(4, 64, 100, 100, 6)

    net = Unit()
    output = net(x)
    # print(output.shape)




