import torch
import torch.nn as nn
from unit import Unit


class iLayer(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.weights = torch.ones(shape, requires_grad=True, device=torch.device('cuda'))
        # self.weights = torch.rand(shape, requires_grad=True)
        #
        # print("weight shape", self.weights.shape)
        self.bias = torch.zeros(shape, requires_grad=True, device=torch.device('cuda'))
        # self.bias = torch.rand(shape, requires_grad=True)

    def forward(self, input):
        return torch.mul(input, self.weights) + self.bias


class seg_dynamic(nn.Module):
    def __init__(self):
        super(seg_dynamic, self).__init__()

        self.out1 = 64
        self.out2 = 128
        self.out3 = 256
        self.out4 = 512

        self.relu = nn.ReLU(inplace=True)

        # mainnet & deconv parameters initialization
        self.bn2d_main_1 = nn.BatchNorm2d(self.out1)
        self.bn2d_main_2 = nn.BatchNorm2d(self.out2)
        self.bn2d_main_3 = nn.BatchNorm2d(self.out3)

        self.bn2d_dev_1 = nn.BatchNorm2d(self.out3)
        self.bn2d_dev_2 = nn.BatchNorm2d(self.out2)
        self.bn2d_dev_3 = nn.BatchNorm2d(self.out1)

        self.bn1d_cat_1 = nn.BatchNorm1d(64 * 22 * 22)

        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.main_conv1 = nn.Conv2d(in_channels=1, out_channels=self.out1, kernel_size=2)
        self.main_conv2 = nn.Conv2d(in_channels=self.out1, out_channels=self.out2, kernel_size=2)
        self.main_conv3 = nn.Conv2d(in_channels=self.out2, out_channels=self.out3, kernel_size=2)

        self.deconv1 = nn.ConvTranspose2d(512, 256, (3, 3), stride=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(256, 128, (2, 2), stride=(2, 2))
        self.deconv3 = nn.ConvTranspose2d(128, 64, (2, 2), stride=(2, 2))
        self.deconv4 = nn.ConvTranspose2d(64, 1, (1, 1), stride=(1, 1))


        # subnet parameters initialization

        # combined channels input
        # self.convs = nn.ModuleList([Unit(10 * 2), Unit(10 * 2), Unit(10 * 2),
        #                             Unit(5 * 2), Unit(5 * 2), Unit(5 * 2), Unit(5 * 2),
        #                             Unit(2 * 2), Unit(2 * 2), Unit(2 * 2), Unit(2 * 2), Unit(2 * 2)])
        # self.affines = nn.ModuleList([iLayer([self.out3,23,23]) for i in range(12)])

        # normal channel input
        self.convs = nn.ModuleList([Unit(10), Unit(10), Unit(10),
                                    Unit(5), Unit(5), Unit(5), Unit(5),
                                    Unit(2), Unit(2), Unit(2), Unit(2), Unit(2)])
        self.affines = nn.ModuleList([iLayer([self.out3,23,23]) for i in range(12)])

        # cat batchnormalization
        self.cat_bn = nn.BatchNorm2d(self.out4)

    def deconv(self, x):
        x = self.deconv1(x)
        x = self.bn2d_dev_1(x)
        x = self.relu(x)
        # print("deconv1", x.shape)

        x = self.deconv2(x)
        x = self.bn2d_dev_2(x)
        x = self.relu(x)
        # print("deconv2", x.shape)

        x = self.deconv3(x)
        x = self.bn2d_dev_3(x)
        x = self.relu(x)
        # print("deconv3", x.shape)

        x = self.deconv4(x)
        # print("deconv4", x.shape)
        return x

    def subnet(self,x):
        # 2D Conv Operation
        sub_x = torch.squeeze(x)

        # 3D Conv Operation
        # sub_x = x.view(4, 1, 60, 100, 100)

        # print("subx shape", sub_x.shape)

        # combined channels input
        # for i in range(0, 12, 1):
        #
        #     subx = []
        #     if i < 3:
        #         step = 10 * 2
        #         subx = sub_x[:, i * step : i * step + step, :, :]
        #     elif i >= 3 and i < 7:
        #         step = 5 * 2
        #         subx = sub_x[:, 60 + (i - 3) * step : 60 + (i - 3) * step + step, :, :]
        #     elif i >= 7 and i < 12:
        #         step = 2 * 2
        #         subx = sub_x[:, 100 + (i - 7) * step : 100 + (i - 7) * step + step, :, :]

        # normal channel input
        for i in range(0, 12, 1):

            subx = []
            if i < 3:
                step = 10
                subx = sub_x[:, i * step : i * step + step, :, :]
            elif i >= 3 and i < 7:
                step = 5
                subx = sub_x[:, 30 + (i - 3) * step : 30 + (i - 3) * step + step, :, :]
            elif i >= 7 and i < 12:
                step = 2
                subx = sub_x[:, 50 + (i - 7) * step : 50 + (i - 7) * step + step, :, :]
            # 3D Conv Operation
            # sub_x = subx.permute(0,1,3,4,2)

            subx = self.convs[i](subx).to(torch.device("cuda")).float()

            _add = self.affines[i](subx)

            if i == 0:
                sub_output = _add
            else:
                sub_output = torch.add(sub_output, _add)

            # print("sub_output shape", sub_output.shape)
            # print(i // 10)

        sub_output = torch.unsqueeze(sub_output, 1)
        # print("sub_output", sub_output.shape)

        return sub_output

    def mainnet(self, x):
        x = self.main_conv1(x)
        # print("conv1", x.shape)
        x = self.bn2d_main_1(x)
        x = self.relu(x)

        x = self.max_pool1(x)
        # print("pool 1", x.shape)

        x = self.main_conv2(x)
        # print("main conv2", x.shape)
        x = self.bn2d_main_2(x)
        x = self.relu(x)

        x = self.max_pool1(x)
        # print("pool 1", x.shape)

        x = self.main_conv3(x)
        x = self.bn2d_main_3(x)
        x = self.relu(x)
        # print("main conv3", x.shape)
        # x = self.max_pool2(x)
        #print("pool 2", x.shape)
        return x

    def forward(self,subx, mainx):
        mainx = self.mainnet(mainx)
        # print("mainx", mainx.shape)

        subx = self.subnet(subx)
        subx = torch.squeeze(subx)
        # print("subx", subx.shape)

        x = torch.cat((subx, mainx), 1)
        x = self.cat_bn(x)

        x = self.deconv(x)

        x = x.view(-1,100,100)

        return x


if __name__ == "__main__":
    subx = torch.rand(2,1,60,100, 100)
    mainx =  torch.rand(2,1,100, 100)

    net = seg_dynamic()
    output = net(subx, mainx)
    print(output.shape)