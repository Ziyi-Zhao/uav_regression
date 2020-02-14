import torch
import torch.nn as nn



class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()

        self.out1 = 64
        self.out2 = 128
        self.out3 = 256

        self.relu = nn.ReLU(inplace=True)

        # 3D Conv Operation
        # self.bn3d_sub_1 = nn.BatchNorm3d(self.out1)
        # self.bn3d_sub_2 = nn.BatchNorm3d(self.out2)
        # self.bn3d_sub_3 = nn.BatchNorm3d(self.out3)

        # 2D Conv Operation
        self.bn3d_sub_1 = nn.BatchNorm2d(self.out1)
        self.bn3d_sub_2 = nn.BatchNorm2d(self.out2)
        self.bn3d_sub_3 = nn.BatchNorm2d(self.out3)

        self.bn2d_main_1 = nn.BatchNorm2d(self.out1)
        self.bn2d_main_2 = nn.BatchNorm2d(self.out2)
        self.bn2d_main_3 = nn.BatchNorm2d(self.out3)

        self.bn2d_dev_1 = nn.BatchNorm2d(self.out3)
        self.bn2d_dev_2 = nn.BatchNorm2d(self.out2)
        self.bn2d_dev_3 = nn.BatchNorm2d(self.out1)

        self.bn1d_cat_1 = nn.BatchNorm1d(64*22*22)

        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3D Conv Operation
        # self.max_pool_3d1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # 2D Conv Operation
        self.max_pool_3d1 = nn.MaxPool2d(kernel_size=2, stride=2)


        # 3D Conv Operation
        # self.sub_conv1 = nn.Conv3d(in_channels=1, out_channels=self.out1, kernel_size=(4,4,10), stride=(2, 2, 2))
        # self.sub_conv2 = nn.Conv3d(in_channels=self.out1, out_channels=self.out2, kernel_size=(2,2,5), stride=(1, 1, 2))
        # self.sub_conv3 = nn.Conv3d(in_channels=self.out2, out_channels=self.out3, kernel_size=(2,2,5), stride=(1, 1, 2))

        # 2D Conv Operation
        self.sub_conv1 = nn.Conv2d(in_channels=60, out_channels=self.out1, kernel_size=(4,4), stride=(2, 2))
        self.sub_conv2 = nn.Conv2d(in_channels=self.out1, out_channels=self.out2, kernel_size=(2,2), stride=(1, 1))
        self.sub_conv3 = nn.Conv2d(in_channels=self.out2, out_channels=self.out3, kernel_size=(2,2), stride=(1, 1))

        self.main_conv1 = nn.Conv2d(in_channels=1, out_channels=self.out1, kernel_size=2)
        self.main_conv2 = nn.Conv2d(in_channels=self.out1, out_channels=self.out2, kernel_size=2)
        self.main_conv3 = nn.Conv2d(in_channels=self.out2, out_channels=self.out3, kernel_size=2)

        self.deconv1 = nn.ConvTranspose2d(512, 256, (3, 3), stride=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(256, 128, (2, 2), stride=(2, 2))
        self.deconv3 = nn.ConvTranspose2d(128, 64, (2, 2), stride=(2, 2))
        self.deconv4 = nn.ConvTranspose2d(64, 1, (1, 1), stride=(1, 1))

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
        x = torch.squeeze(x)

        # 3D Conv Operation
        # x = x.view(4, 1, 60, 100, 100)
        # x = x.permute(0,1,3,4,2)

        # x = x.permute(0, 3, 1, 2)
        x = self.sub_conv1(x)
        x = self.bn3d_sub_1(x)
        x = self.relu(x)

        x = self.sub_conv2(x)
        x = self.bn3d_sub_2(x)
        x = self.relu(x)
        # print("conv2", x.shape)

        x = self.max_pool_3d1(x)
        # print("pool 1", x.shape)

        x = self.sub_conv3(x)
        x = self.bn3d_sub_3(x)
        x = self.relu(x)
        # print("conv3", x.shape)

        return x

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

    def forward(self, subx, mainx):
        subx = self.subnet(subx)
        subx = torch.squeeze(subx)
        mainx = self.mainnet(mainx)
        # print("subx", subx.shape)
        # print("mainx", mainx.shape)
        x = torch.cat((subx, mainx), 1)
        x = self.deconv(x)

        x = x.view(-1,100,100)
        return x


if __name__ == "__main__":
    subx = torch.rand(2,1, 100, 100,60)
    mainx =  torch.rand(2,1,100, 100)

    net = MainNet()
    output = net(subx, mainx)
    print(output.shape)
#[2, 256, 23, 23]