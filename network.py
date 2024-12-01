from blocks import *
import torch
import torch.nn as nn
import torch.nn.functional as F  # 别忘了导入 F

class resnet50(nn.Module):
    def __init__(self, in_num, mid_num=64):
        super(resnet50, self).__init__()
        
        self.conv1 = nn.Conv2d(in_num, mid_num, stride=2, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(mid_num)
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.out_num1 = 256
        self.out_num2 = 512
        self.block1 = nn.Sequential(
            bottleneck_block(mid_num, self.out_num1, downsampling=None),
            bottleneck_block(self.out_num1, self.out_num1, downsampling=None),
            bottleneck_block(self.out_num1, self.out_num1, downsampling=None)
        )
        self.block2 = nn.Sequential(
            bottleneck_block(self.out_num1, self.out_num2, stride=2, mid_num=128, downsampling=1),
            bottleneck_block(self.out_num2, self.out_num2, mid_num=128, downsampling=None),
            bottleneck_block(self.out_num2, self.out_num2, mid_num=128, downsampling=None),
            bottleneck_block(self.out_num2, self.out_num2, mid_num=128, downsampling=None)
        )
        self.out_num3 = 1024
        self.block3 = nn.Sequential(
            bottleneck_block(self.out_num2, self.out_num3, mid_num=256, stride=2, downsampling=1),
            bottleneck_block(self.out_num3, self.out_num3, mid_num=256, downsampling=None),
            bottleneck_block(self.out_num3, self.out_num3, mid_num=256, downsampling=None),
            bottleneck_block(self.out_num3, self.out_num3, mid_num=256, downsampling=None)
        )
        self.out_num4 = 2048
        self.block4 = nn.Sequential(
            bottleneck_block(self.out_num3, self.out_num4, mid_num=512, stride=2, downsampling=1),
            bottleneck_block(self.out_num4, self.out_num4, mid_num=512, downsampling=None),
            bottleneck_block(self.out_num4, self.out_num4, mid_num=512, downsampling=None),
        )
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(self.out_num4, 1000)
        self.flat = lambda x: x.view(x.size(0), -1) 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        print("Input Shape: ", x.shape)  # 输入

        x_out = self.conv1(x)
        print("After Conv1 Shape: ", x_out.shape)

        x_out = self.bn1(x_out)
        x_out = F.relu(x_out)

        x_out = self.maxpooling(x_out)
        print("After MaxPooling Shape: ", x_out.shape)

        x_out = self.block1(x_out)
        print("After Block1 Shape: ", x_out.shape)

        x_out = self.block2(x_out)
        print("After Block2 Shape: ", x_out.shape)

        x_out = self.block3(x_out)
        print("After Block3 Shape: ", x_out.shape)

        x_out = self.block4(x_out)
        print("After Block4 Shape: ", x_out.shape)

        x_out = self.avgpool(x_out)
        print("After AvgPool Shape: ", x_out.shape)

        x_out = self.flat(x_out)
        print("After Flatten Shape: ", x_out.shape)

        x_out = self.fc(x_out)
        print("After FC Shape: ", x_out.shape)

        x_out = self.softmax(x_out)
        return x_out