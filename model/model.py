import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)

class ResidualBlockA(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlockA, self).__init__()
        
        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        self.conv2 = conv3x3(in_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)
        
        return F.relu(x+out, True)

class ResidualBlockB(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlockB, self).__init__()
        
        self.conv1_1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1_1 = nn.BatchNorm2d(out_channel)
        
        self.conv1_2 = conv3x3(out_channel, out_channel, stride=2)
        self.bn1_2 = nn.BatchNorm2d(out_channel)
        
        self.conv2 = conv3x3(in_channel, out_channel, stride=2)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        
    def forward(self, x):
        out1 = self.conv1_1(x)
        out1 = F.relu(self.bn1_1(out1), True)
        
        out1 = self.conv1_2(out1)
        out1 = F.relu(self.bn1_2(out1), True)
        
        out2 = self.conv2(x)
        out2 = F.relu(self.bn2(out2), True)
        
        return F.relu(out1+out2, True)

class FCN(nn.Module):
    def __init__(self, in_channel, out_channel, verbose=False):
        super(FCN, self).__init__()
        self.verbose = verbose
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.block2 = nn.Sequential(
            ResidualBlockA(16, 16),
            ResidualBlockA(16, 16),
            ResidualBlockA(16, 16),
            ResidualBlockB(16, 24)
        )
        
        self.block3 = nn.Sequential(
            ResidualBlockA(24, 24),
            ResidualBlockA(24, 24),
            ResidualBlockA(24, 24),
            ResidualBlockB(24, 32)
        )
        
        self.block4 = nn.Sequential(
            ResidualBlockA(32, 32),
            ResidualBlockA(32, 32),
            ResidualBlockA(32, 32),
            ResidualBlockB(32, 48)
        )
        
        self.block5 = ResidualBlockA(48, out_channel)
        
    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print("block 1 output : {}".format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print("block 2 output : {}".format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print("block 3 output : {}".format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print("block 4 output : {}".format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print("block 5 output : {}".format(x.shape))
        
        return x

class TSMLayer(nn.Module):
    def __init__(self, in_channel, num_classes, kernel_size=3, stride=1, padding=1):
        super(TSMLayer, self).__init__()
        
        self.conv2d = nn.Conv2d(in_channel, num_classes, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_classes)
        self.bnH = nn.AvgPool2d((6, 1)) # 高度归一化层，把一列向量求平均
        
    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.bnH(x)
        
        return x

class WaterMeterModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fcn = FCN(3, 48)
        self.tsm = TSMLayer(48, 21)

    def forward(self, x):
        x = self.fcn(x)
        x = self.tsm(x)
        
        return x