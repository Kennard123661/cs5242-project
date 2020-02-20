import torch
import torch.nn as nn
import numpy as np


class Bottleneck(nn.Module):
    expansion = 4
    Conv3d = nn.Conv3d

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = self.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = self.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = self.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ChannelSeparated3DResNet152(nn.Module):
    def __init__(self, n_class, mode='ir'):
        super(ChannelSeparated3DResNet152, self).__init__()
        assert mode in ['ir', 'ip'], 'mode should be iteration reducing or preserving'

        self.n_class = int(n_class)
        self.mode = mode

        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64, eps=1e-03),
            nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        n_bottlenecks = [3, 8, 36, 3]
    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.pool1(out)
        return out


if __name__ == '__main__':
    network = ChannelSeparated3DResNet152(1000)
    data = torch.from_numpy(np.random.randn(2, 3, 32, 224, 224)).float()
    out = network(data)
    print(out.shape)
