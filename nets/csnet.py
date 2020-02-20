import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


DEEP_FILTER_CONFIG = [
    [256, 64],
    [512, 128],
    [1024, 256],
    [2048, 512]
]


class Bottleneck(nn.Module):
    def __init__(self, input_filters, output_filters, base_filters, downsampling=False,
                 temporal_downsampling=None, spatial_bn=True, block_type='3d', is_real_3d=True,
                 gamma_init=False, group=1, use_shuffle=False):
        super(Bottleneck, self).__init__()
        self.input_filters = int(input_filters)
        self.output_filters = int(output_filters)
        self.base_filters = int(base_filters)
        self.block_type = str(block_type)

        self.downsampling = bool(downsampling)
        self.temporal_downsampling = self.downsampling if temporal_downsampling is None else temporal_downsampling
        self.spatial_bn = bool(spatial_bn)
        self.is_real_3d = bool(is_real_3d)
        if block_type == '2.5d':
            assert self.is_real_3d

        self.gamma_init = bool(gamma_init)
        self.group = int(group)
        self.use_shuffle = bool(use_shuffle)

        # define layers here
        self.layers = [self.add_conv(self.input_filters, self.base_filters, kernels=1)]
        if self.spatial_bn:
            # todo: add init gamma
            self.layers.append(nn.BatchNorm3d(self.base_filters, eps=1e-03))
        self.layers.append(nn.ReLU(inplace=True))

        if self.downsampling:
            if self.is_real_3d and self.temporal_downsampling:
                use_striding = [2, 2, 2]
            else:
                use_striding = [1, 2, 2]
        else:
            use_striding = [1, 1, 1]
        if self.downsampling:
            print(use_striding)

        self.layers.append(self.add_conv(self.base_filters, self.base_filters,
                                         kernels=[3, 3, 3] if self.is_real_3d else [1, 3, 3],
                                         strides=use_striding, pads=[1, 1, 1] if is_real_3d else [0, 1, 1],
                                         block_type=self.block_type, group=self.group))
        if self.spatial_bn:
            self.layers.append(nn.BatchNorm3d(self.base_filters))
        self.layers.append(nn.ReLU(inplace=True))

        self.layers.append(self.add_conv(self.base_filters, self.output_filters, kernels=[1, 1, 1]))
        if self.spatial_bn:
            self.layers.append(nn.BatchNorm3d(self.output_filters))
        self.net = nn.Sequential(*self.layers)

        self.shortcut = lambda tensor: tensor  # identity
        if (self.output_filters != self.input_filters or self.downsampling):
            shortcut_layers = [
                nn.Conv3d(self.input_filters, self.output_filters, kernel_size=1,
                          stride=use_striding, bias=False)
            ]
            if self.spatial_bn:
                shortcut_layers.append(nn.BatchNorm3d(self.output_filters, eps=1e-3))
            self.shortcut = nn.Sequential(*shortcut_layers)

    def forward(self, x):
        return F.relu(self.net(x) + self.shortcut(x), inplace=True)

    def add_conv(self, in_filters, out_filters, kernels, strides=None, pads=None, block_type='3d', group=1):
        if strides is None:
            strides = [1, 1, 1]
        if pads is None:
            pads = [0, 0, 0]

        if group > 1:
            assert self.block_type == '3d-group'

        if block_type == '3d':
            return nn.Conv3d(in_channels=in_filters, out_channels=out_filters, kernel_size=kernels,
                             stride=strides, padding=pads, bias=False)
        elif block_type == '3d-sep':
            # depthwise convolution
            return nn.Conv3d(in_channels=in_filters, out_channels=out_filters, kernel_size=kernels,
                             stride=strides, padding=pads, bias=False, groups=self.input_filters)
        else:
            raise ValueError('no such block type {}'.format(block_type))


class IrCsn152(nn.Module):
    def __init__(self, n_class, mode='ir'):
        super(IrCsn152, self).__init__()
        assert mode in ['ir', 'ip'], 'mode should be iteration reducing or preserving'

        self.n_class = int(n_class)
        self.mode = mode

        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64, eps=1e-03),
            nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        # block type is 3d-sep

        self.n_bottlenecks = [3, 8, 36, 3]

        # conv_2x
        conv2_layers = [
            Bottleneck(64, DEEP_FILTER_CONFIG[0][0], DEEP_FILTER_CONFIG[0][1], block_type=self.block_type,
                       use_shuffle=self.use_shuffle)
        ]
        for _ in range(self.n_bottlenecks[0] - 1):
            conv2_layers.append(Bottleneck(DEEP_FILTER_CONFIG[0][0], DEEP_FILTER_CONFIG[0][0], DEEP_FILTER_CONFIG[0][1],
                                           is_real_3d=False))
        self.conv2 = nn.Sequential(*conv2_layers)

        # conv_3x
        conv3_layers = [
            Bottleneck(DEEP_FILTER_CONFIG[0][0], DEEP_FILTER_CONFIG[1][0], DEEP_FILTER_CONFIG[1][1],
                       downsampling=True, is_real_3d=False)
        ]
        for _ in range(self.n_bottlenecks[1] - 1):
            conv3_layers.append(Bottleneck(DEEP_FILTER_CONFIG[1][0], DEEP_FILTER_CONFIG[1][0], DEEP_FILTER_CONFIG[1][1],
                                           is_real_3d=False))
        self.conv3 = nn.Sequential(*conv3_layers)

        # conv_4x
        conv4_layers = [
            Bottleneck(DEEP_FILTER_CONFIG[1][0], DEEP_FILTER_CONFIG[2][0], DEEP_FILTER_CONFIG[2][1],
                       downsampling=True, is_real_3d=False)
        ]
        for _ in range(self.n_bottlenecks[2] - 1):
            conv4_layers.append(Bottleneck(DEEP_FILTER_CONFIG[2][0], DEEP_FILTER_CONFIG[2][0], DEEP_FILTER_CONFIG[2][1],
                                           is_real_3d=False))
        self.conv4 = nn.Sequential(*conv4_layers)


    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        # out = self.conv4(out)
        return out


if __name__ == '__main__':
    network = IrCsn152(1000)
    data = torch.from_numpy(np.random.randn(2, 3, 8, 224, 224)).float()
    out = network(data)
    print(out.shape)
