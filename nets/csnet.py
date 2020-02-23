import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from nets import init_bn_layer, init_hidden_layer
from nets import MODEL_DIR

CSNET_DIR = os.path.join(MODEL_DIR, 'csnet')
IRCSN_IG65_FILE = os.path.join(CSNET_DIR, 'irCSN_152_ig65m_from_scratch_f125286141.pkl')
IRCSN_KINETICS_FILE = os.path.join(CSNET_DIR, 'irCSN_152_ft_kinetics_from_ig65m_f126851907.pkl')
IR_CSN_IG65_CSV_FILE = os.path.join(CSNET_DIR, 'irCSN_152_ig65m_from_scratch_f125286141.csv')
N_CLASSES_IG65 = 359
N_CLASSES_KINETICS = 400

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
        if (self.output_filters != self.input_filters) or self.downsampling:
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
                             stride=strides, padding=pads, bias=False, groups=in_filters)
        else:
            raise ValueError('no such block type {}'.format(block_type))


class IrCsn152(nn.Module):
    def __init__(self, n_classes, clip_len, crop_size, pretrained_ckpt=None):
        super(IrCsn152, self).__init__()
        self.n_classes = int(n_classes)
        self.pretrained_ckpt = pretrained_ckpt
        self.use_shuffle = False
        self.block_type = '3d-sep'
        self.n_bottlenecks = [3, 8, 36, 3]

        self.clip_len = int(clip_len)
        self.crop_size = int(crop_size)

        self.final_temporal_kernel = self.clip_len // 8
        if self.crop_size == 112 or self.crop_size == 128:
            self.final_spatial_kernel = 7
        elif self.crop_size == 224 or crop_size == 256:
            self.final_spatial_kernel = 14
        elif self.crop_size == 320:
            self.final_spatial_kernel = 14
        else:
            raise ValueError('unknown crop size')

        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64, eps=1e-03),
            nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # conv_2x
        conv2_layers = [
            Bottleneck(64, DEEP_FILTER_CONFIG[0][0], DEEP_FILTER_CONFIG[0][1], block_type=self.block_type,
                       use_shuffle=self.use_shuffle)
        ]
        for _ in range(self.n_bottlenecks[0] - 1):
            conv2_layers.append(Bottleneck(DEEP_FILTER_CONFIG[0][0], DEEP_FILTER_CONFIG[0][0], DEEP_FILTER_CONFIG[0][1],
                                           block_type=self.block_type, use_shuffle=self.use_shuffle))
        self.conv2 = nn.Sequential(*conv2_layers)

        # conv_3x
        conv3_layers = [
            Bottleneck(DEEP_FILTER_CONFIG[0][0], DEEP_FILTER_CONFIG[1][0], DEEP_FILTER_CONFIG[1][1],
                       downsampling=True, block_type=self.block_type, use_shuffle=self.use_shuffle)
        ]
        for _ in range(self.n_bottlenecks[1] - 1):
            conv3_layers.append(Bottleneck(DEEP_FILTER_CONFIG[1][0], DEEP_FILTER_CONFIG[1][0], DEEP_FILTER_CONFIG[1][1],
                                           block_type=self.block_type, use_shuffle=self.use_shuffle))
        self.conv3 = nn.Sequential(*conv3_layers)

        # conv_4x
        conv4_layers = [
            Bottleneck(DEEP_FILTER_CONFIG[1][0], DEEP_FILTER_CONFIG[2][0], DEEP_FILTER_CONFIG[2][1],
                       downsampling=True, block_type=self.block_type, use_shuffle=self.use_shuffle)
        ]
        for _ in range(self.n_bottlenecks[2] - 1):
            conv4_layers.append(Bottleneck(DEEP_FILTER_CONFIG[2][0], DEEP_FILTER_CONFIG[2][0], DEEP_FILTER_CONFIG[2][1],
                                           block_type=self.block_type, use_shuffle=self.use_shuffle))
        self.conv4 = nn.Sequential(*conv4_layers)

        # conv 5x
        conv5_layers = [
            Bottleneck(DEEP_FILTER_CONFIG[2][0], DEEP_FILTER_CONFIG[3][0], DEEP_FILTER_CONFIG[3][1],
                       downsampling=True, block_type=self.block_type, use_shuffle=self.use_shuffle)
        ]
        for _ in range(self.n_bottlenecks[3] - 1):
            conv5_layers.append(Bottleneck(DEEP_FILTER_CONFIG[3][0], DEEP_FILTER_CONFIG[3][0], DEEP_FILTER_CONFIG[3][1],
                                           block_type=self.block_type, use_shuffle=self.use_shuffle))
        self.conv5 = nn.Sequential(*conv5_layers)

        self.last_out = nn.Linear(in_features=DEEP_FILTER_CONFIG[3][0], out_features=self.n_classes)
        if self.pretrained_ckpt is None:
            self._init_weights()
        else:
            self.load_caffe_weights()

    def _init_weights(self):
        conv1_layers = list(self.conv1.children())
        nn.init.kaiming_normal_(conv1_layers[0].weight, nonlinearity='relu')

        def init_bottleneck_layers(bottleneck):
            layers = list(bottleneck.net.children())
            if isinstance(bottleneck.shortcut, nn.Sequential):
                layers += list(bottleneck.shortcut.children())

            for layer in layers:
                if isinstance(layer, nn.Conv3d):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        b_layers = list(self.conv2.children()) + list(self.conv3.children()) + list(self.conv4.children()) + \
                   list(self.conv5.children())
        for b_layer in b_layers:
            init_bottleneck_layers(b_layer)

    def load_caffe_weights(self):
        checkpoint = load_csn_model(ckpt=self.pretrained_ckpt)
        conv1_layers = list(self.conv1.children())
        init_hidden_layer(hidden_layer=conv1_layers[0], scope='conv1', weight_dict=checkpoint)
        init_bn_layer(bn_layer=conv1_layers[1], scope='conv1_spatbn_relu', weight_dict=checkpoint)

        def init_bottleneck_layers(subnet, bi):
            bottleneck_layers = list(subnet.children())
            for bottleneck in bottleneck_layers:
                self._load_bottleneck_weights(bottleneck, bi, checkpoint)
                bi += 1
            return bi
        i = 0  # bottleneck idx
        i = init_bottleneck_layers(self.conv2, i)
        i = init_bottleneck_layers(self.conv3, i)
        i = init_bottleneck_layers(self.conv4, i)
        init_bottleneck_layers(self.conv5, i)

        if self.n_classes == N_CLASSES_IG65:
            init_hidden_layer(hidden_layer=self.last_out, scope='last_out_L359', weight_dict=checkpoint)
        elif self.n_classes == N_CLASSES_KINETICS:
            init_hidden_layer(hidden_layer=self.last_out, scope='last_out_L400', weight_dict=checkpoint)

    @staticmethod
    def _load_bottleneck_weights(bottleneck, idx, weight_dict):
        prefix = 'comp_{}'.format(idx)
        layers = list(bottleneck.net.children())
        layer_idxs = [1, 3, 4]

        i = 0
        for layer in layers:
            l_id = layer_idxs[i]
            if isinstance(layer, nn.BatchNorm3d):
                name = '_'.join([prefix, 'spatbn', str(l_id)])
                init_bn_layer(layer, scope=name, weight_dict=weight_dict)
                i += 1
            elif isinstance(layer, nn.Conv3d):
                name = '_'.join([prefix, 'conv', str(l_id)])
                init_hidden_layer(layer, scope=name, weight_dict=weight_dict)

        if isinstance(bottleneck.shortcut, nn.Sequential):
            shortcut_layers = list(bottleneck.shortcut.children())
            conv_name = 'shortcut_projection_{}'.format(idx)
            bn_name = '_'.join([conv_name, 'spatbn'])
            init_hidden_layer(shortcut_layers[0], scope=conv_name, weight_dict=weight_dict)
            init_bn_layer(shortcut_layers[1], scope=bn_name, weight_dict=weight_dict)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = torch.mean(torch.mean(torch.mean(out, dim=4), dim=3), dim=2)
        out = out.reshape(-1, DEEP_FILTER_CONFIG[3][0])
        out = self.last_out(out)
        return out


def load_csn_model(ckpt='ig65'):
    import _pickle as pickle
    ckpt = str(ckpt).lower()
    if ckpt == 'ig65':
        file = IRCSN_IG65_FILE
    elif ckpt == 'kinetics':
        file = IRCSN_KINETICS_FILE
    else:
        raise ValueError('no such checkpoint file')

    # solution to opening the file: https://github.com/ohtake/VMZ/commit/41800f475ef09624ecf1461bb19f1e5ee2edf0ac
    with open(file, 'rb') as f:
        model_ckpt = pickle.load(f, encoding='latin1')
    model_ckpt = model_ckpt['blobs']
    return model_ckpt


def test_implementation():
    import data.kinetics_data as kinetics
    import utils.csnet_utils as ir_csn
    network = IrCsn152(N_CLASSES_KINETICS, clip_len=ir_csn.CLIP_LEN, crop_size=ir_csn.CROP_SIZE)
    network.eval()
    video_files, labels = kinetics.get_train_data()
    # print(video_files[0])
    # exit()
    video_filenames = [os.path.split(file)[-1] for file in video_files]
    n_correct = 0
    n_vids = 0
    with torch.no_grad():
        for i, vid in enumerate(video_filenames):
            if i == 0:
                continue
            logits = []
            print(labels[i])
            for j in range(30):
                clip_file = os.path.join(kinetics.TRAIN_CLIP_DIR, '{}.{}.npy'.format(j, vid))
                clip = np.load(clip_file)
                clip = torch.from_numpy(clip).unsqueeze(0)  # expand to batch size
                # print(clip)
                logit = network(clip)
                # print(logit)
                print(torch.argmax(logit.squeeze()))
                logits.append(logit)
            logits = torch.cat(logits, dim=0)
            logits = torch.mean(logits, dim=0)
            prediction = torch.argmax(logits)
            label = labels[i]
            print(logits)
            print(prediction)
            print(label)
            break



def main():
    # clip_len = 8
    # crop_size = 224
    # network = IrCsn152(N_CLASSES_IG65, clip_len, crop_size)
    # network.load_caffe_weights()
    # data = torch.from_numpy(np.random.randn(2, 3, 8, 224, 224)).float()
    # out = network(data)
    # print(out.shape)
    test_implementation()
    pass


if __name__ == '__main__':
    main()
