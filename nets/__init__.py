import os
import numpy as np
import pandas as pd
import torch


MODEL_DIR = '/mnt/Data/cs5242-dataset/models'
CSNET_DIR = os.path.join(MODEL_DIR, 'csnet')
IRCSN_IG65_FILE = os.path.join(CSNET_DIR, 'irCSN_152_ig65m_from_scratch_f125286141.pkl')
IRCSN_KINETICS_FILE = os.path.join(CSNET_DIR, 'irCSN_152_ft_kinetics_from_ig65m_f126851907.pkl')
IR_CSN_IG65_CSV_FILE = os.path.join(CSNET_DIR, 'irCSN_152_ig65m_from_scratch_f125286141.csv')


def load_csn_model(ckpt='ig65'):
    import _pickle as pickle
    ckpt = str(ckpt).lower()
    if ckpt == 'ig65':
        file = IRCSN_KINETICS_FILE
    else:
        raise ValueError('no such checkpoint file')
    
    # solution to opening the file: https://github.com/ohtake/VMZ/commit/41800f475ef09624ecf1461bb19f1e5ee2edf0ac
    with open(file, 'rb') as f:
        model_ckpt = pickle.load(f, encoding='latin1')
    model_ckpt = model_ckpt['blobs']
    return model_ckpt


def create_csn_mapping():
    model_ckpt = load_csn_model()

    # layer_names = sorted(model_ckpt.keys())
    layer_names = sorted(list(model_ckpt.keys()))
    layer_shapes = []
    for name in layer_names:
        layer_weights = model_ckpt[name]
        layer_shapes.append(layer_weights.shape)
    data = np.array([layer_names, layer_shapes]).T
    df = pd.DataFrame(data=data, columns=['name', 'shape'])
    df.to_csv(IR_CSN_IG65_CSV_FILE, index=False)
    print(layer_names)
    print(layer_shapes)


def init_bn_layer(bn_layer, scope, weight_dict):
    """ initializes weights for batchnorm layers """
    bn_layer.running_mean.data = torch.from_numpy(weight_dict[scope + '_rm'])
    bn_layer.running_var.data = torch.from_numpy(weight_dict[scope + '_riv'])
    bn_layer.weight.data = torch.from_numpy(weight_dict[scope + '_s'])
    bn_layer.bias.data = torch.from_numpy(weight_dict[scope + '_b'])


def init_hidden_layer(hidden_layer, scope, weight_dict):
    """ initializes weights for linear and conv layers """
    hidden_layer.weight.data = torch.from_numpy(weight_dict[scope + '_w'])
    if scope + '_b' in weight_dict.keys():  # check for bias
        hidden_layer.bias.data = torch.from_numpy(weight_dict[scope + '_b'])


if __name__ == '__main__':
    create_csn_mapping()
