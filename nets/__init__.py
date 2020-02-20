import os
import numpy as np


MODEL_DIR = '/mnt/Data/cs5242-dataset/models'
CSNET_DIR = os.path.join(MODEL_DIR, 'csnet')
IR_CSN_IG65_FILE = os.path.join(CSNET_DIR, 'irCSN_152_ig65m_from_scratch_f125286141.pkl')


def load_csn_model(ckpt='ig65'):
    import _pickle as pickle
    ckpt = str(ckpt).lower()
    if ckpt == 'ig65':
        file = IR_CSN_IG65_FILE
    else:
        raise ValueError('no such checkpoint file')
    
    # solution to opening the file: https://github.com/ohtake/VMZ/commit/41800f475ef09624ecf1461bb19f1e5ee2edf0ac
    with open(file, 'rb') as f:
        model_ckpt = pickle.load(f, encoding='latin1')
    model_ckpt = model_ckpt['blobs']
    return model_ckpt


if __name__ == '__main__':
    ckpt = load_csn_model('ig65')
    layers = np.sort(list(ckpt.keys()))
    for key in layers:
        print(key)

