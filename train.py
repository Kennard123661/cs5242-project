import os
import json
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import data.ir_csn_data as csnet_data
from nets.csnet import IrCsn152

PROJECT_DIR = os.path.dirname(__file__)
CONFIG_DIR = os.path.join(PROJECT_DIR, 'configs')
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'checkpoints')
LOG_DIR = os.path.join(PROJECT_DIR, 'logs')


class Trainer:
    def __init__(self, name):
        self.name = str(name)
        config_file = os.path.join(CONFIG_DIR, name + '.json')
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        model = self.config['model']
        optimizer = self.config['optimizer']
        scheduler = self.config['scheduler']
        dataset_name = self.config['dataset']

        self.clip_len = int(self.config['clip-length'])
        self.lr = float(self.config['lr'])
        self.weight_decay = float(self.config['weight-decay'])

        self.max_epochs = int(self.config['max-epochs'])
        self.epoch = 0

        if dataset_name == 'kinetics':
            import data.kinetics_data as dataset
        elif dataset_name == 'breakfast':
            import data.breakfast_dataset as dataset
        else:
            raise ValueError('no such dataset')

        if model == 'ir_csn':
            self.crop_size = csnet_data.CROP_SIZE
            self.model = IrCsn152(n_classes=dataset.N_CLASSES, clip_len=self.clip_len, crop_size=self.crop_size)
        else:
            raise ValueError('no such model')

        if optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError('no such optimizer')

        if scheduler == 'step':
            step_size = self.config['lr-decay-step-size']
            lr_decay = self.config['lr-decay-rate']
            self.optimizer = StepLR(self.optimizer, gamma=lr_decay, step_size=step_size)
        else:
            raise ValueError('no such scheduler')

        self.checkpoint_dir = os.path.join(CHECKPOINT_DIR, self.name)
        self.log_dir = os.path.join(LOG_DIR, self.name)

        train_clips, train_labels = dataset.get_train_data()
        test_clips, test_labels = dataset.get_test_data()

        if model == 'ir_csn':
            self.train_dataset = csnet_data.TrainDataset(videos=train_clips, labels=train_clips,
                                                         resize=csnet_data.CROP_SIZE, crop_size=csnet_data.CROP_SIZE,
                                                         clip_len=csnet_data.CLIP_LEN)
            # todo: implement test dataset
        else:
            raise ValueError('no such model... how did you even get here...')

    def save_checkpoint(self, ckpt_name):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = os.path.join(self.checkpoint_dir, ckpt_name + '.pth')
        # todo: add save parameters

    def load_checkpoint(self, ckpt_name='model'):
        checkpoint_file = os.path.join(self.checkpoint_dir, ckpt_name + '.pth')
        if not os.path.exists(checkpoint_file):
            print('WARNING: checkpoint does not exist. Continuing...')
        else:
            # todo: add checkpoint loading.
            pass

    def train(self):
        start_epoch = self.epoch
        for i in range(start_epoch, self.max_epochs):
            print('INFO: epoch {0}/{1}'.format(i+1, self.max_epochs))
            self.train_step()
            self.eval_step()
            self.epoch += 1

    def train_step(self):
        print('INFO: training...')
        pass

    def eval_step(self):
        print('INFO: evaluating...')
        self.eval_test()
        self.eval_train()

    def eval_train(self):
        print('INFO: evaluating train dataset...')
        pass

    def eval_test(self):
        print('INFO: evaluating test dataset...')
        pass


def _set_deterministic_experiments():
    torch.manual_seed(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(1234)


def main():
    _set_deterministic_experiments()
    pass


if __name__ == '__main__':
    main()