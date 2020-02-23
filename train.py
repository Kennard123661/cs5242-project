import os
import json
import torch
import numpy as np
from tqdm import tqdm
import torch.utils.data as tdata
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

import utils.csnet_utils as csnet_utils
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
        self.iteration = 0
        self.batch_size = int(self.config['batch-size'])
        self.n_iterations = int(self.config['num-iterations'])

        if dataset_name == 'breakfast':
            import data.breakfast_data as dataset
        else:
            raise ValueError('no such dataset')

        if model == 'ir-csn':
            self.crop_size = csnet_utils.CROP_SIZE
            self.model = IrCsn152(n_classes=dataset.N_CLASSES, clip_len=self.clip_len, crop_size=self.crop_size)
        else:
            raise ValueError('no such model')
        self.loss_fn = nn.CrossEntropyLoss()

        if optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError('no such optimizer')

        if scheduler == 'step':
            step_size = self.config['lr-decay-step-size']
            lr_decay = self.config['lr-decay-rate']
            self.scheduler = StepLR(self.optimizer, gamma=lr_decay, step_size=step_size)
        else:
            raise ValueError('no such scheduler')

        self.checkpoint_dir = os.path.join(CHECKPOINT_DIR, self.name)
        self.log_dir = os.path.join(LOG_DIR, self.name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.logger = SummaryWriter(self.log_dir)

        train_clips, train_labels = dataset.get_train_data()
        if self.n_iterations == 0:
            self.n_iterations = len(train_clips)
        test_clips, test_labels = dataset.get_test_data()

        if model == 'ir_csn':
            self.train_dataset = csnet_utils.TrainDataset(videos=train_clips, labels=train_clips,
                                                          resize=csnet_utils.CROP_SIZE, crop_size=csnet_utils.CROP_SIZE,
                                                          clip_len=csnet_utils.CLIP_LEN)
            # todo: implement test eval and train eval dataset
        else:
            raise ValueError('no such model... how did you even get here...')

    def train(self):
        start_epoch = self.epoch
        for i in range(start_epoch, self.max_epochs):
            print('INFO: epoch {0}/{1}'.format(i+1, self.max_epochs))
            self.train_step()
            self.eval_step()
            self.epoch += 1
            self.save_checkpoint(ckpt_name='model')
            self.save_checkpoint(ckpt_name='model-{}'.format(self.epoch))

    def train_step(self):
        print('INFO: training...')
        dataloader = tdata.DataLoader(self.train_dataset)
        self.model.train()
        epoch_losses = []
        i = 0
        for frames, labels in tqdm(dataloader):
            frames = frames.cuda()
            labels = labels.cuda()
            logits = self.model(frames)
            loss = self.loss_fn(logits, labels)

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            i += 1

            epoch_losses.append(loss.item())

            if i == self.n_iterations:
                break
        epoch_loss = np.mean(epoch_losses)
        print('INFO: training loss: {}'.format(epoch_loss))

        for loss in epoch_losses:
            train_log = {
                'loss': loss
            }
            self.iteration += 1
            self.logger.add_scalars('{}:train'.format(self.name), train_log, self.iteration)

    def eval_step(self):
        print('INFO: evaluating...')
        self.model.eval()
        self.eval_test()
        self.eval_train()

    def eval_train(self):
        print('INFO: evaluating train dataset...')
        pass

    def eval_test(self):
        print('INFO: evaluating test dataset...')
        pass

    def save_checkpoint(self, ckpt_name):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = os.path.join(self.checkpoint_dir, ckpt_name + '.pth')

        checkpoint_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'iteration': self.iteration
        }
        torch.save(checkpoint_dict, checkpoint_file)
        print('INFO: saved checkpoint {}'.format(checkpoint_file))

    def load_checkpoint(self, ckpt_name='model'):
        checkpoint_file = os.path.join(self.checkpoint_dir, ckpt_name + '.pth')
        if not os.path.exists(checkpoint_file):
            print('WARNING: checkpoint does not exist. Continuing...')
        else:
            checkpoint_dict = torch.load(checkpoint_file, map_location='cuda:{}'.format(0))
            self.model.load_state_dict(checkpoint_dict['model'])
            self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
            self.scheduler.load_state_dict(checkpoint_dict['scheduler'])
            self.epoch = checkpoint_dict['epoch']
            self.iteration = checkpoint_dict['iteration']

    def __del__(self):
        self.logger.close()
        del self.model


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