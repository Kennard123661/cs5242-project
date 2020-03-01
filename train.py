import os
import json
import torch
import sys
import numpy as np
import argparse
import torch.utils.data as tdata
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from tqdm import tqdm
from nets.csnet import IrCsn152
from utils.train_utils import CustomLogger

PROJECT_DIR = os.path.dirname(__file__)
CONFIG_DIR = os.path.join(PROJECT_DIR, 'configs')
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'checkpoints')
LOG_DIR = os.path.join(PROJECT_DIR, 'logs')


class Trainer:
    def __init__(self, experiment):
        self.experiment = str(experiment)
        config_file = os.path.join(CONFIG_DIR, experiment + '.json')
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
        self.train_batch_size = int(self.config['train-batch-size'])
        self.eval_batch_size = int(self.config['test-batch-size'])
        self.n_iterations = int(self.config['num-iterations'])

        if dataset_name == 'breakfast':
            import data.breakfast_data as dataset_utils
            self.n_classes = dataset_utils.N_CLASSES
        else:
            raise ValueError('no such dataset')

        if model == 'ir-csn':
            import utils.csnet_utils as train_utils
            self.model = IrCsn152(n_classes=dataset_utils.N_CLASSES, clip_len=self.clip_len,
                                  crop_size=train_utils.CROP_SIZE)
        else:
            raise ValueError('no such model')
        self.train_batch_size = self.train_batch_size * torch.cuda.device_count()

        device_ids = list(range(torch.cuda.device_count()))
        self.model = nn.DataParallel(self.model, device_ids=device_ids)
        self.model = self.model.cuda()

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

        self.checkpoint_dir = os.path.join(CHECKPOINT_DIR, self.experiment)
        self.load_checkpoint(ckpt_name='model')
        self.log_dir = os.path.join(LOG_DIR, self.experiment)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.tensorboard_logger = SummaryWriter(self.log_dir)

        train_video_files, train_labels, train_video_len_files = dataset_utils.get_train_data()
        if self.n_iterations == 0:
            self.n_iterations = len(train_video_files)
        test_video_files, test_labels, test_video_len_files = dataset_utils.get_test_data()

        if model == 'ir-csn':
            self.train_dataset = train_utils.TrainDataset(video_files=train_video_files, labels=train_labels,
                                                          video_len_files=train_video_len_files,
                                                          resize=train_utils.RESIZE, crop_size=train_utils.CROP_SIZE,
                                                          clip_len=train_utils.CLIP_LEN)

            # evaluation datasets
            self.train_eval_dataset = train_utils.EvalDataset(video_files=train_video_files, labels=train_labels,
                                                              video_len_files=train_video_len_files,
                                                              resize=train_utils.RESIZE,
                                                              crop_size=train_utils.CROP_SIZE,
                                                              clip_len=train_utils.CLIP_LEN,
                                                              n_clips=train_utils.N_EVAL_CLIPS)
            self.test_eval_dataset = train_utils.EvalDataset(video_files=test_video_files, labels=test_labels,
                                                             video_len_files=test_video_len_files,
                                                             resize=train_utils.RESIZE,
                                                             crop_size=train_utils.CROP_SIZE,
                                                             clip_len=train_utils.CLIP_LEN,
                                                             n_clips=train_utils.N_EVAL_CLIPS)
        else:
            raise ValueError('no such model... how did you even get here...')

    def train(self):
        start_epoch = self.epoch
        for i in range(start_epoch, self.max_epochs):
            print('INFO: epoch {0}/{1}'.format(i+1, self.max_epochs))
            self.epoch += 1
            self.train_step()
            self.eval_step()
            self.save_checkpoint(ckpt_name='model')
            self.save_checkpoint(ckpt_name='model-{}'.format(self.epoch))

    def train_step(self):
        print('INFO: training at epoch {}'.format(self.epoch))
        dataloader = tdata.DataLoader(self.train_dataset, batch_size=self.train_batch_size,
                                      collate_fn=self.train_dataset.collate_fn, shuffle=True, num_workers=12,
                                      drop_last=True)

        i = 0
        epoch_losses = []
        self.model.train()
        pbar = tqdm(dataloader)
        for frames, labels in pbar:

            frames = frames.cuda()
            labels = labels.cuda()
            self.optimizer.zero_grad()
            logits = self.model(frames)

            loss = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()
            i += 1

            loss = loss.item()
            pbar.set_postfix({'loss:': loss})
            epoch_losses.append(loss)
        epoch_loss = np.mean(epoch_losses)
        print('INFO: training loss: {}'.format(epoch_loss))

        for loss in epoch_losses:
            train_log = {
                'loss': loss
            }
            self.iteration += 1
            self.tensorboard_logger.add_scalars('{}:train'.format(self.experiment), train_log, self.iteration)

    def eval_step(self, evaluate_train=True):
        print('INFO: evaluating...')
        self.model.eval()
        test_accuracy = self.eval_test()
        eval_log = {
            'test-accuracy': test_accuracy
        }

        if evaluate_train:
            eval_log['train-accuracy'] = self.eval_train()
        self.tensorboard_logger.add_scalars('{}:evaluation'.format(self.experiment), eval_log, self.epoch)

    def eval_train(self):
        print('INFO: evaluating train dataset...')
        prediction_file = os.path.join(LOG_DIR, 'epoch-{0}-train-prediction.json'.format(self.epoch))
        train_accuracy = self.eval_dataset(self.train_eval_dataset, prediction_file)
        print('INFO: epoch {0} train accuracy: '.format(train_accuracy))
        return train_accuracy

    def eval_test(self):
        print('INFO: evaluating test dataset...')
        prediction_file = os.path.join(LOG_DIR, 'epoch-{0}-test-prediction.json'.format(self.epoch))
        test_accuracy = self.eval_dataset(self.test_eval_dataset, prediction_file)
        print('INFO: epoch {0} test accuracy: '.format(test_accuracy))
        return test_accuracy

    def eval_dataset(self, dataset, prediction_file):
        dataloader = tdata.DataLoader(dataset=dataset, batch_size=self.eval_batch_size, shuffle=False,
                                      num_workers=12, pin_memory=True, collate_fn=dataset.collate_fn)
        prediction_dict = dict()
        for i, video in enumerate(dataset.video_files):
            video_prediction = {
                'label': int(dataset.labels[i]),
                'n_clips': 0,
                'logit': None
            }
            prediction_dict[video] = video_prediction

        with torch.no_grad():
            for clips, clip_files in tqdm(dataloader):
                n_clips = clips.shape[0]
                clips = clips.cuda()
                logits = self.model(clips).detach().cpu()

                # update prediction dict.
                for i in range(n_clips):
                    logit = logits[i]
                    clip_file = clip_files[i]
                    video_prediction = prediction_dict[clip_file]

                    if video_prediction['n_clips'] == 0:
                        video_prediction['logit'] = logit
                        video_prediction['n_clips'] = 1
                    else:  # keep a running average of logits
                        video_prediction['n_clips'] += 1
                        n_clips = video_prediction['n_clips']
                        video_prediction['logit'] = video_prediction['logit'] * ((n_clips - 1) / n_clips) + \
                                                    logit / n_clips
        n_correct = 0
        n_videos = len(dataset.video_files)

        for video, video_dict in prediction_dict.items():
            logit = video_dict['logit']
            label = int(video_dict['label'])
            prediction = int(torch.argmax(logit).item())
            video_dict['prediction'] = prediction
            if label == prediction:
                n_correct += 1
            del video_dict['logit']
        accuracy = n_correct / n_videos
        with open(prediction_file, 'w') as f:
            json.dump(prediction_dict, f)
        return accuracy

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
        self.tensorboard_logger.close()
        del self.model


def _set_deterministic_experiments():
    torch.manual_seed(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(1234)


def _execute_training():
    _set_deterministic_experiments()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', required=True, type=str, help='config filename e.g -c base')
    args = argparser.parse_args()

    log_file = os.path.join(LOG_DIR, args.config + '.txt')
    sys.stdout = CustomLogger(log_file)
    trainer = Trainer(experiment=args.config)
    trainer.train()


def main():
    _execute_training()
    pass


if __name__ == '__main__':
    main()
