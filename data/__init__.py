import os

DATA_DIR = '/mnt/Data/cs5242-dataset'
VIDEO_DIR = os.path.join(DATA_DIR, 'videos')

SPLIT_DIR = os.path.join(DATA_DIR, 'splits')
TEST_SPLIT_FILE = os.path.join(SPLIT_DIR, 'test.split1.bundle')
TRAIN_SPLIT_FILE = os.path.join(SPLIT_DIR, 'train.split1.bundle')
MAPPING_FILE = os.path.join(SPLIT_DIR, 'mapping_bf.txt')

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TRAIN_VID_DIR = os.path.join(TRAIN_DIR, 'videos')
TRAIN_LABEL_DIR = os.path.join(TRAIN_DIR, 'labels')

TEST_DIR = os.path.join(DATA_DIR, 'test')
TEST_VID_DIR = os.path.join(TEST_DIR, 'videos')
TEST_LABEL_DIR = os.path.join(TEST_DIR, 'labels')

VIDEO_EXT = '.avi'
LABEL_EXT = '.avi.labels'