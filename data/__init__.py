import os
import cv2
import numpy as np

DATA_DIR = '/mnt/Data/cs5242-dataset'
VIDEO_DIR = os.path.join(DATA_DIR, 'videos')

SPLIT_DIR = os.path.join(DATA_DIR, 'splits')
TEST_SPLIT_FILE = os.path.join(SPLIT_DIR, 'test.split1.bundle')
TRAIN_SPLIT_FILE = os.path.join(SPLIT_DIR, 'train.split1.bundle')
OLD_MAPPING_FILE = os.path.join(SPLIT_DIR, 'mapping_bf.txt')
MAPPING_FILE = os.path.join(SPLIT_DIR, 'mapping.txt')

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TRAIN_VID_DIR = os.path.join(TRAIN_DIR, 'videos')
TRAIN_LABEL_DIR = os.path.join(TRAIN_DIR, 'labels')
TRAIN_SEGMENT_DIR = os.path.join(TRAIN_DIR, 'segments')
TRAIN_SEGMENT_DICT = os.path.join(TRAIN_DIR, 'segments-dict.json')
BAD_TRAIN_SEGMENTS_FILE = os.path.join(TRAIN_DIR, 'bad-segments.txt')

TEST_DIR = os.path.join(DATA_DIR, 'test')
TEST_VID_DIR = os.path.join(TEST_DIR, 'videos')
TEST_LABEL_DIR = os.path.join(TEST_DIR, 'labels')
TEST_SEGMENT_DIR = os.path.join(TEST_DIR, 'segments')
TEST_SEGMENT_DICT = os.path.join(TEST_DIR, 'segments-dict.json')
BAD_TEST_SEGMENTS_FILE = os.path.join(TEST_DIR, 'bad-segments.txt')

VIDEO_EXT = '.avi'
LABEL_EXT = '.avi.labels'


def read_label_file(label_file):
    with open(label_file, 'r') as f:
        labels = f.readlines()
    labels = [label.strip() for label in labels]

    labels = [label.split(' ') for label in labels]
    segment_windows = [np.array(label[0].split('-'), dtype=int) for label in labels]
    segment_labels = [str(label[1]) for label in labels]
    return segment_windows, segment_labels


def read_mapping_file(mapping_file):
    with open(mapping_file, 'r') as f:
        content = f.readlines()
    content = [line.strip().split(' ') for line in content]

    logit_to_action_dict = np.array([line[1] for line in content], dtype=str)
    action_to_logit_dict = dict()
    for i, action in enumerate(logit_to_action_dict):
        action_to_logit_dict[action] = i
    return action_to_logit_dict, logit_to_action_dict


def read_segment_from_video(vid_file, window=None):
    # the frames are in one-based indexing.
    return_all_frames = window is None
    if not return_all_frames:
        assert len(window) == 2

    frames = list()
    cap = cv2.VideoCapture(vid_file)

    i = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if return_all_frames or (window[0] <= i < window[1]):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        else:
            break
        i += 1

        if (not return_all_frames) and (i >= window[1]):
            break
    cap.release()
    return frames

