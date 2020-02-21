import os
import cv2
import numpy as np
from PIL import Image
import torch.utils.data as tdata
from torch.utils.data._utils.collate import default_collate

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

# https://github.com/pytorch/pytorch/blob/master/caffe2/video/video_input_op.h#L374-L378
CAFFE_INPUT_MEAN = [110.201, 100.64, 95.9966]
CAFFE_INPUT_STD = [58.1489, 56.4701, 55.3324]


class BaseDataset(tdata.Dataset):
    def __init__(self, videos, labels, crop_size):
        super(BaseDataset, self).__init__()
        assert len(videos) == len(labels)
        self.videos = np.array(videos).astype(str)
        self.labels = np.array(labels).astype(int)
        self.crop_size = int(crop_size)

        # check that all the videos exists
        for video in self.videos:
            assert os.path.exists(video)

    def __len__(self):
        return len(self.videos)

    @staticmethod
    def collate_fn(batch):
        video_frames, labels = zip(*batch)
        video_frames = default_collate(video_frames)
        labels = default_collate(labels)
        return video_frames, labels


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


def get_n_video_frames(video_file):
    """ returns the number of video frames in a video file """
    cap = cv2.VideoCapture(video_file)
    n_frames = 0
    while cap.isOpened():
        ret, _ = cap.read()
        if ret:
            n_frames += 1
        else:
            break
    cap.release()
    return n_frames


def sample_video_clips(video_file, n_frames, idxs):
    """ returns the video frames from at idxs... """
    cap = cv2.VideoCapture(video_file)
    sample_idxs = np.unique(idxs)
    assert np.all(sample_idxs < n_frames)

    frame_dict = dict()
    max_sample = np.max(sample_idxs)
    n_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if n_frames in sample_idxs:
                frame_dict[n_frames] = frame
            n_frames += 1
        else:
            break

        if max_sample < n_frames:
            break  # complete reading
    cap.release()

    frames = []
    for idx in idxs:
        frames.append(frame_dict[idx])
    return frames


def resize_clip(clip, desired_size):
    clip = [np.array(frame) for frame in clip]
    clip = [Image.fromarray(frame) for frame in clip]
    start_frame = clip[0]
    min_size = min(start_frame.size)
    ratio = desired_size / min_size
    final_size = (np.array(start_frame.size) * ratio).astype(int)

    resized_clip = [frame.resize(final_size, resample=Image.BICUBIC) for frame in clip]
    resized_clip = [np.array(frame) for frame in resized_clip]
    return resized_clip
