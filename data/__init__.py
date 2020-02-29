import os
import cv2
import numpy as np
from PIL import Image
import torch.utils.data as tdata
from torch.utils.data._utils.collate import default_collate


DATA_DIR = '/mnt/Data/cs5242-datasets' if os.path.exists('/mnt/Data/cs5242-datasets') else \
    '/home/e/e0036319/data/cs5242-datasets'

# https://github.com/pytorch/pytorch/blob/master/caffe2/video/video_input_op.h#L374-L378
CAFFE_INPUT_MEAN = [110.201, 100.64, 95.9966]
CAFFE_INPUT_STD = [58.1489, 56.4701, 55.3324]

TORCH_MEAN = (0.485, 0.456, 0.406)
TORCH_STD = (0.229, 0.224, 0.225)


class BaseDataset(tdata.Dataset):
    def __init__(self, video_files, video_len_files, labels, resize, crop_size):
        super(BaseDataset, self).__init__()
        assert len(video_files) == len(video_len_files) == len(labels)
        self.video_files = np.array(video_files).astype(str)
        self.video_len_files = np.array(video_len_files).astype(str)
        self.labels = np.array(labels).astype(int)
        self.resize = resize
        self.crop_size = int(crop_size)

        # check that all the videos exists
        for file in self.video_files:
            assert os.path.exists(file), file
        for file in self.video_len_files:
            assert os.path.exists(file), file

    def __len__(self):
        return len(self.video_files)

    @staticmethod
    def collate_fn(batch):
        video_frames, labels = zip(*batch)
        video_frames = default_collate(video_frames)
        labels = default_collate(labels)
        return video_frames, labels


def get_video_fps(video_file):
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps


def write_video_file(frames, fps, save_file):
    start_frame = frames[0]
    frame_height, frame_width, _ = start_frame.shape
    out = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))
    for frame in frames:
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()


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


def sample_video_clip(video_file, n_frames, idxs):
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


def aspect_preserving_clip_resize(clip, desired_size):
    clip = [np.array(frame) for frame in clip]
    clip = [Image.fromarray(frame) for frame in clip]
    start_frame = clip[0]
    min_size = min(start_frame.size)
    ratio = desired_size / min_size
    final_size = (np.array(start_frame.size) * ratio).astype(int)

    resized_clip = [frame.resize(final_size, resample=Image.BICUBIC) for frame in clip]
    resized_clip = [np.array(frame) for frame in resized_clip]
    return resized_clip
