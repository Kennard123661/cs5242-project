import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import cv2
from data import BaseDataset
from data import CAFFE_INPUT_MEAN, CAFFE_INPUT_STD
from data import get_n_video_frames, sample_video_clips, resize_clip
import data.kinetics_data as kinetics

CROP_SIZE = 256
CLIP_LEN = 32


class InferenceDataset(BaseDataset):
    def __init__(self, videos, labels, crop_size, n_clips, clip_len):
        super(InferenceDataset, self).__init__(videos, labels, crop_size)
        self.n_clips = int(n_clips)
        self.clip_len = int(clip_len)
        self.transform_frame = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.CenterCrop(crop_size),
            lambda x: np.array(x),
            lambda x: torch.from_numpy(x),
            transforms.Normalize(mean=CAFFE_INPUT_MEAN, std=CAFFE_INPUT_STD),
        ])

    def __getitem__(self, idx):
        video_file = self.videos[idx]
        label = self.labels[idx]
        is_extracted = video_file == '.npy'
        if is_extracted:
            video_frames = np.load(video_file)
        else:
            raise NotImplementedError
        video_frames = [self.transform_frame(frame) for frame in video_frames]
        return video_frames, label


def generate_inference_crops(video_files, crop_size, n_clips, clip_len, clip_dir):
    if not os.path.exists(clip_dir):
        os.makedirs(clip_dir)

    frame_transforms = transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.CenterCrop(crop_size),
        lambda x: np.array(x).transpose([2, 0, 1]).astype(np.float32),
        lambda x: torch.from_numpy(x),
        transforms.Normalize(mean=CAFFE_INPUT_MEAN, std=CAFFE_INPUT_STD),
        lambda x: x.numpy()
    ])

    pbar = tqdm(video_files)
    for i, file in enumerate(pbar):
        n_frames = get_n_video_frames(file)
        if n_frames < clip_len * 2:
            continue
        else:
            n_start_frames = n_frames - clip_len * 2
        start_frames = np.arange(n_start_frames)
        for j in range(n_clips):
            start_frame = np.random.choice(start_frames)
            end_frame = start_frame + (clip_len * 2)

            clip_idxs = np.arange(start_frame, end_frame, step=2)
            clip = sample_video_clips(file, n_frames, clip_idxs)
            clip = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in clip]
            clip = kinetics.resize_clip(clip)
            clip = np.array([frame_transforms(frame) for frame in clip]).transpose([1, 0, 2, 3])

            video_fn = os.path.split(file)[-1]
            save_file = os.path.join(clip_dir, '{}.{}'.format(j, video_fn))
            np.save(save_file, clip)


def main():
    video_files, _ = kinetics.get_train_data()
    n_clips = 30
    n_vids = 1
    generate_inference_crops(video_files[:n_vids], crop_size=CROP_SIZE, n_clips=n_clips,
                             clip_len=CLIP_LEN, clip_dir=kinetics.TRAIN_CLIP_DIR)


if __name__ == '__main__':
    main()