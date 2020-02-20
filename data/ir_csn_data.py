import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from data import BaseDataset
from data import CAFFE_INPUT_MEAN, CAFFE_INPUT_STD


class InferenceDataset(BaseDataset):
    def __init__(self, videos, labels, crop_size):
        super(InferenceDataset, self).__init__(videos, labels, crop_size)
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
