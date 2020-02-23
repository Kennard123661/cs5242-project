import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from data import videotransforms
from data import BaseDataset
from data import CAFFE_INPUT_MEAN, CAFFE_INPUT_STD
from data import TORCH_MEAN, TORCH_STD
from data import get_n_video_frames, sample_video_clip

CROP_SIZE = 256
CLIP_LEN = 32


class TrainDataset(BaseDataset):
    def __init__(self, videos, labels, resize, crop_size, clip_len):
        super(TrainDataset, self).__init__(videos, labels, resize, crop_size)
        self.clip_len = int(clip_len)

        self.transforms = transforms.Compose([
            videotransforms.BgrToRgbClip(),
            videotransforms.ToImageClip(),
            videotransforms.AspectPreservingResizeClip(resize=self.resize),
            videotransforms.CenterCropClip(self.crop_size),
            videotransforms.NormalizeClip(TORCH_MEAN, TORCH_STD),
            videotransforms.To3dTensor()
        ])

    def __getitem__(self, idx):
        video = self.videos[idx]
        label = self.labels[idx]

        n_frames = get_n_video_frames(video)
        sample_idxs = get_video_sample_idxs(n_frames, self.clip_len)
        clip = sample_video_clip(video, n_frames, sample_idxs)
        clip = self.transforms(clip)
        return clip, label


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


def get_video_sample_idxs(n_frames, n_samples):
    if n_frames < n_samples:  # there are less frames than samples needed
        frame_idxs = list(range(n_frames))
        n_dups = n_samples // n_frames
        n_excess = n_samples - n_dups * n_frames
        if n_excess > 0:
            sampled_idxs = frame_idxs[:n_excess] * (n_dups + 1) + frame_idxs[n_excess:] * n_dups
        else:
            sampled_idxs = frame_idxs * n_dups
    elif n_frames == n_samples:
        sampled_idxs = list(range(n_frames))
    elif n_frames < 2 * n_samples:
        si = 0
        sampled_idxs = np.empty(shape=n_samples, dtype=int)
        for fi in range(n_frames):
            if (fi / n_frames) >= (si / n_samples):
                sampled_idxs[si] = fi
                si += 1

            if si == n_samples:
                break
    else:
        n_start_frames = n_frames - 2 * n_samples
        start_frames = np.arange(n_start_frames)
        start_frame = np.random.choice(start_frames)
        end_frame = start_frame + 2 * n_samples
        sampled_idxs = np.arange(start_frame, end_frame, 2)
    assert len(sampled_idxs) == n_samples
    return np.sort(sampled_idxs).reshape(-1)


def main():
    print(get_video_sample_idxs(n_frames=100, n_samples=30))
    pass


if __name__ == '__main__':
    main()
