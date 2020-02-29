import numpy as np
from torchvision import transforms
from data import videotransforms
from data import BaseDataset
from data import TORCH_MEAN, TORCH_STD
from tqdm import tqdm
from data import get_n_video_frames, sample_video_clip

RESIZE = 256
CROP_SIZE = 256
CLIP_LEN = 32
N_EVAL_CLIPS = 10


class TrainDataset(BaseDataset):
    def __init__(self, videos, labels, resize, crop_size, clip_len):
        super(TrainDataset, self).__init__(videos, labels, resize, crop_size)
        self.clip_len = int(clip_len)

        self.transforms = transforms.Compose([
            videotransforms.BgrToRgbClip(),
            videotransforms.ToImageClip(),
            videotransforms.AspectPreservingResizeClip(resize=self.resize),
            videotransforms.CenterCropClip(self.crop_size),
            videotransforms.ClipToTensor(),
            videotransforms.NormalizeClip(TORCH_MEAN, TORCH_STD),
            videotransforms.To3dTensor()
        ])

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        label = self.labels[idx]

        n_frames = get_n_video_frames(video_file)
        sample_idxs = get_video_sample_idxs(n_frames, self.clip_len)
        clip = sample_video_clip(video_file, n_frames, sample_idxs)
        clip = np.array(clip)
        clip = self.transforms(clip)
        return clip, label


class EvalDataset(BaseDataset):
    def __init__(self, videos, labels, resize, crop_size, clip_len, n_clips):
        super(EvalDataset, self).__init__(videos, labels, resize, crop_size)
        self.input_clip_len = int(clip_len)
        self.n_clips = int(n_clips)
        self.transforms = transforms.Compose([
            videotransforms.BgrToRgbClip(),
            videotransforms.ToImageClip(),
            videotransforms.AspectPreservingResizeClip(resize=self.resize),
            videotransforms.CenterCropClip(self.crop_size),
            videotransforms.ClipToTensor(),
            videotransforms.NormalizeClip(TORCH_MEAN, TORCH_STD),
            videotransforms.To3dTensor()
        ])

        print('INFO: retrieving number of frames for each video...')
        video_lens = []
        for video_file in tqdm(self.video_files):
            video_lens.append(get_n_video_frames(video_file=video_file))
        video_lens = np.array(video_lens)

        print('INFO: creating clip samples for each video...')
        clip_video_lens = []
        clip_start_frames = []
        clip_files = []
        for i, video_file in enumerate(tqdm(self.video_files)):
            video_len = video_lens[i]
            if video_len <= 2 * self.input_clip_len:
                clip_files.append(video_file)
                clip_start_frames.append(0)
                clip_video_lens.append(video_len)
            else:  # video file is long enough to get multiple clips
                n_start_frames = video_len - self.input_clip_len
                if n_start_frames < self.n_clips:
                    clip_start_frames += list(np.arange(n_start_frames).reshape(-1))
                    clip_files += [video_file] * n_start_frames
                    clip_video_lens += [video_len] * n_start_frames
                else:
                    start_frames = np.arange(n_start_frames)
                    sample_idxs = []
                    for fi, start_frame in enumerate(start_frames):
                        if (fi / n_start_frames) >= (len(sample_idxs) / self.n_clips):
                            sample_idxs.append(fi)
                    clip_start_frames += list(start_frames[sample_idxs])
                    clip_files += [video_file] * self.n_clips
                    clip_video_lens += [video_len] * self.n_clips
        assert len(clip_start_frames) == len(clip_files) == len(clip_video_lens)

        self.clip_files = np.array(clip_files)
        self.clip_start_frames = np.array(clip_start_frames)
        self.clip_video_lens = np.array(clip_video_lens)

    def __len__(self):
        return len(self.clip_files)

    def __getitem__(self, idx):
        clip_file = self.clip_files[idx]
        clip_video_len = self.clip_video_lens[idx]
        clip_start_frame = self.clip_video_lens[idx]

        clip_len = clip_video_len - clip_start_frame
        sample_idxs = get_video_sample_idxs(clip_len, self.input_clip_len)
        sample_idxs = np.array(sample_idxs) + clip_start_frame

        clip = sample_video_clip(clip_file, clip_video_len, sample_idxs)
        clip = self.transforms(clip)
        return clip, clip_file


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
