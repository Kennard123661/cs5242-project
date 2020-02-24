import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image


#todo: add random crops
class BgrToRgbClip:
    def __call__(self, clip):
        # assert clip[0] == np.ndarray
        return [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in clip]


class ToImageClip:
    def __call__(self, clip):
        return [Image.fromarray(frame) for frame in clip]


class AspectPreservingResizeClip:
    def __init__(self, resize, interpolation=Image.BICUBIC):
        self.resize = resize
        self.interpolation = interpolation

    def __call__(self, clip):
        start_frame = clip[0]
        min_size = min(start_frame.size)
        ratio = self.resize / min_size
        final_size = (np.array(start_frame.size) * ratio).astype(int)
        return [F.resize(frame, size=final_size, interpolation=self.interpolation) for frame in clip]


class ResizeClip:
    def __init__(self, size):
        self.resize = transforms.Resize(size=size, interpolation=Image.BICUBIC)

    def __call__(self, clip):
        return [self.resize(frame) for frame in clip]


class CenterCropClip:
    def __init__(self, crop_size):
        self.crop = transforms.CenterCrop(crop_size)

    def __call__(self, clip):
        return [self.crop(frame) for frame in clip]


class ClipToTensor:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, clip):
        return [self.to_tensor(frame) for frame in clip]


class NormalizeClip:
    def __init__(self, mean, std, inplace=False):
        self.normalize = transforms.Normalize(mean, std, inplace=inplace)

    def __call__(self, clip):
        return [self.normalize(frame) for frame in clip]


class To3dTensor:
    def __call__(self, clip):
        clip = torch.stack(clip)  # T x C x H x W
        return clip.transpose(1, 0)  # C x T x H x W
