import random
import torch
from typing import List, Union
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from torchvision.transforms import ColorJitter as CJ

class GaussNoise(object):
    def __call__(self, image, target=None):
        scale = random.uniform(0, 0.2)
        noise = scale * torch.randn(image.shape)
        image = image + noise

        return image, target


class ColorJitter_Solid(object):
    def __init__(self):
        self.cj = CJ(brightness=0.6, contrast=0.7, saturation=0.5, hue=0.1)

    def __call__(self, image, target):
        image = self.cj(image)
        return image, target

class ColorJitter_Soft(object):
    def __init__(self):
        bn = random.uniform(0.2, 0.8)
        ct = random.uniform(0.2, 0.8)
        sr = random.uniform(0.2, 0.8)
        hu = random.uniform(-0.5, 0.5)
        self.cj = CJ(brightness=bn, contrast=ct, saturation=sr, hue=hu)

    def __call__(self, image, target):
        image = self.cj(image)
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = F.to_tensor(target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.flip_prob = prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
    

class Resize(object):
    def __init__(self, size: Union[int, List[int]], resize_mask: bool = True):
        self.size = size  # [h, w]
        self.resize_mask = resize_mask

    def __call__(self, image, target=None):
        image = F.resize(image, self.size)
        if self.resize_mask is True:
            target = F.resize(target, self.size)

        return image, target


class RandomCrop(object):
    def __init__(self, size: int):
        self.size = size

    def pad_if_smaller(self, img, fill=0):
        # 如果图像最小边长小于给定size，则用数值fill进行padding
        min_size = min(img.shape[-2:])
        if min_size < self.size:
            ow, oh = img.size
            padh = self.size - oh if oh < self.size else 0
            padw = self.size - ow if ow < self.size else 0
            img = F.pad(img, [0, 0, padw, padh], fill=fill)
        return img

    def __call__(self, image, target):
        image = self.pad_if_smaller(image)
        target = self.pad_if_smaller(target)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target
