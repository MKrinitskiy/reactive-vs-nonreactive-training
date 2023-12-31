import numpy as np
import torch
import torchvision
from torch.cuda import device
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .batch import Batch


class Resizer:
    def __init__(self):
        self.resizer = None

    def __call__(self, images, target_size):
        if self.resizer is None or self.resizer.size != target_size:
            self.init_resizer(target_size)

        return self.resizer(images)

    def init_resizer(self, target_size):
        self.resizer = torchvision.transforms.Resize(
            target_size, interpolation=InterpolationMode.BICUBIC)


class Sampler:
    def __init__(self):
        # affine numbers
        self.rotation_angle = self.sampler(10)  # + np.random.choice([0, 180])
        self.scale = self.sampler(0.05, 1)
        self.translation_x = self.sampler(0.05)
        self.translation_y = self.sampler(0.05)

        # flip probabilities
        # self.flip_ud = self.sampler(0.5, 0.5)
        self.flip_lr = self.sampler(0.5, 0.5)

        # gaussian noize additive
        self.noise_scale = self.sampler(0.01, 0.02)

    @staticmethod
    def sampler(delta: float = 1, center: float = 0):
        """get random float from uniform [center - delta, center + delta]"""
        return (np.random.rand() - 0.5) * 2 * delta + center


class Augmenter:
    # normalizer = transforms.Normalize((0,), (1,))

    normalizer = transforms.Normalize((0.383,), (0.210,))

    inv_normalizer = transforms.Compose([
        transforms.Normalize((0.,), (1 / 0.216,)),
        transforms.Normalize((-0.372,), (1.,)),
    ])

    resizer = Resizer()

    @classmethod
    def augment_image(cls, images: torch.Tensor, only_geometry: bool, sampler: Sampler):
        if not only_geometry:
            # add noise
            # add noize without normalization because noize has zero mean
            noise_shape = [int(i * sampler.noise_scale) for i in images.shape[2:]]
            # mean = torch.mean(images, dim=(2, 3))
            std = torch.std(images, dim=(2, 3))
            mean = torch.zeros_like(std, device=images.get_device())
            noise = cls.get_noise(mean, std * 0.1, noise_shape)
            additive = cls.resizer(noise, images.shape[-2:])
            images = images + additive

        min_image_side = min(images.shape[2:])
        images = transforms.functional.affine(images, sampler.rotation_angle,
                                              [sampler.translation_x * min_image_side,
                                               sampler.translation_y * min_image_side],
                                              sampler.scale,
                                              [0, 0],
                                              interpolation=transforms.InterpolationMode.NEAREST
                                                    if only_geometry else transforms.InterpolationMode.BILINEAR
                                              )

        # if sampler.flip_ud > 0.5:
        #     images = torch.flip(images, dims=[2])
        if sampler.flip_lr > 0.5:
            images = torch.flip(images, dims=[3])

        return images

    @classmethod
    def augment_scalar(cls, scalars: torch.Tensor):
        return torch.normal(scalars, scalars.std(dim=0).item() * 0.1)

    @classmethod
    def __call__(cls, batch: Batch):
        sampler = Sampler()
        images = cls.augment_image(batch.images, False, sampler)
        masks = cls.augment_image(batch.masks, True, sampler)
        significant_wave_height = cls.augment_scalar(batch.significant_wave_height)
        return images, masks, significant_wave_height

    @classmethod
    def call(cls, batch: Batch):
        return cls.__call__(batch)

    @staticmethod
    def get_noise(mean: torch.Tensor, std: torch.Tensor, shape):
        assert mean.shape == std.shape
        noises = []
        for mean_i, std_i in zip(mean.reshape(-1), std.reshape(-1)):
            mean_i, std_i = mean_i.item(), std_i.item()
            noise_i = torch.normal(mean_i, std_i, size=(1, *shape)).to(mean.get_device())
            noises.append(noise_i)

        return torch.stack(noises, dim=0)
