import torch
import pathlib
import h5py
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from .mri_utils import *

import os
import torch
from torch.utils.data.dataset import TensorDataset
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
import pickle
import copy
import argparse

from torch.fft import fft2 as fft
from torch.fft import ifft2 as ifft
from torch.fft import fftshift, ifftshift

import tensorflow as tf

IMG_H = 110
IMG_W = 80


class FastKnee(Dataset):
    def __init__(self, root):
        super().__init__()
        self.examples = []

        files = []
        for fname in list(pathlib.Path(root).iterdir()):
            files.append(fname)

        for volume_i, fname in enumerate(sorted(files)):
            data = h5py.File(fname, "r")
            kspace = data["kspace"]
            num_slices = kspace.shape[0]
            self.examples += [
                (fname, slice_id)
                for slice_id in range(num_slices // 4, num_slices // 4 * 3)
            ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice_id = self.examples[i]
        with h5py.File(fname, "r") as data:
            kspace = data["kspace"][slice_id]
            kspace = torch.from_numpy(np.stack([kspace.real, kspace.imag], axis=-1))

            # For 1.8+
            # pytorch now offers a complex64 data type
            kspace = torch.view_as_complex(kspace)
            kspace = ifftshift(kspace, dim=(0, 1))
            # norm=forward means no normalization
            target = ifft(kspace, dim=(0, 1), norm="forward")
            target = ifftshift(target, dim=(0, 1))

            # Plot images to confirm fft worked
            # t_img = complex_magnitude(target)
            # print(t_img.dtype, t_img.shape)
            # plt.imshow(t_img)
            # plt.show()
            # plt.imshow(target.real)
            # plt.show()

            # center crop and resize
            # target = torch.unsqueeze(target, dim=0)
            # target = center_crop(target, (128, 128))
            # target = torch.squeeze(target)

            # Crop out ends
            target = np.stack([target.real, target.imag], axis=-1)
            target = target[100:-100, 24:-24, :]

            # Downsample in image space
            shape = target.shape
            target = tf.image.resize(
                target,
                (IMG_H, IMG_W),
                method="lanczos5",
                # preserve_aspect_ratio=True,
                antialias=True,
            ).numpy()

            # Get kspace of cropped image
            target = torch.view_as_complex(torch.from_numpy(target))
            kspace = fftshift(target, dim=(0, 1))
            kspace = fft(kspace, dim=(0, 1))
            # Realign kspace to keep high freq signal in center
            # Note that original fastmri code did not do this...
            kspace = fftshift(kspace, dim=(0, 1))

            # Normalize using mean of k-space in training data
            target /= 7.072103529760345e-07
            kspace /= 7.072103529760345e-07

        return kspace, target


class FastKneeTumor(FastKnee):
    def __init__(self, root):
        super().__init__(root)
        self.deform = RandTumor(
            spacing=30.0,
            max_tumor_size=50.0,
            magnitude_range=(50.0, 150.0),
            prob=1.0,
            spatial_size=[640, 368],
            padding_mode="zeros",
        )
        self.deform.set_random_state(seed=0)

    def __getitem__(self, i):
        fname, slice_id = self.examples[i]
        with h5py.File(fname, "r") as data:
            kspace = data["kspace"][slice_id]
            kspace = torch.from_numpy(np.stack([kspace.real, kspace.imag], axis=-1))
            kspace = torch.view_as_complex(kspace)
            kspace = ifftshift(kspace, dim=(0, 1))
            target = ifft(kspace, dim=(0, 1), norm="forward")
            target = ifftshift(target, dim=(0, 1))

            # transform
            target = torch.stack([target.real, target.imag])
            target = self.deform(target)  # outputs numpy
            target = torch.from_numpy(target)
            target = target.permute(1, 2, 0).contiguous()
            # center crop and resize
            # target = center_crop(target, (128, 128))
            # target = resize(target, (128,128))

            # Crop out ends
            target = target.numpy()[100:-100, 24:-24, :]
            # Downsample in image space
            target = tf.image.resize(
                target,
                (IMG_H, IMG_W),
                method="lanczos5",
                # preserve_aspect_ratio=True,
                antialias=True,
            ).numpy()

            # Making contiguous is necessary for complex view
            target = torch.from_numpy(target)
            target = target.contiguous()
            target = torch.view_as_complex(target)

            kspace = fftshift(target, dim=(0, 1))
            kspace = fft(kspace, dim=(0, 1))
            kspace = fftshift(kspace, dim=(0, 1))

            # Normalize using mean of k-space in training data
            target /= 7.072103529760345e-07
            kspace /= 7.072103529760345e-07

        return kspace, target


if __name__ == "__main__":
    dataset = FastKnee("/home/PO3D/raw_data/knee/singlecoil_val")
    ksp, tar = dataset[20]
    print(ksp.shape, tar.shape)
    import matplotlib.pyplot as plt

    img = complex_magnitude(tar)
    plt.imsave("normal.png", img)

    dataset = FastKneeTumor("/home/PO3D/raw_data/knee/singlecoil_val/")
    ksp, tar = dataset[20]
    print(ksp.shape, tar.shape)
    import matplotlib.pyplot as plt

    img = complex_magnitude(tar)
    plt.imsave("tumor.png", img)
