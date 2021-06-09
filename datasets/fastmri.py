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

from torch.fft import fft
from torch.fft import ifft


class FastKnee(Dataset):
    def __init__(self, root):
        super().__init__()
        self.examples = []

        files = []
        for fname in list(pathlib.Path(root).iterdir()):
            files.append(fname)

        for volume_i, fname in enumerate(sorted(files[:10])):
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
            #             print("Kspace Tensor:", kspace.shape)
            kspace = ifftshift(kspace, dim=(0, 1))
            target = ifft(kspace, 2)
            target = ifftshift(target, dim=(0, 1))

            # center crop and resize
            target = target.permute(2, 0, 1)
            target = center_crop(target, (128, 128))
            # target = resize(target, (128,128))
            target = target.permute(1, 2, 0)
            kspace = fftshift(target, dim=(0, 1))
            #             print("After shift", kspace.shape)
            kspace = fft(kspace, 2)

            # Normalize using mean of k-space in training data
            target /= 7.072103529760345e-07
            kspace /= 7.072103529760345e-07
            # to pytorch format
            kspace = kspace.permute(2, 0, 1)
            target = target.permute(2, 0, 1)

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
            kspace = ifftshift(kspace, dim=(0, 1))
            target = ifft(kspace, 2, normalized=False)
            target = ifftshift(target, dim=(0, 1))
            # transform
            target = target.permute(2, 0, 1)
            target = self.deform(target)
            target = torch.from_numpy(target)
            # center crop and resize
            target = center_crop(target, (128, 128))
            # target = resize(target, (128,128))
            target = target.permute(1, 2, 0)
            kspace = fftshift(target, dim=(0, 1))
            #             kspace = torch.fft(kspace, 2, normalized=False)
            kspace = fft(kspace, 2)
            # Normalize using mean of k-space in training data
            target /= 7.072103529760345e-07
            kspace /= 7.072103529760345e-07
            # to pytorch format
            kspace = kspace.permute(2, 0, 1)
            target = target.permute(2, 0, 1)

        return kspace, target


if __name__ == "__main__":
    dataset = FastKnee("/home/PO3D/raw_data/knee/singlecoil_val")
    ksp, tar = dataset[20]
    print(ksp.shape, tar.shape)
    import matplotlib.pyplot as plt

    img = to_magnitude(tar, dim=0)
    plt.imsave("normal.png", img)

    dataset = FastKneeTumor("/home/PO3D/raw_data/knee/singlecoil_val/")
    ksp, tar = dataset[20]
    print(ksp.shape, tar.shape)
    import matplotlib.pyplot as plt

    img = to_magnitude(tar, dim=0)
    plt.imsave("tumor.png", img)
