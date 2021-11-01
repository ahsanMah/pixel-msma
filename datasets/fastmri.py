from numpy.core.fromnumeric import squeeze
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
import configs
import tensorflow as tf

IMG_H = 110
IMG_W = 80


def np_build_mask_fn(constant_mask=False):

    # Building mask of random columns to **keep**
    # batch_sz, img_h, img_w, c = x.shape
    img_sz = configs.dataconfig[configs.config_values.dataset]["downsample_size"]
    img_h, img_w, c = [int(x.strip()) for x in img_sz.split(",")]

    # We do *not* want to mask out the middle (low) frequencies
    # Keeping 10% of low freq is equivalent to Scenario-30L in activemri paper
    low_freq_start = int(0.45 * img_w)
    low_freq_end = img_w - int(0.45 * img_w)
    low_freq_cols = np.arange(low_freq_start, low_freq_end)

    high_freq_cols = np.concatenate(
        (np.arange(0, low_freq_start), np.arange(low_freq_end, img_w))
    )

    def apply_random_mask(x):
        np.random.shuffle(high_freq_cols)
        # rand_ratio = np.random.uniform(
        #     low=configs.config_values.min_marginal_ratio,
        #     high=configs.config_values.marginal_ratio,
        #     size=1,
        # )
        rand_ratio = configs.config_values.marginal_ratio
        n_mask_cols = int(rand_ratio * img_w)
        rand_cols = high_freq_cols[:n_mask_cols]

        mask = np.zeros((img_h, img_w, 2), dtype=np.float32)
        mask[:, rand_cols, :] = 1.0
        mask[:, low_freq_cols, :] = 1.0

        # Applying + Appending mask
        x = x * mask
        # x = np.concatenate([x, mask], axis=-1)
        return x, mask

    # Build a single mask for the entire dataset - mainly useful for evaluation
    np.random.shuffle(high_freq_cols)
    n_mask_cols = int(configs.config_values.marginal_ratio * img_w)
    rand_cols = high_freq_cols[:n_mask_cols]

    mask = np.zeros((img_h, img_w, 2), dtype=np.float32)
    mask[:, rand_cols, :] = 1.0
    mask[:, low_freq_cols, :] = 1.0

    def apply_constant_mask(x):
        # Applying the same mask to all samples
        x = x * mask
        # x = np.concatenate([x, mask], axis=-1)
        return x, mask

    if constant_mask:
        print("Using constant mask function...")
        mask_fn = apply_constant_mask
    else:
        mask_fn = apply_random_mask

    return mask_fn


class FastKnee(Dataset):
    def __init__(self, root, partial=False, constant=False):
        super().__init__()
        self.partial = partial
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

        if self.partial:
            self.mask_fn = np_build_mask_fn(constant_mask=constant)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice_id = self.examples[i]
        mask = None

        with h5py.File(fname, "r") as data:
            kspace = data["kspace"][slice_id]
            kspace = np.stack([kspace.real, kspace.imag], axis=-1)
            kspace = torch.from_numpy(kspace)

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
            target = torch.stack([target.real, target.imag])
            target = center_crop(target, (128, 128))
            # print("After crop:", target.shape)
            target = target.permute(1, 2, 0)
            target = target.contiguous()
            target = torch.view_as_complex(target)

            if self.partial:
                # Get kspace of cropped image
                kspace = fftshift(target, dim=(0, 1))
                kspace = fft(kspace, dim=(0, 1))

                # Realign kspace to keep high freq signal in center
                # Note that original fastmri code did not do this...
                kspace = fftshift(kspace, dim=(0, 1))
                kspace = np.stack([kspace.real, kspace.imag], axis=-1)

                # Mask out regions
                kspace, mask = self.mask_fn(kspace)

                # Recompute target image with masked kspace
                kspace = torch.from_numpy(kspace)
                kspace = kspace.contiguous()
                kspace = torch.view_as_complex(kspace)
                target = ifft(kspace, dim=(0, 1), norm="forward")
                target = ifftshift(target, dim=(0, 1))

            target = complex_magnitude(target)

            # Plot images to confirm fft worked
            import matplotlib.pyplot as plt

            t_img = mask[..., 0]
            print(t_img.dtype, t_img.shape)
            plt.imshow(t_img)
            plt.show()
            plt.savefig("mask_c.png")
            exit()

            # Normalize using mean of k-space in training data
            target /= 7.072103529760345e-07
            kspace /= 7.072103529760345e-07
        target = target.unsqueeze(dim=-1)
        return target, mask


class FastKneeTumor(FastKnee):
    def __init__(self, root, partial=False, constant=False):
        super().__init__(root, partial)
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
            target = self.deform(target)

            # center crop and resize
            print(target.shape)
            target = center_crop(target, (128, 128))
            # print("After crop:", target.shape)
            target = target.permute(1, 2, 0)
            target = target.contiguous()
            target = torch.view_as_complex(target)

            if self.partial:
                # Get kspace of cropped image
                kspace = fftshift(target, dim=(0, 1))
                kspace = fft(kspace, dim=(0, 1))

                # Realign kspace to keep high freq signal in center
                # Note that original fastmri code did not do this...
                kspace = fftshift(kspace, dim=(0, 1))
                kspace = np.stack([kspace.real, kspace.imag], axis=-1)

                # Mask out regions
                kspace, mask = self.mask_fn(kspace)

                # Recompute target image with masked kspace
                kspace = torch.from_numpy(kspace)
                kspace = kspace.contiguous()
                kspace = torch.view_as_complex(kspace)
                target = ifft(kspace, dim=(0, 1), norm="forward")
                target = ifftshift(target, dim=(0, 1))

            target = complex_magnitude(target)

            # Plot images to confirm fft worked
            # import matplotlib.pyplot as plt

            # t_img = target
            # print(t_img.dtype, t_img.shape)
            # plt.imshow(t_img)
            # plt.show()
            # plt.savefig("mask_tumor.png")
            # exit()

            # Normalize using mean of k-space in training data
            target /= 7.072103529760345e-07
            kspace /= 7.072103529760345e-07
        target = target.unsqueeze(dim=-1)
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
