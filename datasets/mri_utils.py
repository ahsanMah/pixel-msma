import torch
import numpy as np
from torch.nn.functional import interpolate

from monai.config import USE_COMPILED, DtypeLike
from monai.data.utils import compute_shape_offset, to_affine_nd, zoom_affine
from monai.networks.layers import AffineTransform, GaussianFilter, grid_pull
from monai.transforms.croppad.array import CenterSpatialCrop
from monai.transforms.transform import Randomizable, RandomizableTransform, Transform
from monai.transforms.utils import (
    create_control_grid,
    create_grid,
    create_rotate,
    create_scale,
    create_shear,
    create_translate,
    map_spatial_axes,
)
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    fall_back_tuple,
    issequenceiterable,
    optional_import,
)

from monai.transforms.spatial.array import *
from typing import *
from monai.transforms import *

def to_magnitude(tensor, dim):
    return (tensor ** 2).sum(dim=dim) ** 0.5

def roll_one_dim(x, shift, dim):
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)

def roll(x, shift, dim):
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x

def fftshift(x, dim):
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)

def ifftshift(x, dim):
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)

def center_crop(img, output_size):
    _, image_height, image_width = img.shape
    crop_height, crop_width = output_size
    top = int((image_height - crop_height + 1) * 0.5)
    left = int((image_width - crop_width + 1) * 0.5)
    right = left + crop_width
    bottom = top + crop_height
    return img[..., top:bottom, left:right]

def resize(img, output_size):
    img = img.unsqueeze(0)
    img = interpolate(img, output_size, mode="bilinear", align_corners=False)
    img = img.squeeze(0)
    return img

class TumorGrowthGrid(RandDeformGrid):
    def __init__(
        self,
        spacing: Union[Sequence[float], float],
        max_tumor_size: float,
        magnitude_range: Tuple[float, float],
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super(TumorGrowthGrid, self).__init__(spacing, magnitude_range, as_tensor_output, device)
        self.max_tumor_size = max_tumor_size

    def randomize(self) -> None:
        h1, w1 = self.R.randint(-self.max_tumor_size, self.max_tumor_size, size=2)
        h2, w2 = self.R.randint(-self.max_tumor_size, self.max_tumor_size, size=2)
        self.center1 = np.array([h1,w1,1], dtype=np.float32).reshape(3,1,1)
        self.center2 = np.array([h2,w2,1], dtype=np.float32).reshape(3,1,1)
        self.rand_mag = self.R.uniform(self.magnitude[0], self.magnitude[1])

    def __call__(self, spatial_size: Sequence[int]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            spatial_size: spatial size of the grid.
        """
        self.spacing = fall_back_tuple(self.spacing, (1.0,) * len(spatial_size))
        control_grid = create_control_grid(spatial_size, self.spacing)
        self.randomize()
        dist1 = np.sqrt(np.sum(np.square(control_grid - self.center1)[:2], axis=0) + 1e-6)
        dist2 = np.sqrt(np.sum(np.square(control_grid - self.center2)[:2], axis=0) + 1e-6)
        deform_mag = self.rand_mag / (dist1 + dist2)
        center = (self.center1 + self.center2) / 2.
        deform = (control_grid - center)[:2] * deform_mag
        control_grid[:len(spatial_size)] -= deform
        if self.as_tensor_output:
            control_grid = torch.as_tensor(np.ascontiguousarray(control_grid), device=self.device)
        return control_grid

class RandTumor(Randomizable, Transform):
    def __init__(
        self,
        spacing: Union[Tuple[float, float], float],
        max_tumor_size: float,
        magnitude_range: Tuple[float, float],
        prob: float = 0.1,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.REFLECTION,
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        self.deform_grid = TumorGrowthGrid(
            spacing=spacing, max_tumor_size=max_tumor_size, magnitude_range=magnitude_range, 
            as_tensor_output=True, device=device
        )
        self.resampler = Resample(as_tensor_output=as_tensor_output, device=device)

        self.spatial_size = spatial_size
        self.mode: GridSampleMode = GridSampleMode(mode)
        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)
        self.prob = prob
        self.do_transform = False

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ):
        self.deform_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self) -> None:
        self.do_transform = self.R.rand() < self.prob
        self.deform_grid.randomize()

    def __call__(
        self,
        img: Union[np.ndarray, torch.Tensor],
        spatial_size: Optional[Union[Tuple[int, int], int]] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        sp_size = fall_back_tuple(spatial_size or self.spatial_size, img.shape[1:])
        self.randomize()
        if self.do_transform:
            grid = self.deform_grid(spatial_size=sp_size)
            grid = torch.nn.functional.interpolate(
                input=grid.unsqueeze(0),
                scale_factor=list(ensure_tuple(self.deform_grid.spacing)),
                mode=InterpolateMode.BICUBIC.value,
                align_corners=False,
            )
            grid = CenterSpatialCrop(roi_size=sp_size)(grid[0])
        else:
            grid = create_grid(spatial_size=sp_size)
        return self.resampler(img, grid, mode=mode or self.mode, padding_mode=padding_mode or self.padding_mode)
