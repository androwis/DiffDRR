# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/api/00_drr.ipynb.

# %% ../notebooks/api/00_drr.ipynb 3
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from fastcore.basics import patch

from .siddon import siddon_raycast
from .detector import Detector
from .utils import reshape_subsampled_drr

# %% auto 0
__all__ = ['DRR']

# %% ../notebooks/api/00_drr.ipynb 5
class DRR(nn.Module):
    """Torch module that computes differentiable digitally reconstructed radiographs."""

    def __init__(
        self,
        volume: np.ndarray,  # CT volume
        spacing: np.ndarray,  # Dimensions of voxels in the CT volume
        sdr: float,  # Source-to-detector radius for the C-arm (half of the source-to-detector distance)
        height: int,  # Height of the rendered DRR
        delx: float,  # X-axis pixel size
        width: int
        | None = None,  # Width of the rendered DRR (if not provided, set to `height`)
        dely: float | None = None,  # Y-axis pixel size (if not provided, set to `delx`)
        p_subsample: float | None = None,  # Proportion of pixels to randomly subsample
        reshape: bool = True,  # Return DRR with shape (b, h, w)
        convention: str = "diffdrr",  # Either `diffdrr` or `deepdrr`, order of basis matrix multiplication
        batch_size: int = 1,  # Number of DRRs to generate per forward pass
    ):
        super().__init__()

        params = torch.empty(batch_size, 6)
        self.rotations = nn.Parameter(params[..., :3])
        self.translations = nn.Parameter(params[..., 3:])

        # Initialize the X-ray detector
        width = height if width is None else width
        dely = delx if dely is None else dely
        self.detector = Detector(
            sdr,
            height,
            width,
            delx,
            dely,
            n_subsample=int(height * width * p_subsample)
            if p_subsample is not None
            else None,
            convention=convention,
        )

        # Initialize the volume
        self.register_buffer("spacing", torch.tensor(spacing))
        self.register_buffer("volume", torch.tensor(volume).flip([0]))
        self.reshape = reshape

        # Dummy tensor for device and dtype
        self.register_buffer("dummy", torch.tensor([0.0]))

    def reshape_transform(self, img, batch_size):
        if self.reshape:
            if self.detector.n_subsample is None:
                img = img.view(-1, 1, self.detector.height, self.detector.width)
            else:
                img = reshape_subsampled_drr(img, self.detector, batch_size)
        return img

# %% ../notebooks/api/00_drr.ipynb 7
@patch
def move_carm(self: DRR, rotations: torch.Tensor, translations: torch.Tensor):
    state_dict = self.state_dict()
    state_dict["rotations"].copy_(rotations)
    state_dict["translations"].copy_(translations)

# %% ../notebooks/api/00_drr.ipynb 8
@patch
def forward(self: DRR):
    """Generate DRR with rotations and translations parameters."""
    source, target = self.detector.make_xrays(
        rotations=self.rotations,
        translations=self.translations,
    )
    img = siddon_raycast(source, target, self.volume, self.spacing)
    return self.reshape_transform(img, batch_size=len(self.rotations))
