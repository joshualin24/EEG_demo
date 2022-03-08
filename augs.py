"""Utility modules of data augmentations and transformations."""


import torch
import torchaudio
import random
from collections.abc import Sequence


class Identity:
    """No augmentation applies."""

    def __init__(self):
        pass
    
    def __call__(self, data: torch.Tensor):
        return data


class DCShift:
    """Add a random constant shift to data."""

    def __init__(self, min_shift: float, max_shift: float):
        self.base = min_shift
        self.scale = max_shift - min_shift

    def __call__(self, data: torch.Tensor):
        """Data must have dimensions `(..., time steps)`."""
        shifts = self.base + self.scale * torch.rand_like(data[..., :1])
        return data + shifts


class AmplitudeScale:
    """Scale data with a randomly selected constant."""

    def __init__(self, min_scale: float, max_scale: float):
        self.base = min_scale
        self.scale = max_scale - min_scale

    def __call__(self, data: torch.Tensor):
        """Data must have dimensions `(..., time steps)`."""
        scales = self.base + self.scale * torch.rand_like(data[..., :1])
        return data * scales


class AdditiveGaussianNoise:
    """Add Gaussian noise with a random standard deviation."""

    def __init__(self, min_sigma: float, max_sigma: float):
        self.base = min_sigma
        self.scale = max_sigma - min_sigma

    def __call__(self, data: torch.Tensor):
        """Data must have dimensions `(..., time steps)`."""
        sigmas = self.base + self.scale * torch.rand_like(data[..., :1])
        return data + torch.randn_like(data) * sigmas


class TimeMasking:
    """Mask data for a randomly selected time segment."""

    def __init__(self, min_len: int, max_len: int, mask_val: float = 0.0):
        self.base = min_len
        self.scale = max_len - min_len
        self.val = mask_val
    
    def __call__(self, data: torch.Tensor):
        """Data must have dimensions `(..., time steps)`."""
        lens = self.base + self.scale * torch.rand_like(data[..., :1])
        starts = (data.size(-1) - lens) * torch.rand_like(data[..., :1])
        ends = starts + lens
        inds = torch.arange(data.size(-1), dtype=data.dtype, device=data.device)
        return data.masked_fill((inds >= starts) & (inds < ends), self.val)


class TimeShift:
    """Shift data along the time axis with a periodic boundary condition."""

    def __init__(self, min_shift: int, max_shift: int):
        self.base = min_shift
        self.scale = max_shift - min_shift
    
    def __call__(self, data: torch.Tensor):
        """Data must have dimensions `(..., time steps)`."""
        shifts = self.base + self.scale * torch.rand_like(data[..., :1])
        shifts = torch.remainder(shifts, data.size(-1))
        # Prepare indices for masking
        inds = torch.arange(data.size(-1), dtype=data.dtype, device=data.device)
        inds_rev = data.size(-1) - 1 - inds
        # Allocate and fill in shifted data
        shifted = torch.empty_like(data)
        shifted[inds < shifts] = data[inds_rev < shifts]
        shifted[inds >= shifts] = data[inds_rev >= shifts]
        return shifted


class BandStopFilter:
    """Filter out data for a frequency band."""

    def __init__(self, min_freq, max_freq, bandwidth, sample_rate):
        self.base = min_freq
        self.scale = max_freq - min_freq
        self.bandwidth = bandwidth
        self.sample_rate = sample_rate

    def __call_(self, data: torch.Tensor):
        """Data must have dimensions `(..., time steps)`."""
        central_freqs = self.base + self.scale * torch.rand_like(data[..., :1])
        return torchaudio.functional.bandreject_biquad(
            data,
            self.sample_rate,
            central_freqs,
            central_freqs / self.bandwidth
        )


class RandomAugmentationPair:
    """Randomly selected pair of augmentations from a given collection."""

    def __init__(self, augmentations: Sequence):
        self.augs = augmentations
    
    def __call__(self, data: torch.Tensor):
        """Return tuple of augmented data."""
        aug_1, aug_2 = random.sample(self.augs, 2)
        return aug_1(data), aug_2(data)


def main():
    """Empty main."""
    return


if __name__ == '__main__':
    main()