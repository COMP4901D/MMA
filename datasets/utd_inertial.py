"""
UTD-MHAD Inertial Dataset — Pure 6-axis IMU.

Loads accelerometer + gyroscope data from .mat files.
File: a{action}_s{subject}_t{trial}_inertial.mat
Variable: 'd_iner' -> shape (num_timesteps, 6)
"""

import os

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


# ================================================================
#  Time-series augmentation helpers
# ================================================================

def _jitter(x: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """Add Gaussian noise."""
    return x + np.random.normal(0, sigma, x.shape).astype(np.float32)


def _scaling(x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """Per-channel random scaling."""
    factor = np.random.normal(1.0, sigma, (1, x.shape[1])).astype(np.float32)
    return x * factor


def _rotation(x: np.ndarray) -> np.ndarray:
    """Small random 3D rotation applied to acc (0:3) and gyro (3:6) separately."""
    angle = np.random.uniform(-0.2, 0.2, 3)
    cx, sx = np.cos(angle[0]), np.sin(angle[0])
    cy, sy = np.cos(angle[1]), np.sin(angle[1])
    cz, sz = np.cos(angle[2]), np.sin(angle[2])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    R = Rz @ Ry @ Rx
    out = x.copy()
    out[:, :3] = x[:, :3] @ R.T
    if x.shape[1] >= 6:
        out[:, 3:6] = x[:, 3:6] @ R.T
    return out


def _permutation(x: np.ndarray, n_segments: int = 5) -> np.ndarray:
    """Split into segments and permute their order."""
    T = x.shape[0]
    if T < n_segments:
        return x
    splits = np.array_split(np.arange(T), n_segments)
    np.random.shuffle(splits)
    return x[np.concatenate(splits)]


def _time_warp(x: np.ndarray, sigma: float = 0.2) -> np.ndarray:
    """Smooth time-warping via random cumulative distortion."""
    T = x.shape[0]
    if T < 4:
        return x
    warp = np.cumsum(np.random.normal(1.0, sigma, T))
    warp = (warp - warp[0]) / (warp[-1] - warp[0]) * (T - 1)
    warp = np.clip(warp, 0, T - 1)
    indices = np.round(warp).astype(int)
    return x[indices]


class UTDMADInertialDataset(Dataset):
    """
    Loads 6-axis IMU data from UTD-MAD .mat files.
    Pads/truncates to a fixed sequence length and normalises channel-wise.
    """

    def __init__(
        self,
        data_dir: str,
        subjects: list,
        max_len: int = 256,
        mean: np.ndarray = None,
        std: np.ndarray = None,
        augment: bool = False,
    ):
        self.max_len = max_len
        self.augment = augment
        self.samples: list = []
        self.labels: list = []

        for action in range(1, 28):
            for subj in subjects:
                for trial in range(1, 5):
                    fn = f"a{action}_s{subj}_t{trial}_inertial.mat"
                    fp = os.path.join(data_dir, fn)
                    if not os.path.exists(fp):
                        continue
                    try:
                        mat = sio.loadmat(fp)
                        d_iner = mat["d_iner"].astype(np.float32)
                        self.samples.append(d_iner)
                        self.labels.append(action - 1)
                    except Exception as e:
                        print(f"  [WARN] skip {fn}: {e}")

        lengths = [s.shape[0] for s in self.samples]
        print(
            f"  Loaded {len(self.samples)} samples  "
            f"(subjects {subjects})  "
            f"seq_len: min={min(lengths)}, max={max(lengths)}, "
            f"mean={np.mean(lengths):.0f}"
        )

        if mean is not None and std is not None:
            self.mean, self.std = mean, std
        else:
            all_frames = np.concatenate(self.samples, axis=0)
            self.mean = all_frames.mean(axis=0)
            self.std = all_frames.std(axis=0) + 1e-8

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx].copy()

        # Apply augmentations BEFORE normalisation (on raw signal)
        # NOTE: no _permutation — shuffling time order destroys the causal
        # structure that the Mamba SSM relies on.
        if self.augment:
            if np.random.random() < 0.5:
                data = _jitter(data, sigma=0.03)
            if np.random.random() < 0.5:
                data = _scaling(data, sigma=0.1)
            if np.random.random() < 0.3:
                data = _rotation(data)
            if np.random.random() < 0.2:
                data = _time_warp(data, sigma=0.15)

        data = (data - self.mean) / self.std
        T, C = data.shape

        if T >= self.max_len:
            data = data[:self.max_len]
        else:
            data = np.pad(data, ((0, self.max_len - T), (0, 0)), mode="constant")

        x = torch.as_tensor(data, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
