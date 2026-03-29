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
    ):
        self.max_len = max_len
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
        data = (self.samples[idx] - self.mean) / self.std
        T, C = data.shape

        if T >= self.max_len:
            data = data[:self.max_len]
        else:
            data = np.pad(data, ((0, self.max_len - T), (0, 0)), mode="constant")

        x = torch.as_tensor(data, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
