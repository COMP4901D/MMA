"""
UTD-MHAD Skeleton + IMU Dataset — Dual-modality loader.

Loads skeleton joint positions and 6-axis IMU data from .mat files.
Skeleton: d_skel -> (20, 3, T_skel) reshaped to (T_skel, 60)
IMU:      d_iner -> (T_imu, 6)

Both modalities are independently padded/truncated to their own max_len,
normalised channel-wise, and optionally augmented.
"""

import os

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


# ================================================================
#  Skeleton augmentation helpers
# ================================================================

def _skel_jitter(x: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    return x + np.random.normal(0, sigma, x.shape).astype(np.float32)


def _skel_scaling(x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    factor = np.random.normal(1.0, sigma, (1, x.shape[1])).astype(np.float32)
    return x * factor


def _skel_rotation(x: np.ndarray) -> np.ndarray:
    """Small random 3D rotation applied to each joint's (x,y,z)."""
    angle = np.random.uniform(-0.15, 0.15, 3)
    cx, sx = np.cos(angle[0]), np.sin(angle[0])
    cy, sy = np.cos(angle[1]), np.sin(angle[1])
    cz, sz = np.cos(angle[2]), np.sin(angle[2])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    R = Rz @ Ry @ Rx
    # x is (T, 60) = (T, 20_joints * 3_coords)
    T = x.shape[0]
    out = x.reshape(T, 20, 3) @ R.T
    return out.reshape(T, 60)


# IMU augmentation (shared with utd_inertial.py)
def _imu_jitter(x: np.ndarray, sigma: float = 0.03) -> np.ndarray:
    return x + np.random.normal(0, sigma, x.shape).astype(np.float32)


def _imu_scaling(x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    factor = np.random.normal(1.0, sigma, (1, x.shape[1])).astype(np.float32)
    return x * factor


def _imu_rotation(x: np.ndarray) -> np.ndarray:
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


def _pad_or_truncate(data: np.ndarray, max_len: int) -> np.ndarray:
    T = data.shape[0]
    if T >= max_len:
        return data[:max_len]
    return np.pad(data, ((0, max_len - T), (0, 0)), mode="constant")


class UTDMADSkelIMUDataset(Dataset):
    """
    Dual-modality dataset: skeleton joints (20×3 = 60-dim) + IMU (6-dim).
    Returns: (skel_tensor, imu_tensor, label)
    """

    def __init__(
        self,
        data_dir: str,
        subjects: list,
        max_len_skel: int = 128,
        max_len_imu: int = 256,
        skel_mean: np.ndarray = None,
        skel_std: np.ndarray = None,
        imu_mean: np.ndarray = None,
        imu_std: np.ndarray = None,
        augment: bool = False,
    ):
        self.max_len_skel = max_len_skel
        self.max_len_imu = max_len_imu
        self.augment = augment

        self.skel_samples: list = []
        self.imu_samples: list = []
        self.labels: list = []

        # Resolve directories: data_dir should be the UTD-MHAD root
        skel_dir = os.path.join(data_dir, "Skeleton")
        imu_dir = os.path.join(data_dir, "Inertial")

        for action in range(1, 28):
            for subj in subjects:
                for trial in range(1, 5):
                    skel_fn = f"a{action}_s{subj}_t{trial}_skeleton.mat"
                    imu_fn = f"a{action}_s{subj}_t{trial}_inertial.mat"
                    skel_fp = os.path.join(skel_dir, skel_fn)
                    imu_fp = os.path.join(imu_dir, imu_fn)

                    if not (os.path.exists(skel_fp) and os.path.exists(imu_fp)):
                        continue
                    try:
                        skel = sio.loadmat(skel_fp)["d_skel"]  # (20, 3, T)
                        imu = sio.loadmat(imu_fp)["d_iner"]    # (T, 6)
                        # Reshape skeleton: (20, 3, T) -> (T, 60)
                        skel = skel.transpose(2, 0, 1).reshape(-1, 60).astype(np.float32)
                        imu = imu.astype(np.float32)
                        self.skel_samples.append(skel)
                        self.imu_samples.append(imu)
                        self.labels.append(action - 1)
                    except Exception as e:
                        print(f"  [WARN] skip a{action}_s{subj}_t{trial}: {e}")

        skel_lens = [s.shape[0] for s in self.skel_samples]
        imu_lens = [s.shape[0] for s in self.imu_samples]
        print(
            f"  Loaded {len(self.labels)} paired samples  "
            f"(subjects {subjects})  "
            f"skel_len: {min(skel_lens)}-{max(skel_lens)} (mean={np.mean(skel_lens):.0f})  "
            f"imu_len: {min(imu_lens)}-{max(imu_lens)} (mean={np.mean(imu_lens):.0f})"
        )

        # Compute or load normalisation stats
        if skel_mean is not None:
            self.skel_mean, self.skel_std = skel_mean, skel_std
        else:
            all_skel = np.concatenate(self.skel_samples, axis=0)
            self.skel_mean = all_skel.mean(axis=0)
            self.skel_std = all_skel.std(axis=0) + 1e-8

        if imu_mean is not None:
            self.imu_mean, self.imu_std = imu_mean, imu_std
        else:
            all_imu = np.concatenate(self.imu_samples, axis=0)
            self.imu_mean = all_imu.mean(axis=0)
            self.imu_std = all_imu.std(axis=0) + 1e-8

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        skel = self.skel_samples[idx].copy()
        imu = self.imu_samples[idx].copy()

        if self.augment:
            # Skeleton augmentation
            if np.random.random() < 0.5:
                skel = _skel_jitter(skel, sigma=0.01)
            if np.random.random() < 0.5:
                skel = _skel_scaling(skel, sigma=0.1)
            if np.random.random() < 0.3:
                skel = _skel_rotation(skel)
            # IMU augmentation
            if np.random.random() < 0.5:
                imu = _imu_jitter(imu, sigma=0.03)
            if np.random.random() < 0.5:
                imu = _imu_scaling(imu, sigma=0.1)
            if np.random.random() < 0.3:
                imu = _imu_rotation(imu)

        # Normalise
        skel = (skel - self.skel_mean) / self.skel_std
        imu = (imu - self.imu_mean) / self.imu_std

        # Pad/truncate independently
        skel = _pad_or_truncate(skel, self.max_len_skel)
        imu = _pad_or_truncate(imu, self.max_len_imu)

        return (
            torch.as_tensor(skel, dtype=torch.float32),
            torch.as_tensor(imu, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
