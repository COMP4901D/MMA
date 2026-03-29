import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import scipy.io as sio
from scipy.interpolate import interp1d
from .transforms import GAFEncoder
from .groups import UTD_ACTIVITY_TO_GROUP

# ================== Base helpers ==================
TRAIN_SUBJECTS = [1, 3, 5, 7]
TEST_SUBJECTS = [2, 4, 6, 8]

# ================== Unified Inertial Dataset ==================
class UTDMADInertialDataset(Dataset):
    def __init__(self, data_dir: str, subjects: list, max_len: int = 256,
                 mean=None, std=None, return_group: bool = False):
        self.max_len = max_len
        self.return_group = return_group
        self.samples = []
        self.labels = []

        for action in range(1, 28):
            for subj in subjects:
                for trial in range(1, 5):
                    fn = f"a{action}_s{subj}_t{trial}_inertial.mat"
                    fp = os.path.join(data_dir, fn)
                    if os.path.exists(fp):
                        d_iner = sio.loadmat(fp)["d_iner"].astype(np.float32)
                        self.samples.append(d_iner)
                        self.labels.append(action - 1)

        # Normalization
        if mean is None or std is None:
            all_frames = np.concatenate(self.samples, axis=0)
            self.mean = all_frames.mean(axis=0)
            self.std = all_frames.std(axis=0) + 1e-8
        else:
            self.mean, self.std = mean, std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = (self.samples[idx] - self.mean) / self.std
        if len(data) > self.max_len:
            data = data[:self.max_len]
        else:
            data = np.pad(data, ((0, self.max_len - len(data)), (0, 0)), mode="constant")

        x = torch.as_tensor(data, dtype=torch.float32)
        fine = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.return_group:
            group = torch.tensor(UTD_ACTIVITY_TO_GROUP[self.labels[idx]], dtype=torch.long)
            return x, fine, group
        return x, fine


# ================== RGB-D + IMU Dataset ==================
class UTDMADRGBDIMUDataset(Dataset):
    # ... (keep your original implementation, just add return_group parameter)
    def __init__(self, data_dir: str, subjects: list, n_frames=16, frame_size=112,
                 max_imu_len=128, return_group: bool = False, **kwargs):
        # ... your original preloading code ...
        self.return_group = return_group
        # ... rest unchanged ...

    def __getitem__(self, idx):
        # ... your original rgbd + imu processing ...
        fine_label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.return_group:
            group_label = torch.tensor(UTD_ACTIVITY_TO_GROUP[int(fine_label)], dtype=torch.long)
            return rgbd, imu, fine_label, group_label
        return rgbd, imu, fine_label


# ================== Depth + IMU (MMTSA style) ==================
class UTD_MHAD_Dataset(Dataset):
    # ... your original implementation ...
    def __init__(self, data_root: str, subjects: list, return_group: bool = False, **kwargs):
        # ... original init ...
        self.return_group = return_group

    def __getitem__(self, idx):
        # ... original processing to get d, g, label ...
        fine = torch.tensor(label, dtype=torch.long)
        if self.return_group:
            group = torch.tensor(UTD_ACTIVITY_TO_GROUP[label], dtype=torch.long)
            return torch.from_numpy(d), torch.from_numpy(g), fine, group
        return torch.from_numpy(d), torch.from_numpy(g), fine