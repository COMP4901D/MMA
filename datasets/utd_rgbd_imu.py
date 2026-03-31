"""
UTD-MHAD RGB-D + IMU Dataset.

Loads and precomputes RGB-D frames + IMU sequences into RAM.
Used by the MMA-RGBD-IMU pipeline.

Data sources:
  RGB  : RGB-part{1..4}/a{i}_s{j}_t{k}_color.avi  -> 640x480 color video
  Depth: Depth/a{i}_s{j}_t{k}_depth.mat            -> 320x240 depth maps
  IMU  : Inertial/a{i}_s{j}_t{k}_inertial.mat      -> (T, 6)
"""

import os
import time

import cv2
import numpy as np
import scipy.io as sio
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

from .transforms import GAFEncoder, ModalityDropout


class UTDMADRGBDIMUDataset(Dataset):
    """
    Loads RGB-D + IMU for multimodal HAR.
    Preloads all data to RAM for fast training.
    Only samples where ALL three files exist are included.
    """

    def __init__(
        self,
        data_dir: str,
        subjects: list,
        n_frames: int = 16,
        frame_size: int = 112,
        max_imu_len: int = 128,
        depth_clip: float = 4000.0,
        iner_mean=None,
        iner_std=None,
        imu_as_gaf: bool = False,
        gaf_size: int = 64,
        modality_dropout: dict = None,
        augment: bool = False,
        use_flow: bool = False,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.max_imu_len = max_imu_len
        self.depth_clip = depth_clip
        self.imu_as_gaf = imu_as_gaf
        self.gaf_size = gaf_size
        self.augment = augment
        self.use_flow = use_flow

        # Modality-loss simulation (training only)
        if modality_dropout is not None:
            self.modality_dropout = ModalityDropout(**modality_dropout)
        else:
            self.modality_dropout = None
        self._rng = np.random.default_rng()

        # Locate folders
        iner_dir = os.path.join(data_dir, "Inertial")
        depth_dir = os.path.join(data_dir, "Depth")
        rgb_dirs = [
            os.path.join(data_dir, d)
            for d in sorted(os.listdir(data_dir))
            if d.lower().startswith("rgb")
            and os.path.isdir(os.path.join(data_dir, d))
        ]

        for d, name in [(iner_dir, "Inertial"), (depth_dir, "Depth")]:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"{name} folder not found: {d}")
        if not rgb_dirs:
            raise FileNotFoundError(f"No RGB-part* folders found in {data_dir}")

        print(f"  Found RGB folders: {[os.path.basename(d) for d in rgb_dirs]}")

        # Preload
        self.rgbd_data: list = []
        self.imu_data: list = []
        self.labels: list = []
        skipped = 0

        print(f"  Preloading subjects {subjects} ...")
        t_start = time.time()

        for action in range(1, 28):
            for subj in subjects:
                for trial in range(1, 5):
                    tag = f"a{action}_s{subj}_t{trial}"

                    iner_fp = os.path.join(iner_dir, f"{tag}_inertial.mat")
                    depth_fp = os.path.join(depth_dir, f"{tag}_depth.mat")

                    rgb_fp = None
                    for rd in rgb_dirs:
                        candidate = os.path.join(rd, f"{tag}_color.avi")
                        if os.path.exists(candidate):
                            rgb_fp = candidate
                            break

                    if not (
                        os.path.exists(iner_fp)
                        and os.path.exists(depth_fp)
                        and rgb_fp is not None
                    ):
                        skipped += 1
                        continue

                    try:
                        d_iner = sio.loadmat(iner_fp)["d_iner"].astype(np.float32)
                        rgb_frames = self._read_avi(rgb_fp)
                        if len(rgb_frames) == 0:
                            raise ValueError("empty RGB video")
                        depth_frames = self._read_depth_mat(depth_fp)
                        if depth_frames.shape[0] == 0:
                            raise ValueError("empty depth data")

                        rgbd = self._build_rgbd(rgb_frames, depth_frames)

                        self.rgbd_data.append(rgbd)
                        self.imu_data.append(d_iner)
                        self.labels.append(action - 1)

                    except Exception as e:
                        print(f"    [WARN] skip {tag}: {e}")
                        skipped += 1

                    n = len(self.labels)
                    if n > 0 and n % 100 == 0:
                        print(f"    ... {n} samples loaded")

        elapsed = time.time() - t_start
        print(
            f"  \u2713 {len(self.labels)} samples  "
            f"(skipped {skipped})  [{elapsed:.1f}s]"
        )

        # IMU normalisation stats
        if iner_mean is not None:
            self.iner_mean, self.iner_std = iner_mean, iner_std
        else:
            all_iner = np.concatenate(self.imu_data, axis=0)
            self.iner_mean = all_iner.mean(axis=0)
            self.iner_std = all_iner.std(axis=0) + 1e-8

    # --- I/O helpers ---

    @staticmethod
    def _read_avi(path: str) -> list:
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    @staticmethod
    def _read_depth_mat(path: str) -> np.ndarray:
        mat = sio.loadmat(path)
        d = None
        for key in ["d_depth", "depth"]:
            if key in mat:
                d = mat[key]
                break
        if d is None:
            for key in mat:
                if not key.startswith("__"):
                    d = mat[key]
                    break
        if d is None:
            raise KeyError(f"no depth variable found in {path}")
        d = d.astype(np.float32)

        if d.ndim == 3:
            if d.shape[2] <= d.shape[0] and d.shape[2] <= d.shape[1]:
                d = d.transpose(2, 0, 1)
        elif d.ndim == 2:
            d = d[np.newaxis]
        return d

    # --- Frame sampling & RGBD construction ---

    @staticmethod
    def _uniform_sample(total: int, n: int) -> np.ndarray:
        if total >= n:
            return np.linspace(0, total - 1, n).astype(int)
        idx = np.arange(total)
        return np.concatenate([idx, np.full(n - total, total - 1)]).astype(int)

    def _build_rgbd(self, rgb_frames, depth_frames):
        n_rgb = len(rgb_frames)
        n_dep = depth_frames.shape[0]
        S = self.frame_size

        rgb_idx = self._uniform_sample(n_rgb, self.n_frames)
        dep_idx = self._uniform_sample(n_dep, self.n_frames)

        # Build RGBD-Flow: 4ch(RGBD) + 2ch(optical flow) = 6 channels
        n_ch = 6 if self.use_flow else 4
        out = np.empty((self.n_frames, n_ch, S, S), dtype=np.float32)

        prev_gray = None
        for i in range(self.n_frames):
            rgb = (
                cv2.resize(rgb_frames[rgb_idx[i]], (S, S)).astype(np.float32)
                / 255.0
            )
            out[i, :3] = rgb.transpose(2, 0, 1)

            dep = cv2.resize(depth_frames[dep_idx[i]], (S, S))
            dep = np.clip(dep, 0, self.depth_clip) / self.depth_clip
            out[i, 3] = dep

            if self.use_flow:
                gray = cv2.cvtColor(
                    (rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
                )
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                    )
                    # Normalize flow to [-1, 1] (clamp at ±20 pixels)
                    flow = np.clip(flow / 20.0, -1.0, 1.0)
                    out[i, 4] = flow[:, :, 0]  # horizontal
                    out[i, 5] = flow[:, :, 1]  # vertical
                else:
                    out[i, 4:6] = 0.0
                prev_gray = gray

        return out

    # --- IMU resampling ---

    @staticmethod
    def _resample_1d(data: np.ndarray, target: int) -> np.ndarray:
        T = data.shape[0]
        if T == target:
            return data
        if T < 2:
            return np.tile(data, (target, 1))
        x_old = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, target)
        return interp1d(x_old, data, axis=0, kind="linear")(x_new).astype(
            np.float32
        )

    # --- PyTorch interface ---

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        rgbd = self.rgbd_data[idx].copy()  # (T, 4, H, W) float32 ndarray

        # Training augmentation (applied consistently across all frames)
        if self.augment:
            rgbd = self._augment_rgbd(rgbd)

        imu = (self.imu_data[idx] - self.iner_mean) / self.iner_std
        if self.imu_as_gaf:
            imu = GAFEncoder.encode_multi(imu, self.gaf_size)
        else:
            imu = self._resample_1d(imu, self.max_imu_len)

        # IMU augmentation (jitter + scaling, same as skeleton pipeline)
        if self.augment:
            imu = self._augment_imu(imu)

        # Apply modality dropout *before* converting to tensor
        if self.modality_dropout is not None:
            rgbd, imu = self.modality_dropout(rgbd, imu, rng=self._rng)

        rgbd = torch.as_tensor(rgbd, dtype=torch.float32)
        imu = torch.as_tensor(imu, dtype=torch.float32)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return rgbd, imu, label

    # --- Augmentation helpers ---

    def _augment_rgbd(self, rgbd):
        """Apply spatial augmentations consistently across all frames.

        - Random horizontal flip (p=0.5)
        - Random crop (scale 0.8-1.0) then resize back
        - Color jitter on RGB channels only (brightness, contrast)
        """
        T, C, H, W = rgbd.shape

        # Random horizontal flip
        if self._rng.random() < 0.5:
            rgbd = rgbd[:, :, :, ::-1].copy()
            # Flip horizontal flow channel sign if present
            if C >= 6:
                rgbd[:, 4] = -rgbd[:, 4]

        # Random crop: crop a subregion and resize back
        scale = self._rng.uniform(0.8, 1.0)
        crop_h, crop_w = int(H * scale), int(W * scale)
        top = self._rng.integers(0, H - crop_h + 1)
        left = self._rng.integers(0, W - crop_w + 1)
        for i in range(T):
            cropped = rgbd[i, :, top : top + crop_h, left : left + crop_w]
            # Resize back: transpose to (H, W, C) for cv2, then back
            cropped_hwc = cropped.transpose(1, 2, 0)
            resized = cv2.resize(cropped_hwc, (W, H))
            rgbd[i] = resized.transpose(2, 0, 1)

        # Color jitter on RGB channels only (brightness + contrast)
        brightness = self._rng.uniform(0.85, 1.15)
        contrast = self._rng.uniform(0.85, 1.15)
        rgbd[:, :3] = np.clip(
            (rgbd[:, :3] - 0.5) * contrast + 0.5 + (brightness - 1.0),
            0.0, 1.0,
        )

        return rgbd

    def _augment_imu(self, imu):
        """Apply jitter + scaling to IMU (mirrors skeleton pipeline)."""
        # Jitter
        imu = imu + self._rng.normal(0, 0.03, size=imu.shape).astype(np.float32)
        # Scaling
        scale = self._rng.normal(1.0, 0.1, size=(1, imu.shape[1])).astype(np.float32)
        imu = imu * scale
        return imu
