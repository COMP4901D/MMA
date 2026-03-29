"""
UTD-MHAD Depth + Inertial Dataset.

Loads Depth frames and IMU data, with GAF encoding and segment sampling.
Used by the MMA-MMTSA pipeline.

File naming: a{1-27}_s{1-8}_t{1-4}_{modality}.mat
  Depth .mat   -> 'd_depth' shape (H, W, T) or (T, H, W)
  Inertial .mat -> 'd_iner' shape (T, 6)
"""

import os
import re
import glob
import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from scipy.ndimage import rotate as nd_rotate

from .transforms import GAFEncoder


class UTD_MHAD_Dataset(Dataset):
    """
    UTD-MHAD dataset (Depth + Inertial) with GAF encoding and segment sampling.
    All data is preloaded into RAM for fast training.
    """

    PAT = re.compile(r"a(\d+)_s(\d+)_t(\d+)")

    def __init__(
        self,
        data_root: str,
        subjects: list,
        num_segments: int = 3,
        gaf_size: int = 64,
        frame_size: int = 64,
        use_gaf: bool = True,
        use_segments: bool = True,
        augment: bool = False,
        depth_channels: int = 3,
    ):
        self.subjects = subjects
        self.num_segments = num_segments
        self.gaf_size = gaf_size
        self.frame_size = frame_size
        self.use_gaf = use_gaf
        self.use_segments = use_segments
        self.augment = augment
        self.depth_channels = depth_channels

        depth_dir = os.path.join(data_root, "Depth")
        inertial_dir = os.path.join(data_root, "Inertial")

        # Discover valid samples
        self.samples = []
        for ip in sorted(glob.glob(os.path.join(inertial_dir, "*.mat"))):
            m = self.PAT.search(os.path.basename(ip))
            if m is None:
                continue
            a, s, t = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if s not in subjects:
                continue
            dp = os.path.join(depth_dir, f"a{a}_s{s}_t{t}_depth.mat")
            if not os.path.exists(dp):
                cands = glob.glob(os.path.join(depth_dir, f"a{a}_s{s}_t{t}*"))
                dp = cands[0] if cands else None
            if dp is None:
                continue
            self.samples.append(dict(dp=dp, ip=ip, label=a - 1, a=a, s=s, t=t))

        print(f"    subjects {subjects}:  {len(self.samples)} samples")

        # Preload and precompute
        self._depth_cache = []
        self._imu_cache = []

        print("    preloading…", end="", flush=True)
        for i, sp in enumerate(self.samples):
            d = self._load_depth(sp["dp"])
            d = self._resize_frames(d)
            self._depth_cache.append(d)

            raw = self._load_inertial(sp["ip"])
            if use_gaf:
                if use_segments:
                    enc = GAFEncoder.encode_segments(raw, num_segments, gaf_size)
                else:
                    enc = GAFEncoder.encode_multi(raw, gaf_size)
            else:
                enc = raw
            self._imu_cache.append(enc)

            if (i + 1) % 200 == 0:
                print(f" {i+1}", end="", flush=True)
        print(f"  done ({len(self.samples)})")

    # --- Raw loaders ---

    @staticmethod
    def _load_inertial(path: str) -> np.ndarray:
        mat = loadmat(path)
        for k in ("d_iner", "d_inertial", "data"):
            if k in mat:
                return mat[k].astype(np.float32)
        for k, v in mat.items():
            if not k.startswith("__") and isinstance(v, np.ndarray):
                if v.ndim == 2 and v.shape[1] == 6:
                    return v.astype(np.float32)
        raise ValueError(f"No inertial data found in {path}")

    @staticmethod
    def _load_depth(path: str) -> np.ndarray:
        mat = loadmat(path)
        for k in ("d_depth", "depth", "data"):
            if k in mat:
                d = mat[k]
                break
        else:
            for k, v in mat.items():
                if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim == 3:
                    d = v
                    break
            else:
                raise ValueError(f"No depth data in {path}")
        if d.ndim == 3:
            if d.shape[2] < d.shape[0] and d.shape[2] < d.shape[1]:
                d = d.transpose(2, 0, 1)
        return d.astype(np.float32)

    def _resize_frames(self, frames: np.ndarray) -> np.ndarray:
        """(T,H,W) -> (T, frame_size, frame_size), normalised to [0,1]."""
        T, H, W = frames.shape
        fs = self.frame_size
        if H == fs and W == fs:
            out = frames
        else:
            idx_h = np.linspace(0, H - 1, fs).astype(int)
            idx_w = np.linspace(0, W - 1, fs).astype(int)
            out = frames[:, idx_h][:, :, idx_w]
        for t in range(T):
            lo, hi = out[t].min(), out[t].max()
            if hi - lo > 1e-8:
                out[t] = (out[t] - lo) / (hi - lo)
            else:
                out[t] = 0.0
        return out.astype(np.float32)

    # --- Segment sampling ---

    def _compute_depth_segment_repr(self, frames, start, end):
        seg = frames[start:end]
        if len(seg) == 0:
            seg = frames[start:start + 1]

        if self.depth_channels == 1:
            idx = (
                random.randint(start, max(start, end - 1))
                if self.augment
                else (start + end) // 2
            )
            idx = min(idx, len(frames) - 1)
            return frames[idx][np.newaxis]

        mean_frame = seg.mean(axis=0)
        if len(seg) > 1:
            diffs = np.abs(np.diff(seg, axis=0))
            motion_frame = diffs.mean(axis=0)
        else:
            motion_frame = np.zeros_like(mean_frame)
        max_frame = seg.max(axis=0)

        for f in [mean_frame, motion_frame, max_frame]:
            lo, hi = f.min(), f.max()
            if hi - lo > 1e-8:
                f[:] = (f - lo) / (hi - lo)
            else:
                f[:] = 0.0

        return np.stack(
            [mean_frame, motion_frame, max_frame], axis=0
        ).astype(np.float32)

    def _sample_depth_segments(self, frames):
        T = len(frames)
        seg_len = max(T // self.num_segments, 1)
        segs = []
        for i in range(self.num_segments):
            s = i * seg_len
            e = min(s + seg_len, T) if i < self.num_segments - 1 else T
            segs.append(self._compute_depth_segment_repr(frames, s, e))
        return np.stack(segs, axis=0)

    def _sample_depth_global(self, frames):
        return self._compute_depth_segment_repr(frames, 0, len(frames))

    # --- __getitem__ ---

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label = self.samples[idx]["label"]
        depth = self._depth_cache[idx]
        imu = self._imu_cache[idx].copy()

        if self.use_segments:
            d = self._sample_depth_segments(depth)
            g = imu
        else:
            d = self._sample_depth_global(depth)
            g = imu

        if self.augment:
            g = g + np.random.randn(*g.shape).astype(np.float32) * 0.03

            if random.random() > 0.5:
                d = np.ascontiguousarray(d[..., ::-1])
                g = np.ascontiguousarray(g[..., ::-1])

            if random.random() > 0.5:
                d = self._random_erase(d)
            if random.random() > 0.5:
                g = self._random_erase(g)

            if random.random() > 0.5:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-0.05, 0.05)
                d = np.clip(d * alpha + beta, 0.0, 1.0).astype(np.float32)

        return (
            torch.from_numpy(d),
            torch.from_numpy(g),
            torch.tensor(label, dtype=torch.long),
        )

    @staticmethod
    def _random_erase(arr, area_ratio=(0.02, 0.2)):
        h, w = arr.shape[-2], arr.shape[-1]
        area = h * w
        ratio = random.uniform(*area_ratio)
        erase_area = int(area * ratio)
        aspect = random.uniform(0.3, 3.3)
        eh = min(int(math.sqrt(erase_area * aspect)), h)
        ew = min(int(math.sqrt(erase_area / aspect)), w)
        if eh == 0 or ew == 0:
            return arr
        y = random.randint(0, h - eh)
        x = random.randint(0, w - ew)
        arr[..., y:y + eh, x:x + ew] = np.float32(np.random.rand())
        return arr

    @staticmethod
    def _rotate_array(arr, angle):
        shape = arr.shape
        flat = arr.reshape(-1, shape[-2], shape[-1])
        for i in range(flat.shape[0]):
            flat[i] = nd_rotate(
                flat[i], angle, reshape=False, order=1, mode="nearest"
            )
        return flat.reshape(shape).astype(np.float32)
