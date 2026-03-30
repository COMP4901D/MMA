"""
Gramian Angular Field (GAF) encoder for IMU data + modality-loss simulation.

GAF: Converts 1-D IMU time series to 2-D grayscale images.
     Reference: Wang & Oates (AAAI-WS 2015) + MMTSA (IMWUT 2023)

ModalityDropout: Simulates sensor failure scenarios for robustness training.
     - full:        Entire modality zeroed out for the sample.
     - consecutive: Contiguous temporal blocks dropped with probability p.
"""

import numpy as np
from scipy.signal import resample


# ================================================================
#  Modality-Loss Simulation
# ================================================================

class ModalityDropout:
    """Simulate modality-level sensor failures during training.

    Parameters
    ----------
    mode : str
        ``"full"``        – zero out entire modality.
        ``"consecutive"`` – drop contiguous temporal blocks.
        ``"mixed"``       – randomly pick full or consecutive per sample.
    p_full : float
        Probability a *single* modality is fully dropped (per sample).
        Applied independently to each modality.
    p_consecutive : float
        Per-timestep probability of *starting* a dropout block.
    consecutive_len_range : tuple[int, int]
        (min_len, max_len) of each consecutive dropout block in frames.
    modalities : list[str]
        Which modalities are eligible for dropout.
        Accepts ``"rgbd"`` and/or ``"imu"``.
    both_drop_prob : float
        Probability that *both* modalities are dropped simultaneously
        (only meaningful for ``"full"`` mode). Default 0 (never drop both).
    """

    VALID_MODES = {"full", "consecutive", "mixed"}
    VALID_MODALITIES = {"rgbd", "imu"}

    def __init__(
        self,
        mode: str = "full",
        p_full: float = 0.2,
        p_consecutive: float = 0.3,
        consecutive_len_range: tuple = (2, 6),
        modalities: list = None,
        both_drop_prob: float = 0.0,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")
        self.mode = mode
        self.p_full = p_full
        self.p_consecutive = p_consecutive
        self.consecutive_len_range = consecutive_len_range
        self.modalities = [m for m in (modalities or ["rgbd", "imu"])
                           if m in self.VALID_MODALITIES]
        self.both_drop_prob = both_drop_prob

    # ---- public API ----

    def __call__(self, rgbd: np.ndarray, imu: np.ndarray,
                 rng: np.random.Generator = None):
        """Apply modality dropout in-place (on copies) and return results.

        Parameters
        ----------
        rgbd : ndarray, shape (T, 4, H, W)  or similar leading-time axis
        imu  : ndarray, shape (T, C) for conv1d  or  (C, H, W) for GAF

        Returns
        -------
        rgbd, imu : ndarrays (same shapes, potentially zeroed regions)
        """
        if rng is None:
            rng = np.random.default_rng()

        rgbd = rgbd.copy()
        imu = imu.copy()

        if self.mode == "full":
            rgbd, imu = self._apply_full(rgbd, imu, rng)
        elif self.mode == "consecutive":
            rgbd, imu = self._apply_consecutive(rgbd, imu, rng)
        else:  # mixed
            if rng.random() < 0.5:
                rgbd, imu = self._apply_full(rgbd, imu, rng)
            else:
                rgbd, imu = self._apply_consecutive(rgbd, imu, rng)

        return rgbd, imu

    # ---- full dropout ----

    def _apply_full(self, rgbd, imu, rng):
        drop_rgbd = "rgbd" in self.modalities and rng.random() < self.p_full
        drop_imu = "imu" in self.modalities and rng.random() < self.p_full

        # Optionally allow both to drop simultaneously
        if drop_rgbd and drop_imu and rng.random() >= self.both_drop_prob:
            # Keep at least one — randomly restore one
            if rng.random() < 0.5:
                drop_rgbd = False
            else:
                drop_imu = False

        if drop_rgbd:
            rgbd[:] = 0.0
        if drop_imu:
            imu[:] = 0.0

        return rgbd, imu

    # ---- consecutive dropout ----

    def _apply_consecutive(self, rgbd, imu, rng):
        if "rgbd" in self.modalities:
            rgbd = self._drop_blocks(rgbd, rng, is_temporal_first=True)
        if "imu" in self.modalities:
            imu = self._drop_blocks_imu(imu, rng)
        return rgbd, imu

    def _drop_blocks(self, data, rng, is_temporal_first=True):
        """Zero out contiguous blocks along the temporal (first) axis."""
        T = data.shape[0]
        lo, hi = self.consecutive_len_range
        t = 0
        while t < T:
            if rng.random() < self.p_consecutive:
                block_len = rng.integers(lo, hi + 1)
                end = min(t + block_len, T)
                data[t:end] = 0.0
                t = end
            else:
                t += 1
        return data

    def _drop_blocks_imu(self, imu, rng):
        """Handle both conv1d (T, C) and GAF (C, H, W) layouts."""
        if imu.ndim == 2:
            # Conv1D layout: (T, C) — drop along time axis
            return self._drop_blocks(imu, rng, is_temporal_first=True)
        elif imu.ndim == 3:
            # GAF layout: (C, H, W) — drop entire channels
            C = imu.shape[0]
            for c in range(C):
                if rng.random() < self.p_consecutive:
                    imu[c] = 0.0
        return imu

    def __repr__(self):
        return (f"ModalityDropout(mode={self.mode!r}, p_full={self.p_full}, "
                f"p_consec={self.p_consecutive}, "
                f"block_range={self.consecutive_len_range}, "
                f"modalities={self.modalities})")


class GAFEncoder:
    """Transforms IMU time series into Gramian Angular Summation Field images."""

    @staticmethod
    def encode_axis(series: np.ndarray, size: int) -> np.ndarray:
        """Single-axis IMU -> (size, size) GASF image, values in [0, 1]."""
        if len(series) < 2:
            return np.zeros((size, size), dtype=np.float32)

        x = resample(series.astype(np.float64), size)

        lo, hi = x.min(), x.max()
        if hi - lo < 1e-8:
            return np.zeros((size, size), dtype=np.float32)
        scaled = np.clip((2.0 * x - hi - lo) / (hi - lo), -1.0, 1.0)

        phi = np.arccos(scaled)
        gaf = np.cos(np.add.outer(phi, phi))

        g_lo, g_hi = gaf.min(), gaf.max()
        if g_hi - g_lo < 1e-8:
            return np.zeros((size, size), dtype=np.float32)
        return ((gaf - g_lo) / (g_hi - g_lo)).astype(np.float32)

    @staticmethod
    def encode_multi(imu: np.ndarray, size: int) -> np.ndarray:
        """Multi-axis (T, C) -> (C, size, size)."""
        return np.stack(
            [GAFEncoder.encode_axis(imu[:, c], size)
             for c in range(imu.shape[1])],
            axis=0,
        )

    @staticmethod
    def encode_segments(imu: np.ndarray, n_seg: int, size: int) -> np.ndarray:
        """Segment-wise encoding -> (n_seg, C, size, size)."""
        T, C = imu.shape
        seg_len = max(T // n_seg, 2)
        out = []
        for i in range(n_seg):
            s = i * seg_len
            e = min(s + seg_len, T) if i < n_seg - 1 else T
            seg = imu[s:e]
            if len(seg) < 2:
                out.append(np.zeros((C, size, size), dtype=np.float32))
            else:
                out.append(GAFEncoder.encode_multi(seg, size))
        return np.stack(out, axis=0)
