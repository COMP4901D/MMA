"""
Gramian Angular Field (GAF) encoder for IMU data + modality-loss simulation.

GAF: Converts 1-D IMU time series to 2-D grayscale images.
     Reference: Wang & Oates (AAAI-WS 2015) + MMTSA (IMWUT 2023)

ModalityDropout: Simulates sensor failure scenarios for robustness training.
     - full:        Entire modality zeroed out for the sample.
     - consecutive: Contiguous temporal blocks dropped with probability p.
     - noise:       Gaussian noise injection at specified noise level.
     - noise_full:  Noise + full dropout combined (noise on one, drop the other).
     - noise_consec: Noise on one modality + consecutive blocks on the other.

Reference: Xaviar, Yang & Ardakanian, "Centaur: Robust Multimodal Fusion
for Human Activity Recognition", IEEE Sensors Journal, 2024.
"""

import numpy as np
from scipy.signal import resample


# ================================================================
#  Modality-Loss Simulation
# ================================================================

class ModalityDropout:
    """Simulate modality-level sensor failures during training/evaluation.

    Parameters
    ----------
    mode : str
        ``"full"``         – zero out entire modality.
        ``"consecutive"``  – drop contiguous temporal blocks.
        ``"mixed"``        – randomly pick full or consecutive per sample.
        ``"noise"``        – add Gaussian noise to modality signals.
        ``"noise_full"``   – noise on one modality + full dropout on the other.
        ``"noise_consec"`` – noise on one modality + consecutive blocks on the other.
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
    noise_std : float
        Standard deviation of Gaussian noise for noise-based modes.
        Applied relative to the data's local std (SNR-style).
    noise_snr_db : float or None
        If set, overrides noise_std with SNR-based noise level.
        noise_std = signal_std / (10^(snr_db/20)).
    """

    VALID_MODES = {"full", "consecutive", "mixed",
                   "noise", "noise_full", "noise_consec"}
    VALID_MODALITIES = {"rgbd", "imu"}

    def __init__(
        self,
        mode: str = "full",
        p_full: float = 0.2,
        p_consecutive: float = 0.3,
        consecutive_len_range: tuple = (2, 6),
        modalities: list = None,
        both_drop_prob: float = 0.0,
        noise_std: float = 0.5,
        noise_snr_db: float = None,
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
        self.noise_std = noise_std
        self.noise_snr_db = noise_snr_db

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
        elif self.mode == "noise":
            rgbd, imu = self._apply_noise(rgbd, imu, rng)
        elif self.mode == "noise_full":
            rgbd, imu = self._apply_noise_full(rgbd, imu, rng)
        elif self.mode == "noise_consec":
            rgbd, imu = self._apply_noise_consec(rgbd, imu, rng)
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

    # ---- noise injection ----

    def _compute_noise_std(self, data, rng):
        """Compute noise standard deviation, either fixed or SNR-based."""
        if self.noise_snr_db is not None:
            signal_std = np.std(data)
            if signal_std < 1e-8:
                return 0.0
            return signal_std / (10.0 ** (self.noise_snr_db / 20.0))
        return self.noise_std

    def _add_noise(self, data, rng):
        """Add Gaussian noise to a data array."""
        std = self._compute_noise_std(data, rng)
        if std > 0:
            noise = rng.normal(0.0, std, size=data.shape).astype(data.dtype)
            data = data + noise
        return data

    def _apply_noise(self, rgbd, imu, rng):
        """Add Gaussian noise to eligible modalities."""
        if "rgbd" in self.modalities:
            rgbd = self._add_noise(rgbd, rng)
        if "imu" in self.modalities:
            imu = self._add_noise(imu, rng)
        return rgbd, imu

    def _apply_noise_full(self, rgbd, imu, rng):
        """Noise on one modality, full dropout on the other.

        Simulates a scenario where one sensor is noisy while the other
        has completely failed (Centaur-style combined corruption).
        """
        if rng.random() < 0.5:
            # Noise on RGBD, drop IMU
            if "rgbd" in self.modalities:
                rgbd = self._add_noise(rgbd, rng)
            if "imu" in self.modalities:
                imu[:] = 0.0
        else:
            # Noise on IMU, drop RGBD
            if "imu" in self.modalities:
                imu = self._add_noise(imu, rng)
            if "rgbd" in self.modalities:
                rgbd[:] = 0.0
        return rgbd, imu

    def _apply_noise_consec(self, rgbd, imu, rng):
        """Noise on one modality, consecutive block dropout on the other.

        Simulates partial temporal sensor failure with degraded signal quality
        on the other sensor (Centaur-style combined corruption).
        """
        if rng.random() < 0.5:
            # Noise on RGBD, consecutive blocks on IMU
            if "rgbd" in self.modalities:
                rgbd = self._add_noise(rgbd, rng)
            if "imu" in self.modalities:
                imu = self._drop_blocks_imu(imu, rng)
        else:
            # Noise on IMU, consecutive blocks on RGBD
            if "imu" in self.modalities:
                imu = self._add_noise(imu, rng)
            if "rgbd" in self.modalities:
                rgbd = self._drop_blocks(rgbd, rng, is_temporal_first=True)
        return rgbd, imu

    def __repr__(self):
        return (f"ModalityDropout(mode={self.mode!r}, p_full={self.p_full}, "
                f"p_consec={self.p_consecutive}, "
                f"block_range={self.consecutive_len_range}, "
                f"noise_std={self.noise_std}, noise_snr_db={self.noise_snr_db}, "
                f"modalities={self.modalities})")


# ================================================================
#  Sensor-Corruption Augmentation (Centaur-style, training-time)
# ================================================================

class SensorCorruptionAugment:
    """Stochastic Centaur-style corruption applied during training.

    Each call randomly picks one of four corruption modes (matching the
    evaluation protocol from Xaviar et al., IEEE Sensors 2024):

        Mode 1 — Gaussian noise N(0, σ) per element
        Mode 2 — Per-channel consecutive missing (exponential intervals)
        Mode 3 — Per-sensor consecutive missing
        Mode 4 — Combined: noise then per-channel missing

    Severity is sampled uniformly within the configured ranges so the
    model sees a wide distribution of corruption levels.

    Parameters
    ----------
    p_apply : float
        Probability of applying *any* corruption to a given sample.
    sigma_range : tuple(float, float)
        Uniform range for Gaussian noise σ (Mode 1 / 4).
    rgbd_s_norm / imu_s_norm : float
        Expected normal-phase length (exponential scale) for RGBD / IMU.
    rgbd_s_corr_range / imu_s_corr_range : tuple(float, float)
        Uniform range for corrupted-phase length (exponential scale).
    mode_weights : tuple of 4 floats
        Relative weights for picking Mode 1 / 2 / 3 / 4.
    """

    def __init__(
        self,
        p_apply: float = 0.5,
        sigma_range: tuple = (0.02, 0.25),
        rgbd_s_norm: float = 8.0,
        rgbd_s_corr_range: tuple = (1.0, 6.0),
        imu_s_norm: float = 60.0,
        imu_s_corr_range: tuple = (10.0, 45.0),
        mode_weights: tuple = (1, 1, 1, 1),
    ):
        self.p_apply = p_apply
        self.sigma_range = sigma_range
        self.rgbd_s_norm = rgbd_s_norm
        self.rgbd_s_corr_range = rgbd_s_corr_range
        self.imu_s_norm = imu_s_norm
        self.imu_s_corr_range = imu_s_corr_range
        w = np.array(mode_weights, dtype=np.float64)
        self._mode_probs = w / w.sum()

    def __call__(self, rgbd: np.ndarray, imu: np.ndarray,
                 rng: np.random.Generator = None):
        if rng is None:
            rng = np.random.default_rng()

        if rng.random() >= self.p_apply:
            return rgbd, imu

        rgbd = rgbd.copy()
        imu = imu.copy()

        mode = rng.choice([1, 2, 3, 4], p=self._mode_probs)
        sigma = rng.uniform(*self.sigma_range)
        rc = rng.uniform(*self.rgbd_s_corr_range)
        ic = rng.uniform(*self.imu_s_corr_range)

        if mode == 1:
            rgbd = self._add_noise(rgbd, sigma, rng)
            imu = self._add_noise(imu, sigma, rng)
        elif mode == 2:
            rgbd = self._per_channel_missing(rgbd, self.rgbd_s_norm, rc, rng,
                                             is_rgbd=True)
            imu = self._per_channel_missing(imu, self.imu_s_norm, ic, rng,
                                            is_rgbd=False)
        elif mode == 3:
            rgbd = self._per_sensor_missing(rgbd, self.rgbd_s_norm, rc, rng)
            imu = self._per_sensor_missing(imu, self.imu_s_norm, ic, rng)
        else:  # mode 4: noise then per-channel missing
            rgbd = self._add_noise(rgbd, sigma, rng)
            imu = self._add_noise(imu, sigma, rng)
            rgbd = self._per_channel_missing(rgbd, self.rgbd_s_norm, rc, rng,
                                             is_rgbd=True)
            imu = self._per_channel_missing(imu, self.imu_s_norm, ic, rng,
                                            is_rgbd=False)
        return rgbd, imu

    # ---- primitives ----

    @staticmethod
    def _add_noise(data, sigma, rng):
        return data + rng.normal(0.0, sigma, size=data.shape).astype(data.dtype)

    @staticmethod
    def _exponential_intervals(T, s_norm, s_corr, rng):
        """Generate alternating normal / corrupted intervals."""
        mask = np.ones(T, dtype=bool)  # True = keep
        t, in_normal = 0, True
        while t < T:
            dur = max(1, int(rng.exponential(s_norm if in_normal else s_corr)))
            end = min(t + dur, T)
            if not in_normal:
                mask[t:end] = False
            t = end
            in_normal = not in_normal
        return mask

    @staticmethod
    def _per_channel_missing(data, s_norm, s_corr, rng, is_rgbd=False):
        if is_rgbd:
            # (T, C, H, W) — independent mask per channel
            T, C = data.shape[0], data.shape[1]
            for c in range(C):
                mask = SensorCorruptionAugment._exponential_intervals(
                    T, s_norm, s_corr, rng)
                data[~mask, c] = 0.0
        else:
            # (T, C) — independent mask per channel
            T, C = data.shape
            for c in range(C):
                mask = SensorCorruptionAugment._exponential_intervals(
                    T, s_norm, s_corr, rng)
                data[~mask, c] = 0.0
        return data

    @staticmethod
    def _per_sensor_missing(data, s_norm, s_corr, rng):
        T = data.shape[0]
        mask = SensorCorruptionAugment._exponential_intervals(
            T, s_norm, s_corr, rng)
        # Zero all channels at masked timesteps
        data[~mask] = 0.0
        return data

    def __repr__(self):
        return (f"SensorCorruptionAugment(p={self.p_apply}, "
                f"σ={self.sigma_range}, "
                f"rgbd_s=[{self.rgbd_s_norm},{self.rgbd_s_corr_range}], "
                f"imu_s=[{self.imu_s_norm},{self.imu_s_corr_range}])")


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
