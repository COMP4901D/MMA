"""
Gramian Angular Field (GAF) encoder for IMU data.

Converts 1-D IMU time series to 2-D grayscale images.
Reference: Wang & Oates (AAAI-WS 2015) + MMTSA (IMWUT 2023)
"""

import numpy as np
from scipy.signal import resample


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
