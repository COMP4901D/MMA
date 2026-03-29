"""Data inspection / validation visualizations.

Verify that encoder inputs (IMU, Depth, GAF, RGBD) are correctly processed
before feeding into models. All functions accept raw numpy arrays or tensors
and produce informative plots.

Usage:
    from vis.data_inspection import plot_imu_signal, plot_gaf_image, ...
    fig = plot_imu_signal(imu_array, title="Sample a1_s1_t1")
"""

import os
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

_IMU_AXIS_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


# ---------------------------------------------------------------------- #
#  IMU signal                                                              #
# ---------------------------------------------------------------------- #

def plot_imu_signal(
    imu,
    axis_names: List[str] = None,
    title: str = "IMU Signal",
    save_dir: str = None,
    filename: str = "imu_signal",
    show: bool = False,
):
    """Plot 6-axis IMU time series.

    Args:
        imu: (T, C) array — raw or normalized IMU sequence
        axis_names: channel labels (default: acc_xyz + gyro_xyz)
        title: plot title
    """
    imu = _to_numpy(imu)
    if imu.ndim == 1:
        imu = imu[:, None]
    T, C = imu.shape
    axis_names = axis_names or _IMU_AXIS_NAMES[:C]

    fig, axes = plt.subplots(C, 1, figsize=(10, 1.5 * C), sharex=True)
    if C == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(imu[:, i], linewidth=0.8)
        ax.set_ylabel(axis_names[i], fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time Step")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _save_and_show(fig, save_dir, filename, show)
    return fig


def plot_imu_comparison(
    imu_raw,
    imu_processed,
    axis_names: List[str] = None,
    title: str = "IMU: Raw vs Processed",
    save_dir: str = None,
    show: bool = False,
):
    """Side-by-side comparison of raw and preprocessed IMU signals.

    Useful for verifying normalization, padding/truncation.
    """
    imu_raw = _to_numpy(imu_raw)
    imu_processed = _to_numpy(imu_processed)
    if imu_raw.ndim == 1:
        imu_raw = imu_raw[:, None]
    if imu_processed.ndim == 1:
        imu_processed = imu_processed[:, None]
    C = imu_raw.shape[1]
    axis_names = axis_names or _IMU_AXIS_NAMES[:C]

    fig, axes = plt.subplots(C, 2, figsize=(12, 1.5 * C), sharex="col")
    if C == 1:
        axes = axes[None, :]
    axes[0, 0].set_title("Raw")
    axes[0, 1].set_title("Processed")
    for i in range(C):
        axes[i, 0].plot(imu_raw[:, i], linewidth=0.8)
        axes[i, 0].set_ylabel(axis_names[i], fontsize=8)
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].plot(imu_processed[:, i], linewidth=0.8, color="tab:orange")
        axes[i, 1].grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Time Step")
    axes[-1, 1].set_xlabel("Time Step")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _save_and_show(fig, save_dir, "imu_comparison", show)
    return fig


# ---------------------------------------------------------------------- #
#  GAF (Gramian Angular Field) images                                      #
# ---------------------------------------------------------------------- #

def plot_gaf_image(
    gaf,
    axis_names: List[str] = None,
    title: str = "GAF Encoded IMU",
    save_dir: str = None,
    filename: str = "gaf_image",
    show: bool = False,
):
    """Plot GAF-encoded IMU channels as a row of images.

    Args:
        gaf: (C, H, W) array — 6-channel GAF image for one time segment
    """
    gaf = _to_numpy(gaf)
    if gaf.ndim == 2:
        gaf = gaf[None]
    C = gaf.shape[0]
    axis_names = axis_names or _IMU_AXIS_NAMES[:C]

    fig, axes = plt.subplots(1, C, figsize=(2.5 * C, 2.5))
    if C == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        im = ax.imshow(gaf[i], cmap="viridis", vmin=0, vmax=1, aspect="equal")
        ax.set_title(axis_names[i], fontsize=9)
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.8, label="GASF value")
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(top=0.88)
    _save_and_show(fig, save_dir, filename, show)
    return fig


def plot_gaf_segments(
    gaf_segments,
    axis_names: List[str] = None,
    title: str = "GAF Segments",
    save_dir: str = None,
    show: bool = False,
):
    """Plot GAF images across multiple time segments.

    Args:
        gaf_segments: (N_seg, C, H, W) — GAF images per segment
    """
    gaf_segments = _to_numpy(gaf_segments)
    N, C, H, W = gaf_segments.shape
    axis_names = axis_names or _IMU_AXIS_NAMES[:C]

    fig, axes = plt.subplots(N, C, figsize=(2.2 * C, 2.2 * N))
    if N == 1:
        axes = axes[None, :]
    if C == 1:
        axes = axes[:, None]
    for seg in range(N):
        for ch in range(C):
            ax = axes[seg, ch]
            ax.imshow(gaf_segments[seg, ch], cmap="viridis", vmin=0, vmax=1)
            ax.axis("off")
            if seg == 0:
                ax.set_title(axis_names[ch], fontsize=8)
        axes[seg, 0].set_ylabel(f"Seg {seg}", fontsize=9, rotation=0, labelpad=30)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _save_and_show(fig, save_dir, "gaf_segments", show)
    return fig


# ---------------------------------------------------------------------- #
#  Depth frames                                                            #
# ---------------------------------------------------------------------- #

def plot_depth_frames(
    depth_seq,
    n_show: int = 8,
    title: str = "Depth Frames",
    save_dir: str = None,
    filename: str = "depth_frames",
    show: bool = False,
):
    """Visualize a sequence of depth frames.

    Args:
        depth_seq: (T, H, W) array — raw depth frame sequence
        n_show: max number of frames to display
    """
    depth_seq = _to_numpy(depth_seq)
    T = depth_seq.shape[0]
    indices = np.linspace(0, T - 1, min(n_show, T), dtype=int)
    n = len(indices)

    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 2.5))
    if n == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices):
        ax.imshow(depth_seq[idx], cmap="inferno")
        ax.set_title(f"t={idx}", fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _save_and_show(fig, save_dir, filename, show)
    return fig


def plot_depth_representation(
    depth_repr,
    channel_names: List[str] = None,
    title: str = "Depth Multi-Channel Representation",
    save_dir: str = None,
    show: bool = False,
):
    """Visualize multi-channel depth representation (mean / motion / max).

    Args:
        depth_repr: (C, H, W) — typically C=3 for mean+motion+max
    """
    depth_repr = _to_numpy(depth_repr)
    if depth_repr.ndim == 2:
        depth_repr = depth_repr[None]
    C = depth_repr.shape[0]
    channel_names = channel_names or ["Mean", "Motion", "Max"][:C]

    fig, axes = plt.subplots(1, C, figsize=(3 * C, 3))
    if C == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        im = ax.imshow(depth_repr[i], cmap="inferno")
        ax.set_title(channel_names[i], fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _save_and_show(fig, save_dir, "depth_repr", show)
    return fig


# ---------------------------------------------------------------------- #
#  RGBD frames                                                             #
# ---------------------------------------------------------------------- #

def plot_rgbd_frames(
    rgbd,
    n_show: int = 8,
    title: str = "RGBD Frames",
    save_dir: str = None,
    show: bool = False,
):
    """Visualize RGB-D frames: top row RGB, bottom row depth.

    Args:
        rgbd: (N, 4, H, W) — 4-channel RGBD frames (values assumed [0,1])
    """
    rgbd = _to_numpy(rgbd)
    N = rgbd.shape[0]
    indices = np.linspace(0, N - 1, min(n_show, N), dtype=int)
    n = len(indices)

    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.5))
    if n == 1:
        axes = axes[:, None]
    for col, idx in enumerate(indices):
        rgb = np.clip(rgbd[idx, :3].transpose(1, 2, 0), 0, 1)
        depth = rgbd[idx, 3]
        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f"t={idx}", fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].imshow(depth, cmap="inferno")
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("RGB", fontsize=10)
    axes[1, 0].set_ylabel("Depth", fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _save_and_show(fig, save_dir, "rgbd_frames", show)
    return fig


# ---------------------------------------------------------------------- #
#  Batch overview                                                          #
# ---------------------------------------------------------------------- #

def plot_batch_overview(
    batch_data,
    labels=None,
    class_names: List[str] = None,
    n_show: int = 8,
    data_type: str = "auto",
    title: str = "Batch Overview",
    save_dir: str = None,
    show: bool = False,
):
    """Quick overview of a batch of samples.

    Args:
        batch_data: (B, ...) tensor/array — first modality from a batch
        labels: (B,) labels
        class_names: list of action names
        n_show: max samples to show
        data_type: "imu" | "image" | "auto"
    """
    batch_data = _to_numpy(batch_data)
    B = batch_data.shape[0]
    n = min(n_show, B)

    if data_type == "auto":
        if batch_data.ndim == 2 or (batch_data.ndim == 3 and batch_data.shape[-1] <= 10):
            data_type = "imu"
        else:
            data_type = "image"

    if data_type == "imu":
        fig, axes = plt.subplots(n, 1, figsize=(10, 1.5 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            sample = batch_data[i]
            if sample.ndim == 1:
                ax.plot(sample, linewidth=0.6)
            else:
                for ch in range(sample.shape[-1]):
                    ax.plot(sample[:, ch], linewidth=0.6, alpha=0.7)
            lbl = _format_label(i, labels, class_names)
            ax.set_ylabel(lbl, fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time Step")
    else:
        fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 2.5))
        if n == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            img = batch_data[i]
            if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                img = img[:3].transpose(1, 2, 0)
            if img.ndim == 3 and img.shape[2] == 1:
                img = img[:, :, 0]
            img = np.clip(img, 0, 1)
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(_format_label(i, labels, class_names), fontsize=8)
            ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _save_and_show(fig, save_dir, "batch_overview", show)
    return fig


# ---------------------------------------------------------------------- #
#  Data distribution                                                       #
# ---------------------------------------------------------------------- #

def plot_data_distribution(
    data,
    channel_names: List[str] = None,
    title: str = "Data Value Distribution",
    save_dir: str = None,
    filename: str = "data_distribution",
    show: bool = False,
):
    """Histogram of per-channel value distributions.

    Args:
        data: (N, ..., C) or (N, C, ...) — any multi-channel data
              Flattened per channel for histogram.
    """
    data = _to_numpy(data)
    if data.ndim == 1:
        data = data[:, None]

    # Assume last dim is channels if small, else first dim after batch
    if data.ndim >= 3 and data.shape[1] <= 10:
        # (B, C, ...) format
        C = data.shape[1]
        channels = [data[:, c].flatten() for c in range(C)]
    elif data.ndim >= 2 and data.shape[-1] <= 10:
        C = data.shape[-1]
        channels = [data[..., c].flatten() for c in range(C)]
    else:
        C = 1
        channels = [data.flatten()]

    channel_names = channel_names or [f"Ch {i}" for i in range(C)]
    fig, axes = plt.subplots(1, C, figsize=(3 * C, 3))
    if C == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.hist(channels[i], bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_title(channel_names[i], fontsize=9)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _save_and_show(fig, save_dir, filename, show)
    return fig


def plot_class_distribution(
    labels,
    class_names: List[str] = None,
    title: str = "Class Distribution",
    save_dir: str = None,
    show: bool = False,
):
    """Bar chart of label frequencies.

    Args:
        labels: (N,) array of integer class labels
    """
    labels = _to_numpy(labels).astype(int)
    classes, counts = np.unique(labels, return_counts=True)
    n = len(classes)
    if class_names:
        names = [class_names[c] if c < len(class_names) else str(c) for c in classes]
    else:
        names = [str(c) for c in classes]

    fig, ax = plt.subplots(figsize=(max(6, 0.4 * n), 4))
    ax.bar(range(n), counts, tick_label=names, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    _save_and_show(fig, save_dir, "class_distribution", show)
    return fig


# ---------------------------------------------------------------------- #
#  Internal helpers                                                        #
# ---------------------------------------------------------------------- #

def _to_numpy(x):
    """Convert tensor to numpy if needed."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _format_label(idx, labels, class_names):
    if labels is None:
        return f"#{idx}"
    lbl = int(labels[idx]) if hasattr(labels, "__getitem__") else int(labels)
    name = class_names[lbl] if class_names and lbl < len(class_names) else ""
    return f"#{idx} c={lbl}" + (f" ({name})" if name else "")


def _save_and_show(fig, save_dir, name, show):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
