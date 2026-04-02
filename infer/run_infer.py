"""
Unified Inference Runner
========================
Loads a trained checkpoint and evaluates on the test set.  Supports the same
pipeline registry and custom model/class overrides as the training runner.

If the checkpoint was saved by the unified trainer, pipeline/model_kwargs
are restored automatically (override with explicit args).

Usage:
  python infer/run_infer.py --pipeline mmtsa --data_root datasets/UTD-MHAD \\
         --checkpoint checkpoints/best.pt

  python infer/run_infer.py --pipeline utdmad --data_root datasets/UTD-MHAD/Inertial \\
         --checkpoint checkpoints/best.pt

  # Custom model:
  python infer/run_infer.py \\
         --model_module model.mma_utdmad --model_class MomentumMambaHAR \\
         --dataset_module datasets.utd_inertial --dataset_class UTDMADInertialDataset \\
         --data_root datasets/UTD-MHAD/Inertial --checkpoint best.pt \\
         --input_mode unpack --output_mode logits
"""

import argparse
import importlib
import inspect
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
)

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vis.evaluation import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_per_class_accuracy,
)


# ================================================================
#  Pipeline Registry  (shared with train runner)
# ================================================================

PIPELINES = {
    "utdmad": {
        "model_module": "model.mma_utdmad",
        "model_class": "MomentumMambaHAR",
        "dataset_module": "datasets.utd_inertial",
        "dataset_class": "UTDMADInertialDataset",
        "data_key": "data_dir",
        "input_mode": "unpack",
        "output_mode": "logits",
        "trainer_module": "train.default_train",
        "default_model_kwargs": {},
        "default_dataset_kwargs": {"max_len": 256},
        "norm_keys": ["mean", "std"],
    },
    "mmtsa": {
        "model_module": "model.mma_mmtsa",
        "model_class": "MMA_MMTSA",
        "dataset_module": "datasets.utd_depth_imu",
        "dataset_class": "UTD_MHAD_Dataset",
        "data_key": "data_root",
        "input_mode": "unpack",
        "output_mode": "tuple_first",
        "trainer_module": "train.default_train",
        "default_model_kwargs": {"fusion": "attention"},
        "default_dataset_kwargs": {"num_segments": 3, "gaf_size": 64},
        "norm_keys": [],
    },
    "rgbd_imu": {
        "model_module": "model.mma_rgbd_imu",
        "model_class": "MultimodalMMA",
        "dataset_module": "datasets.utd_rgbd_imu",
        "dataset_class": "UTDMADRGBDIMUDataset",
        "data_key": "data_dir",
        "input_mode": "unpack",
        "output_mode": "logits",
        "trainer_module": "train.default_train",
        "default_model_kwargs": {"fusion": "attention"},
        "default_dataset_kwargs": {"n_frames": 16, "frame_size": 112, "max_imu_len": 128},
        "norm_keys": ["iner_mean", "iner_std"],
    },
    "mumu": {
        "model_module": "baselines.MuMu.MuMu",
        "model_class": "MuMu",
        "dataset_module": "datasets.utd_inertial",
        "dataset_class": "UTDMADInertialDataset",
        "data_key": "data_dir",
        "input_mode": "list",
        "output_mode": "mumu",
        "trainer_module": "train.mumu_train",
        "default_model_kwargs": {
            "num_modalities": 1, "feature_dim": 128,
            "input_dim": 6, "num_activities": 27, "num_activity_groups": 5,
        },
        "default_dataset_kwargs": {"max_len": 256},
        "default_trainer_kwargs": {"beta_aux": 0.5},
        "norm_keys": ["mean", "std"],
    },
    "mumu_rgbd_imu": {
        "model_module": "baselines.MuMu.MuMu_rgbd_imu",
        "model_class": "MuMuRGBDIMU",
        "dataset_module": "datasets.utd_rgbd_imu",
        "dataset_class": "UTDMADRGBDIMUDataset",
        "data_key": "data_dir",
        "input_mode": "unpack",
        "output_mode": "mumu",
        "trainer_module": "train.mumu_train",
        "default_model_kwargs": {
            "feature_dim": 128, "cnn_d_model": 128, "freeze": "partial",
        },
        "default_dataset_kwargs": {"n_frames": 16, "frame_size": 112, "max_imu_len": 128},
        "default_trainer_kwargs": {"beta_aux": 0.5},
        "norm_keys": ["iner_mean", "iner_std"],
    },
    "skel_imu": {
        "model_module": "model.mma_skel_imu",
        "model_class": "MMA_SkeletonIMU",
        "dataset_module": "datasets.utd_skel_imu",
        "dataset_class": "UTDMADSkelIMUDataset",
        "data_key": "data_dir",
        "input_mode": "unpack",
        "output_mode": "logits",
        "trainer_module": "train.default_train",
        "default_model_kwargs": {
            "d_model": 128, "n_layers": 2, "fusion": "attention",
        },
        "default_dataset_kwargs": {"max_len_skel": 128, "max_len_imu": 192},
        "norm_keys": ["skel_mean", "skel_std", "imu_mean", "imu_std"],
    },
}


# ================================================================
#  Utilities
# ================================================================

def import_class(module_path: str, class_name: str):
    """Import a class from a dotted module path."""
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def parse_json_kwargs(text: str) -> dict:
    if not text:
        return {}
    return json.loads(text)


def _accepts_kwarg(cls, name: str) -> bool:
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    if name in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


# ================================================================
#  Dataset Creation
# ================================================================

TRAIN_SUBJECTS = [1, 3, 5, 7]
TEST_SUBJECTS = [2, 4, 6, 8]


def _build_modality_dropout_kwargs(args) -> dict | None:
    """Build `modality_dropout` dict from CLI corruption args.

    Returns None when corruption is disabled.
    """
    if args.corruption_mode == "none":
        return None
    kw = {
        "mode": args.corruption_mode,
        "p_full": args.corruption_p_full,
        "p_consecutive": args.corruption_p_consecutive,
        "consecutive_len_range": tuple(args.corruption_block_range),
        "modalities": args.corruption_modalities,
        "both_drop_prob": args.corruption_both_drop_prob,
        "noise_std": args.corruption_noise_std,
    }
    if args.corruption_noise_snr_db is not None:
        kw["noise_snr_db"] = args.corruption_noise_snr_db
    return kw


def create_test_dataset(DatasetClass, data_key, data_root,
                        default_ds_kwargs, user_ds_kwargs, norm_keys,
                        modality_dropout_kwargs=None):
    """Create train (for norm stats) and test datasets."""
    base_kw = {}
    base_kw.update(default_ds_kwargs)
    base_kw.update(user_ds_kwargs)

    # Build train set first if we need normalisation stats
    test_kw = {data_key: data_root, "subjects": TEST_SUBJECTS}
    test_kw.update(base_kw)
    if _accepts_kwarg(DatasetClass, "augment"):
        test_kw["augment"] = False
    # Inject modality dropout for robustness evaluation
    if modality_dropout_kwargs and _accepts_kwarg(DatasetClass, "modality_dropout"):
        test_kw["modality_dropout"] = modality_dropout_kwargs

    if norm_keys:
        train_kw = {data_key: data_root, "subjects": TRAIN_SUBJECTS}
        train_kw.update(base_kw)
        if _accepts_kwarg(DatasetClass, "augment"):
            train_kw["augment"] = False
        train_ds = DatasetClass(**train_kw)
        for key in norm_keys:
            if hasattr(train_ds, key):
                test_kw[key] = getattr(train_ds, key)

    test_ds = DatasetClass(**test_kw)
    return test_ds


# ================================================================
#  Trainer Module Loader
# ================================================================

def load_trainer(trainer_module_path: str):
    """Import a trainer module and return its evaluate function."""
    mod = importlib.import_module(trainer_module_path)
    return mod.get_criterion, mod.evaluate


# ================================================================
#  CLI
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Unified MMA HAR inference runner")

    p.add_argument("--pipeline", type=str, default=None,
                   choices=list(PIPELINES.keys()))
    p.add_argument("--model_module", type=str, default=None)
    p.add_argument("--model_class", type=str, default=None)
    p.add_argument("--dataset_module", type=str, default=None)
    p.add_argument("--dataset_class", type=str, default=None)
    p.add_argument("--model_kwargs", type=str, default="",
                   help="Model constructor kwargs as JSON")
    p.add_argument("--dataset_kwargs", type=str, default="",
                   help="Dataset constructor kwargs as JSON")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True, nargs="+",
                   help="Path(s) to checkpoint file(s). Multiple for ensemble.")

    p.add_argument("--input_mode", type=str, default=None,
                   choices=["unpack", "list"])
    p.add_argument("--output_mode", type=str, default=None,
                   help="How to extract logits (handled by trainer module)")
    p.add_argument("--trainer_module", type=str, default=None,
                   help="Dotted path to trainer module")
    p.add_argument("--trainer_kwargs", type=str, default="",
                   help="Trainer-specific constructor kwargs as JSON")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")

    # Corruption / modality dropout (for robustness evaluation)
    p.add_argument("--corruption_mode", type=str, default="none",
                   choices=["none", "full", "consecutive", "mixed",
                            "noise", "noise_full", "noise_consec"],
                   help="Modality dropout mode (none = disabled)")
    p.add_argument("--corruption_p_full", type=float, default=0.2,
                   help="Per-modality probability of full dropout")
    p.add_argument("--corruption_p_consecutive", type=float, default=0.3,
                   help="Per-timestep probability of starting a dropout block")
    p.add_argument("--corruption_block_range", type=int, nargs=2,
                   default=[2, 6], metavar=("MIN", "MAX"),
                   help="Min/max block length for consecutive dropout")
    p.add_argument("--corruption_modalities", type=str, nargs="+",
                   default=["rgbd", "imu"],
                   help="Modalities eligible for dropout")
    p.add_argument("--corruption_both_drop_prob", type=float, default=0.0,
                   help="Probability that both modalities are dropped")
    p.add_argument("--corruption_noise_std", type=float, default=0.5,
                   help="Gaussian noise std for noise-based corruption modes")
    p.add_argument("--corruption_noise_snr_db", type=float, default=None,
                   help="SNR in dB for noise-based modes (overrides noise_std)")

    # Comprehensive robustness evaluation
    p.add_argument("--eval_robustness", action="store_true",
                   help="Run comprehensive robustness evaluation across all "
                        "corruption modes (Centaur-style protocol)")

    # Test-time denoising (applied AFTER corruption, BEFORE model inference)
    p.add_argument("--test_denoise", type=str, default="none",
                   choices=["none", "moving_avg", "gaussian",
                            "median", "savgol", "bilateral_imu"],
                   help="Test-time denoising filter for corrupted data")
    p.add_argument("--test_denoise_k", type=int, default=3,
                   help="Kernel size for test-time denoising filter")
    p.add_argument("--test_denoise_sigma", type=float, default=0.5,
                   help="Sigma for Gaussian / bilateral test-time denoising")

    # Centaur-style Cleaning DAE (applied AFTER corruption, BEFORE model)
    p.add_argument("--cleaning_dae_ckpt", type=str, default=None,
                   help="Path to trained CleaningDAE checkpoint (.pt). "
                        "When set, corrupted data is passed through the DAE "
                        "for cleaning before HAR model inference.")

    p.add_argument("--output", type=str, default="preds.npz",
                   help="Path to save predictions (.npz)")
    p.add_argument("--compile", action="store_true",
                   help="Use torch.compile for faster inference")
    p.add_argument("--vis_dir", type=str, default="",
                   help="Directory for evaluation plots (empty = off)")
    p.add_argument("--tta", type=int, default=0,
                   help="Test-Time Augmentation passes (0 = disabled). "
                        "Averages logits over N augmented forward passes.")
    p.add_argument("--eval_missing", type=str, default="none",
                   choices=["none", "rgbd", "imu", "all"],
                   help="Evaluate with missing modality: none=full, "
                        "rgbd=drop RGBD, imu=drop IMU, all=run 3 evaluations")

    return p.parse_args()


# ================================================================
#  Missing-modality evaluation
# ================================================================

@torch.no_grad()
def _eval_missing_condition(model, loader, device, condition):
    """Evaluate with a specific missing-modality condition.
    condition: "full", "rgbd_only", "imu_only"
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    model.eval()
    all_preds, all_labels = [], []
    use_amp = (device.type == "cuda")
    for batch in loader:
        batch = [b.to(device) for b in batch]
        labels = batch[-1]
        rgbd, imu = batch[0], batch[1]
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            if condition == "rgbd_only":
                logits = model(rgbd, None)
            elif condition == "imu_only":
                logits = model(None, imu)
            else:
                logits = model(rgbd, imu)
        # Handle dict output (simultaneous mode at eval — shouldn't happen, but be safe)
        if isinstance(logits, dict):
            logits = logits["logits"]
        elif isinstance(logits, (tuple, list)):
            logits = logits[1]  # MuMu: (y_aux, y_target, alpha, attn)
        all_preds.append(logits.float().cpu().argmax(1).numpy())
        all_labels.append(labels.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = accuracy_score(labels, preds)
    _, _, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0)
    return acc, f1, preds, labels


def _eval_missing(model, loader, criterion, device, pipeline_cfg,
                  infer_args, args, full_results):
    """Run missing-modality evaluation and print summary table."""
    conditions = []
    if args.eval_missing in ("none",):
        return
    if args.eval_missing == "all":
        conditions = ["full", "rgbd_only", "imu_only"]
    elif args.eval_missing == "rgbd":
        conditions = ["rgbd_only"]
    elif args.eval_missing == "imu":
        conditions = ["imu_only"]

    results_table = {}
    for cond in conditions:
        if cond == "full":
            results_table[cond] = (full_results["acc"], full_results["f1"])
        else:
            acc, f1, _, _ = _eval_missing_condition(model, loader, device, cond)
            results_table[cond] = (acc, f1)

    # Print table
    print(f"\n{'='*60}")
    print(f"Missing-Modality Evaluation")
    print(f"{'='*60}")
    print(f"{'Condition':<20} {'Accuracy':>10} {'F1':>10}")
    print(f"{'-'*40}")
    for cond in conditions:
        acc, f1 = results_table[cond]
        label = {"full": "Full (both)", "rgbd_only": "RGBD-only",
                 "imu_only": "IMU-only"}.get(cond, cond)
        print(f"{label:<20} {acc:>10.4f} {f1:>10.4f}")
    if len(results_table) == 3:
        avg_acc = sum(v[0] for v in results_table.values()) / 3
        avg_f1 = sum(v[1] for v in results_table.values()) / 3
        min_acc = min(v[0] for k, v in results_table.items() if k != "full")
        degradation = results_table["full"][0] - min_acc
        print(f"{'-'*40}")
        print(f"{'Average':<20} {avg_acc:>10.4f} {avg_f1:>10.4f}")
        print(f"{'Degradation':<20} {degradation:>10.4f}")
    print(f"{'='*60}")


# ================================================================
#  Test-time denoising filters
# ================================================================

def _build_test_denoise_fn(args):
    """Build a test-time denoising function from CLI args.

    Returns a function (rgbd, imu) -> (rgbd, imu) that denoises GPU tensors,
    or None if no denoising is requested.
    """
    mode = getattr(args, "test_denoise", "none")
    if mode == "none":
        return None

    k = getattr(args, "test_denoise_k", 3)
    sigma = getattr(args, "test_denoise_sigma", 0.5)

    # Build 2D Gaussian kernel for RGBD spatial denoising (shared by all modes)
    k2d = k if k % 2 == 1 else k + 1
    half2d = k2d // 2
    ax = torch.arange(-half2d, half2d + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    g2d = torch.exp(-0.5 * (xx ** 2 + yy ** 2) / (sigma ** 2))
    g2d = g2d / g2d.sum()

    def _blur_rgbd(rgbd, _g2d=g2d):
        """Apply spatial 2D Gaussian blur to each RGBD frame independently."""
        B, T, C, H, W = rgbd.shape
        kernel = _g2d.to(rgbd.device, rgbd.dtype).unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand(C, 1, -1, -1)
        frames = rgbd.reshape(B * T, C, H, W)
        frames = torch.nn.functional.conv2d(
            frames, kernel, padding=half2d, groups=C,
        )
        return frames.reshape(B, T, C, H, W)

    if mode == "moving_avg":
        k = k if k % 2 == 1 else k + 1
        pad = k // 2

        def _ma(rgbd, imu):
            # RGBD: spatial 2D Gaussian blur per frame
            rgbd_s = _blur_rgbd(rgbd)
            # IMU: temporal moving average per channel
            B, T, C = imu.shape
            kernel = torch.ones(C, 1, k, device=imu.device, dtype=imu.dtype) / k
            imu_s = torch.nn.functional.conv1d(
                imu.transpose(1, 2), kernel, padding=pad, groups=C,
            ).transpose(1, 2)
            return rgbd_s, imu_s

        return _ma

    elif mode == "gaussian":
        k = k if k % 2 == 1 else k + 1
        half = k // 2
        x = torch.arange(-half, half + 1, dtype=torch.float32)
        g = torch.exp(-0.5 * (x / sigma) ** 2)
        g = g / g.sum()

        def _gauss(rgbd, imu):
            # RGBD: spatial 2D Gaussian blur per frame
            rgbd_s = _blur_rgbd(rgbd)
            # IMU: temporal 1D Gaussian per channel
            B, T, C = imu.shape
            kernel = g.to(imu.device, imu.dtype).unsqueeze(0).unsqueeze(0).expand(C, 1, -1)
            imu_s = torch.nn.functional.conv1d(
                imu.transpose(1, 2), kernel, padding=half, groups=C,
            ).transpose(1, 2)
            return rgbd_s, imu_s

        return _gauss

    elif mode == "median":
        k = k if k % 2 == 1 else k + 1
        pad = k // 2

        def _median(rgbd, imu):
            # RGBD: spatial 2D Gaussian blur per frame
            rgbd_s = _blur_rgbd(rgbd)
            # IMU: temporal median per channel
            B, T, C = imu.shape
            imu_padded = torch.nn.functional.pad(
                imu.transpose(1, 2), (pad, pad), mode="reflect"
            )
            imu_unf = imu_padded.unfold(2, k, 1)
            imu_s = imu_unf.median(dim=-1).values.transpose(1, 2)
            return rgbd_s, imu_s

        return _median

    elif mode == "savgol":
        # Savitzky-Golay filter: polynomial least-squares fitting
        # Order 2 polynomial, window size k
        from scipy.signal import savgol_coeffs
        k = k if k % 2 == 1 else k + 1
        coeffs = savgol_coeffs(k, min(2, k - 1))  # 2nd order poly
        half = k // 2

        def _savgol(rgbd, imu):
            # RGBD: spatial 2D Gaussian blur per frame
            rgbd_s = _blur_rgbd(rgbd)
            # IMU: Savitzky-Golay per channel
            B, T, C = imu.shape
            kernel_t = torch.tensor(
                coeffs, device=imu.device, dtype=imu.dtype
            ).unsqueeze(0).unsqueeze(0).expand(C, 1, -1)
            imu_s = torch.nn.functional.conv1d(
                imu.transpose(1, 2), kernel_t, padding=half, groups=C,
            ).transpose(1, 2)
            return rgbd_s, imu_s

        return _savgol

    elif mode == "bilateral_imu":
        # 1D bilateral filter for IMU: range-weighted + spatial Gaussian
        k = k if k % 2 == 1 else k + 1
        half = k // 2

        def _bilateral(rgbd, imu):
            # RGBD: spatial 2D Gaussian blur per frame
            rgbd_s = _blur_rgbd(rgbd)
            # IMU: bilateral 1D filter per channel
            B, T, C = imu.shape
            imu_padded = torch.nn.functional.pad(
                imu.transpose(1, 2), (half, half), mode="reflect"
            )  # (B, C, T+2*half)
            imu_unf = imu_padded.unfold(2, k, 1)  # (B, C, T, k)
            center = imu.transpose(1, 2).unsqueeze(-1)  # (B, C, T, 1)
            # Range weights: penalise neighbours with very different values
            range_w = torch.exp(-0.5 * ((imu_unf - center) / (sigma + 1e-6)) ** 2)
            # Spatial weights: Gaussian over position
            pos = torch.arange(-half, half + 1, device=imu.device, dtype=imu.dtype)
            spatial_w = torch.exp(-0.5 * (pos / max(sigma, 0.3)) ** 2)
            w = range_w * spatial_w  # (B, C, T, k)
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
            imu_s = (imu_unf * w).sum(dim=-1).transpose(1, 2)  # (B, T, C)
            return rgbd_s, imu_s

        return _bilateral

    return None


# ================================================================
#  Comprehensive Robustness Evaluation (Centaur-style)
# ================================================================

def _eval_robustness(model, DatasetClass, data_key, data_root,
                     default_ds_kwargs, user_ds_kw, norm_keys,
                     device, batch_size, num_workers, args):
    """Run Centaur-style comprehensive robustness evaluation.

    Implements the four corruption modes from:
      Xaviar et al., "Centaur: Robust Multimodal Fusion for HAR",
      IEEE Sensors Journal, 2024.  (arXiv: 2303.04636)

    Mode 1: Additive white Gaussian noise N(0,sigma) per channel per timestep
    Mode 2: Per-channel consecutive missing (exponential intervals, independent)
    Mode 3: Per-sensor consecutive missing (all channels of a modality together)
    Mode 4: Mode 1 + Mode 2 combined (noise first, then per-channel missing)
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    # Paper's noise levels (sigma on normalised data)
    SIGMA_LEVELS = [0.05, 0.1, 0.2, 0.3]
    # Consecutive missing: s_corr levels (s_norm is fixed)
    # RGBD has T=16 frames → s_norm=8, s_corr ∈ {2,4,6}
    # IMU  has T=128 steps → s_norm=60, s_corr ∈ {15,30,45}
    RGBD_S_NORM = 8
    RGBD_S_CORR = [2, 4, 6]
    IMU_S_NORM  = 60
    IMU_S_CORR  = [15, 30, 45]
    N_TRIALS = 3

    results = {}

    # Build test-time denoise function (if any)
    denoise_fn = _build_test_denoise_fn(args)
    if denoise_fn is not None:
        print(f"  Test-time denoising: {args.test_denoise} "
              f"(k={args.test_denoise_k}, σ={args.test_denoise_sigma})")

    # Build Centaur-style Cleaning DAE (if checkpoint provided)
    cleaning_dae = None
    dae_ckpt_path = getattr(args, "cleaning_dae_ckpt", None)
    if dae_ckpt_path:
        from model.cleaning_dae import CleaningDAE
        dae_ckpt = torch.load(dae_ckpt_path, map_location=device, weights_only=False)
        cleaning_dae = CleaningDAE(
            enable_imu=dae_ckpt.get("enable_imu", True),
            enable_rgbd=dae_ckpt.get("enable_rgbd", True),
            imu_latent_dim=dae_ckpt.get("latent_dim", 64),
        ).to(device)
        cleaning_dae.load_state_dict(dae_ckpt["model_state"])
        cleaning_dae.eval()
        print(f"  Cleaning DAE loaded: {dae_ckpt_path} "
              f"(imu={dae_ckpt.get('enable_imu')}, "
              f"rgbd={dae_ckpt.get('enable_rgbd')})")

    # Build a single clean test dataset and cache data
    clean_ds = create_test_dataset(
        DatasetClass, data_key, data_root,
        default_ds_kwargs, user_ds_kw, norm_keys,
        modality_dropout_kwargs=None,
    )
    clean_loader = DataLoader(
        clean_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    @torch.no_grad()
    def _evaluate_loader(loader):
        model.eval()
        all_preds, all_labels = [], []
        use_amp = (device.type == "cuda")
        for batch in loader:
            batch = [b.to(device) for b in batch]
            labels = batch[-1]
            rgbd, imu = batch[0], batch[1]
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(rgbd, imu)
            if isinstance(logits, dict):
                logits = logits["logits"]
            elif isinstance(logits, (tuple, list)):
                logits = logits[1]
            all_preds.append(logits.float().cpu().argmax(1).numpy())
            all_labels.append(labels.cpu().numpy())
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        acc = accuracy_score(labels, preds)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0)
        return acc, f1

    @torch.no_grad()
    def _evaluate_corrupted(corruption_fn, n_trials=N_TRIALS):
        """Apply corruption to cached tensors and evaluate."""
        model.eval()
        use_amp = (device.type == "cuda")
        accs, f1s = [], []
        for _ in range(n_trials):
            all_preds, all_labels = [], []
            for batch in clean_loader:
                batch = [b.to(device) for b in batch]
                labels = batch[-1]
                rgbd, imu = batch[0].clone(), batch[1].clone()
                rgbd, imu = corruption_fn(rgbd, imu)
                # Apply test-time denoising (after corruption, before model)
                if denoise_fn is not None:
                    rgbd, imu = denoise_fn(rgbd, imu)
                # Apply Cleaning DAE (after corruption/denoise, before model)
                if cleaning_dae is not None:
                    rgbd, imu = cleaning_dae(rgbd, imu)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(rgbd, imu)
                if isinstance(logits, dict):
                    logits = logits["logits"]
                elif isinstance(logits, (tuple, list)):
                    logits = logits[1]
                all_preds.append(logits.float().cpu().argmax(1).numpy())
                all_labels.append(labels.cpu().numpy())
            preds = np.concatenate(all_preds)
            labels = np.concatenate(all_labels)
            acc = accuracy_score(labels, preds)
            _, _, f1, _ = precision_recall_fscore_support(
                labels, preds, average="weighted", zero_division=0)
            accs.append(acc)
            f1s.append(f1)
        return np.mean(accs), np.mean(f1s)

    # --- Corruption primitives (GPU tensors) ---

    def _add_gaussian_noise_sigma(data, sigma):
        """Mode 1: Add N(0, sigma) noise independently per channel per timestep."""
        return data + torch.randn_like(data) * sigma

    def _consec_missing_per_channel(data, s_norm, s_corr):
        """Mode 2: Per-channel independent consecutive missing.

        For each sample and each channel, alternate between normal (Exp(1/s_norm))
        and corrupted (Exp(1/s_corr)) intervals. Missing values are set to zero.
        data shape: (..., T, C) for IMU or (B, T, C, H, W) for RGBD.
        """
        if data.dim() == 3:
            # IMU: (B, T, C)
            B, T, C = data.shape
            for b in range(B):
                for c in range(C):
                    t = 0
                    normal_phase = True
                    while t < T:
                        if normal_phase:
                            length = int(np.random.exponential(s_norm)) + 1
                        else:
                            length = int(np.random.exponential(s_corr)) + 1
                            end = min(t + length, T)
                            data[b, t:end, c] = 0.0
                            t = end
                            normal_phase = True
                            continue
                        t += length
                        normal_phase = False
        elif data.dim() == 5:
            # RGBD: (B, T, C, H, W)
            B, T, C, H, W = data.shape
            for b in range(B):
                for c in range(C):
                    t = 0
                    normal_phase = True
                    while t < T:
                        if normal_phase:
                            length = int(np.random.exponential(s_norm)) + 1
                        else:
                            length = int(np.random.exponential(s_corr)) + 1
                            end = min(t + length, T)
                            data[b, t:end, c, :, :] = 0.0
                            t = end
                            normal_phase = True
                            continue
                        t += length
                        normal_phase = False
        return data

    def _consec_missing_per_sensor(data, s_norm, s_corr):
        """Mode 3: Per-sensor consecutive missing (all channels of a modality
        share the same missing pattern).

        Same as Mode 2 except all channels of a sensor share the same intervals.
        """
        if data.dim() == 3:
            # IMU: (B, T, C) — treat entire sensor as one unit
            B, T, C = data.shape
            for b in range(B):
                t = 0
                normal_phase = True
                while t < T:
                    if normal_phase:
                        length = int(np.random.exponential(s_norm)) + 1
                    else:
                        length = int(np.random.exponential(s_corr)) + 1
                        end = min(t + length, T)
                        data[b, t:end, :] = 0.0
                        t = end
                        normal_phase = True
                        continue
                    t += length
                    normal_phase = False
        elif data.dim() == 5:
            # RGBD: (B, T, C, H, W) — all 4 channels share pattern
            B, T, C, H, W = data.shape
            for b in range(B):
                t = 0
                normal_phase = True
                while t < T:
                    if normal_phase:
                        length = int(np.random.exponential(s_norm)) + 1
                    else:
                        length = int(np.random.exponential(s_corr)) + 1
                        end = min(t + length, T)
                        data[b, t:end, :, :, :] = 0.0
                        t = end
                        normal_phase = True
                        continue
                    t += length
                    normal_phase = False
        return data

    # ---- 1. Clean evaluation ----
    print("\n[1/4] Clean (no corruption)...")
    acc, f1 = _evaluate_loader(clean_loader)
    results["clean"] = (acc, f1)
    print(f"  Clean: Acc={acc:.4f}  F1={f1:.4f}")

    # ---- 2. Mode 1: Gaussian noise ----
    print("\n[2/4] Mode 1: Gaussian noise N(0,σ) on all channels...")
    for sigma in SIGMA_LEVELS:
        def noise_fn(rgbd, imu, _s=sigma):
            return _add_gaussian_noise_sigma(rgbd, _s), _add_gaussian_noise_sigma(imu, _s)
        acc, f1 = _evaluate_corrupted(noise_fn)
        results[f"mode1_sigma{sigma}"] = (acc, f1)
        print(f"  σ={sigma:.2f}: Acc={acc:.4f}  F1={f1:.4f}")

    # ---- 3. Mode 2: Per-channel consecutive missing ----
    print("\n[3/4] Mode 2: Per-channel consecutive missing...")
    for i, (rc, ic) in enumerate(zip(RGBD_S_CORR, IMU_S_CORR)):
        def consec_ch_fn(rgbd, imu, _rc=rc, _ic=ic):
            return (_consec_missing_per_channel(rgbd, RGBD_S_NORM, _rc),
                    _consec_missing_per_channel(imu, IMU_S_NORM, _ic))
        acc, f1 = _evaluate_corrupted(consec_ch_fn)
        label = f"mode2_rc{rc}_ic{ic}"
        results[label] = (acc, f1)
        print(f"  s_corr(rgbd={rc},imu={ic}): Acc={acc:.4f}  F1={f1:.4f}")

    # ---- 4. Mode 3: Per-sensor consecutive missing ----
    print("\n[3.5/4] Mode 3: Per-sensor consecutive missing...")
    for i, (rc, ic) in enumerate(zip(RGBD_S_CORR, IMU_S_CORR)):
        def consec_sensor_fn(rgbd, imu, _rc=rc, _ic=ic):
            return (_consec_missing_per_sensor(rgbd, RGBD_S_NORM, _rc),
                    _consec_missing_per_sensor(imu, IMU_S_NORM, _ic))
        acc, f1 = _evaluate_corrupted(consec_sensor_fn)
        label = f"mode3_rc{rc}_ic{ic}"
        results[label] = (acc, f1)
        print(f"  s_corr(rgbd={rc},imu={ic}): Acc={acc:.4f}  F1={f1:.4f}")

    # ---- 5. Mode 4: Noise + per-channel consecutive missing ----
    print("\n[4/4] Mode 4: Noise + per-channel consecutive missing...")
    mode4_configs = [
        (0.05, RGBD_S_CORR[0], IMU_S_CORR[0]),
        (0.1,  RGBD_S_CORR[1], IMU_S_CORR[1]),
        (0.2,  RGBD_S_CORR[2], IMU_S_CORR[2]),
    ]
    for sigma, rc, ic in mode4_configs:
        def mode4_fn(rgbd, imu, _s=sigma, _rc=rc, _ic=ic):
            # First noise, then per-channel missing (paper: c4 = c2(c1(x)))
            rgbd = _add_gaussian_noise_sigma(rgbd, _s)
            imu  = _add_gaussian_noise_sigma(imu, _s)
            rgbd = _consec_missing_per_channel(rgbd, RGBD_S_NORM, _rc)
            imu  = _consec_missing_per_channel(imu, IMU_S_NORM, _ic)
            return rgbd, imu
        acc, f1 = _evaluate_corrupted(mode4_fn)
        label = f"mode4_s{sigma}_rc{rc}_ic{ic}"
        results[label] = (acc, f1)
        print(f"  σ={sigma:.2f}, s_corr(rgbd={rc},imu={ic}): Acc={acc:.4f}  F1={f1:.4f}")

    # ---- Summary tables (one per mode) ----
    clean_acc, clean_f1 = results["clean"]
    print(f"\n{'='*70}")
    print(f"  CENTAUR-STYLE ROBUSTNESS EVALUATION — SUMMARY")
    print(f"{'='*70}")

    # Table 1: Clean
    print(f"\n--- Baseline ---")
    print(f"  {'Condition':<40} {'Acc':>8} {'F1':>8}")
    print(f"  {'Clean (no corruption)':<40} {clean_acc:>8.4f} {clean_f1:>8.4f}")

    # Table 2: Mode 1 — Gaussian noise
    print(f"\n--- Mode 1: Gaussian Noise N(0,σ) ---")
    print(f"  {'σ':<40} {'Acc':>8} {'F1':>8}")
    for sigma in SIGMA_LEVELS:
        k = f"mode1_sigma{sigma}"
        acc, f1 = results[k]
        print(f"  {sigma:<40.2f} {acc:>8.4f} {f1:>8.4f}")

    # Table 3: Mode 2 — per-channel consecutive missing
    print(f"\n--- Mode 2: Per-Channel Consecutive Missing ---")
    print(f"  {'s_corr (rgbd, imu)':<40} {'Acc':>8} {'F1':>8}")
    for rc, ic in zip(RGBD_S_CORR, IMU_S_CORR):
        k = f"mode2_rc{rc}_ic{ic}"
        acc, f1 = results[k]
        print(f"  {'('+str(rc)+', '+str(ic)+')':<40} {acc:>8.4f} {f1:>8.4f}")

    # Table 4: Mode 3 — per-sensor consecutive missing
    print(f"\n--- Mode 3: Per-Sensor Consecutive Missing ---")
    print(f"  {'s_corr (rgbd, imu)':<40} {'Acc':>8} {'F1':>8}")
    for rc, ic in zip(RGBD_S_CORR, IMU_S_CORR):
        k = f"mode3_rc{rc}_ic{ic}"
        acc, f1 = results[k]
        print(f"  {'('+str(rc)+', '+str(ic)+')':<40} {acc:>8.4f} {f1:>8.4f}")

    # Table 5: Mode 4 — combined
    print(f"\n--- Mode 4: Noise + Per-Channel Missing (c4=c2∘c1) ---")
    print(f"  {'σ, s_corr (rgbd, imu)':<40} {'Acc':>8} {'F1':>8}")
    for sigma, rc, ic in mode4_configs:
        k = f"mode4_s{sigma}_rc{rc}_ic{ic}"
        acc, f1 = results[k]
        print(f"  {f'{sigma:.2f}, ({rc}, {ic})':<40} {acc:>8.4f} {f1:>8.4f}")

    # Overall summary
    corrupted = {k: v for k, v in results.items() if k != "clean"}
    avg_acc = np.mean([v[0] for v in corrupted.values()])
    avg_f1  = np.mean([v[1] for v in corrupted.values()])
    print(f"\n{'-'*60}")
    print(f"  {'Avg (all corrupted)':<40} {avg_acc:>8.4f} {avg_f1:>8.4f}")
    print(f"  {'Robustness ratio (corrupted/clean)':<40} {avg_acc/clean_acc:>8.4f}")
    print(f"{'='*70}")

    return results


# ================================================================
#  Main
# ================================================================

def main():
    args = parse_args()
    device = torch.device(args.device)

    checkpoints = args.checkpoint if isinstance(args.checkpoint, list) else [args.checkpoint]
    is_ensemble = len(checkpoints) > 1

    # -- Load first checkpoint for config --
    ckpt = torch.load(checkpoints[0], map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})

    # -- Resolve pipeline config --
    pipeline_name = args.pipeline or saved_args.get("pipeline")
    cfg = PIPELINES.get(pipeline_name, {}).copy() if pipeline_name else {}

    model_module = args.model_module or cfg.get("model_module")
    model_class_name = args.model_class or cfg.get("model_class")
    dataset_module = args.dataset_module or cfg.get("dataset_module")
    dataset_class_name = args.dataset_class or cfg.get("dataset_class")
    input_mode = args.input_mode or cfg.get("input_mode", "unpack")
    output_mode = args.output_mode or cfg.get("output_mode", "logits")
    data_key = cfg.get("data_key", "data_root")
    norm_keys = cfg.get("norm_keys", [])
    trainer_module_path = (args.trainer_module
                           or cfg.get("trainer_module", "train.default_train"))

    if not (model_module and model_class_name):
        raise ValueError("Specify --pipeline or --model_module / --model_class")
    if not (dataset_module and dataset_class_name):
        raise ValueError("Specify --pipeline or --dataset_module / --dataset_class")

    # -- Import classes --
    ModelClass = import_class(model_module, model_class_name)
    DatasetClass = import_class(dataset_module, dataset_class_name)

    # -- Merge kwargs: registry defaults < checkpoint saved < CLI --
    model_kwargs = cfg.get("default_model_kwargs", {}).copy()
    model_kwargs.update(ckpt.get("model_kwargs", {}))
    model_kwargs.update(parse_json_kwargs(args.model_kwargs))

    user_ds_kw = parse_json_kwargs(args.dataset_kwargs)

    # -- Corruption / modality dropout --
    modality_dropout_kwargs = _build_modality_dropout_kwargs(args)
    if modality_dropout_kwargs:
        print(f"Corruption: mode={modality_dropout_kwargs['mode']}, "
              f"p_full={modality_dropout_kwargs['p_full']}, "
              f"p_consec={modality_dropout_kwargs['p_consecutive']}")

    # -- Model --
    model = ModelClass(**model_kwargs).to(device)

    state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    # -- torch.compile --
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("torch.compile enabled")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_class_name}  params={n_params:,}")
    print(f"Checkpoint(s): {', '.join(checkpoints)}")

    # -- Dataset --
    test_ds = create_test_dataset(
        DatasetClass, data_key, args.data_root,
        cfg.get("default_dataset_kwargs", {}),
        user_ds_kw, norm_keys,
        modality_dropout_kwargs=modality_dropout_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Test samples: {len(test_ds)}")

    # -- Load trainer module --
    get_criterion, do_evaluate = load_trainer(trainer_module_path)
    trainer_kwargs = cfg.get("default_trainer_kwargs", {}).copy()
    trainer_kwargs.update(parse_json_kwargs(args.trainer_kwargs))

    # Build a minimal args-like object for the trainer's evaluate()
    class _InferArgs:
        label_smoothing = 0.0
        clip_grad = 0.0
        mixup_alpha = 0.0
    infer_args = _InferArgs()

    criterion = get_criterion(infer_args, **trainer_kwargs)
    pipeline_cfg = {"input_mode": input_mode, "output_mode": output_mode}

    # -- Inference (single or ensemble) --
    if is_ensemble:
        print(f"\nEnsemble mode: {len(checkpoints)} models")
        all_logits = []
        labels = None
        for i, ckpt_path in enumerate(checkpoints):
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_i = ck.get("model_state") or ck.get("state_dict") or ck
            model.load_state_dict(state_i)
            model.eval()
            res = do_evaluate(model, test_loader, criterion, device, pipeline_cfg, infer_args)
            all_logits.append(res.get("logits"))
            if labels is None:
                labels = res["labels"]
            print(f"  Model {i+1}: Acc={res['acc']:.4f} F1={res['f1']:.4f}")

        # Average logits and re-compute metrics
        if all_logits[0] is not None:
            avg_logits = np.mean(all_logits, axis=0)
            preds = np.argmax(avg_logits, axis=1)
        else:
            # Fallback: majority vote on predictions
            from scipy.stats import mode as scipy_mode
            all_preds = np.array([r.get("preds") for r in [
                do_evaluate(model, test_loader, criterion, device, pipeline_cfg, infer_args)
            ]])
            preds = scipy_mode(all_preds, axis=0).mode.flatten()

        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        acc = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0,
        )
        n = len(labels)
        results = {"acc": acc, "prec": prec, "rec": rec, "f1": f1,
                    "preds": preds, "labels": labels, "loss": 0.0}
    else:
        results = do_evaluate(model, test_loader, criterion, device, pipeline_cfg, infer_args)
        preds = results["preds"]
        labels = results["labels"]
        n = len(labels)
        acc = results["acc"]
        prec = results["prec"]
        rec = results["rec"]
        f1 = results["f1"]

    # -- Test-Time Augmentation (TTA) --
    if args.tta > 0 and not is_ensemble:
        print(f"\nTTA: averaging over {args.tta} augmented passes...")
        # Build augmented test dataset
        tta_ds = create_test_dataset(
            DatasetClass, data_key, args.data_root,
            cfg.get("default_dataset_kwargs", {}),
            user_ds_kw, norm_keys,
        )
        tta_ds.augment = True  # Enable augmentation
        tta_loader = DataLoader(
            tta_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        # Collect logits from N augmented passes + 1 clean pass
        tta_logits = [results["logits"]]  # clean pass
        for i in range(args.tta):
            res_i = do_evaluate(model, tta_loader, criterion, device, pipeline_cfg, infer_args)
            tta_logits.append(res_i["logits"])
        avg_logits = np.mean(tta_logits, axis=0)
        preds = np.argmax(avg_logits, axis=1)
        from sklearn.metrics import accuracy_score as _acc_score, \
            precision_recall_fscore_support as _prf
        acc = _acc_score(labels, preds)
        prec, rec, f1, _ = _prf(labels, preds, average="weighted", zero_division=0)
        results.update({"acc": acc, "prec": prec, "rec": rec, "f1": f1, "preds": preds})
        print(f"  TTA Results: Acc={acc:.4f} F1={f1:.4f}")

    # -- Missing-modality evaluation --
    if args.eval_missing != "none" and not is_ensemble:
        _eval_missing(model, test_loader, criterion, device, pipeline_cfg,
                      infer_args, args, results)
        return

    # -- Comprehensive robustness evaluation (Centaur-style) --
    if args.eval_robustness and not is_ensemble:
        _eval_robustness(model, DatasetClass, data_key, args.data_root,
                         cfg.get("default_dataset_kwargs", {}),
                         user_ds_kw, norm_keys,
                         device, args.batch_size, args.num_workers, args)
        return

    print(f"\nTest Results ({n} samples):")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  Loss     : {results['loss']:.4f}")

    try:
        from datasets import ACTION_NAMES
        class_names = ACTION_NAMES
    except Exception:
        class_names = None

    print("\nClassification Report:")
    print(classification_report(
        labels, preds,
        target_names=class_names[:max(labels.max(), preds.max()) + 1] if class_names else None,
        zero_division=0,
    ))

    # -- Save predictions --
    save_dict = {"preds": preds, "labels": labels}
    np.savez_compressed(args.output, **save_dict)
    print(f"Predictions saved to {args.output}")

    # -- Visualisations --
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)
        plot_confusion_matrix(
            labels, preds,
            class_names=class_names,
            save_dir=args.vis_dir,
        )
        plot_per_class_accuracy(
            labels, preds,
            class_names=class_names,
            save_dir=args.vis_dir,
        )
        print(f"Plots saved to {args.vis_dir}/")


if __name__ == "__main__":
    main()
