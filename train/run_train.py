"""
Unified Training Runner
========================
Supports all MMA pipelines (utdmad, mmtsa, rgbd_imu) and baselines (mumu).
Models and datasets are dynamically imported based on pipeline config.
Each pipeline can specify a **trainer module** (e.g. ``train.mumu_train``)
that provides its own ``get_criterion``, ``train_one_epoch``, and
``evaluate`` functions.  Pipelines without a trainer module fall back to
``train.default_train`` (standard CE loss).

Usage:
	python train/run_train.py --pipeline rgbd_imu --data_root datasets/UTD-MHAD \
  --corruption_mode consecutive --corruption_p_consecutive 0.4

  python train/run_train.py --pipeline utdmad --data_root datasets/UTD-MHAD/Inertial
  python train/run_train.py --pipeline mmtsa  --data_root datasets/UTD-MHAD \\
         --model_kwargs '{"fusion":"gated"}'
  python train/run_train.py --pipeline mumu   --data_root datasets/UTD-MHAD/Inertial

  # Custom model (not in registry):
  python train/run_train.py \\
         --model_module model.mma_utdmad --model_class MomentumMambaHAR \\
         --dataset_module datasets.utd_inertial --dataset_class UTDMADInertialDataset \\
         --data_root datasets/UTD-MHAD/Inertial --input_mode unpack
"""

import argparse
import importlib
import inspect
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.optim.swa_utils import AveragedModel, SWALR

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from torch.utils.tensorboard import SummaryWriter

from vis.training import TrainingLogger
from vis.evaluation import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_per_class_accuracy,
)


# ================================================================
#  Pipeline Registry
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
        "default_model_kwargs": {"fusion": "cross_mamba", "d_model": 160},
        "default_dataset_kwargs": {"n_frames": 16, "frame_size": 112, "max_imu_len": 128},
        "norm_keys": ["iner_mean", "iner_std"],
    },
    "staged_rgbd_imu": {
        "model_module": "model.mma_rgbd_imu",
        "model_class": "MultimodalMMA",
        "dataset_module": "datasets.utd_rgbd_imu",
        "dataset_class": "UTDMADRGBDIMUDataset",
        "data_key": "data_dir",
        "input_mode": "unpack",
        "output_mode": "logits",
        "trainer_module": "train.staged_train",
        "default_model_kwargs": {"fusion": "cross_mamba", "d_model": 128,
                                  "aux_weight": 0.3, "dropout": 0.4,
                                  "encoder": "pretrained", "freeze": "partial",
                                  "temporal_velocity": True,
                                  "md_schedule": "none"},
        "default_dataset_kwargs": {"n_frames": 16, "frame_size": 112, "max_imu_len": 128},
        "default_trainer_kwargs": {
            "phase1_end": 20, "phase2_end": 40,
            "phase1_lr": 3e-4, "phase2_lr": 1e-4, "phase3_lr": 3e-4,
            "phase2_aux_weight": 0.3, "phase3_aux_weight": 0.3,
            "phase3_rgbd_lr_scale": 0.1, "staged_warmup_epochs": 5,
        },
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
    """Parse a JSON string into a dict. Returns {} for empty input."""
    if not text:
        return {}
    return json.loads(text)


def _accepts_kwarg(cls, name: str) -> bool:
    """Check if a class __init__ accepts a specific keyword argument."""
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
    return {
        "mode": args.corruption_mode,
        "p_full": args.corruption_p_full,
        "p_consecutive": args.corruption_p_consecutive,
        "consecutive_len_range": tuple(args.corruption_block_range),
        "modalities": args.corruption_modalities,
        "both_drop_prob": args.corruption_both_drop_prob,
    }


def create_datasets(DatasetClass, data_key, data_root,
                    default_ds_kwargs, user_ds_kwargs, norm_keys,
                    modality_dropout_kwargs=None,
                    sensor_corruption_kwargs=None):
    """Create train and test datasets with proper normalisation propagation."""
    base_kw = {}
    base_kw.update(default_ds_kwargs)
    base_kw.update(user_ds_kwargs)

    # -- Train --
    train_kw = {data_key: data_root, "subjects": TRAIN_SUBJECTS}
    train_kw.update(base_kw)
    if _accepts_kwarg(DatasetClass, "augment"):
        train_kw.setdefault("augment", True)
    # Inject modality dropout into training set only
    if modality_dropout_kwargs and _accepts_kwarg(DatasetClass, "modality_dropout"):
        train_kw["modality_dropout"] = modality_dropout_kwargs
    # Inject sensor corruption augmentation into training set only
    if sensor_corruption_kwargs and _accepts_kwarg(DatasetClass, "sensor_corruption"):
        train_kw["sensor_corruption"] = sensor_corruption_kwargs
    train_ds = DatasetClass(**train_kw)

    # -- Test --
    test_kw = {data_key: data_root, "subjects": TEST_SUBJECTS}
    test_kw.update(base_kw)
    if _accepts_kwarg(DatasetClass, "augment"):
        test_kw["augment"] = False
    for key in norm_keys:
        if hasattr(train_ds, key):
            test_kw[key] = getattr(train_ds, key)
    test_ds = DatasetClass(**test_kw)

    return train_ds, test_ds


# ================================================================
#  Trainer Module Loader
# ================================================================

def load_trainer(trainer_module_path: str):
    """Import a trainer module and return its interface functions.

    A trainer module must export:
        get_criterion(args, **kwargs) -> nn.Module
        train_one_epoch(model, loader, optimizer, criterion, device, cfg, args) -> (loss, acc)
        evaluate(model, loader, criterion, device, cfg, args) -> metrics_dict

    Returns: (get_criterion, train_one_epoch, evaluate)
    """
    mod = importlib.import_module(trainer_module_path)
    return mod.get_criterion, mod.train_one_epoch, mod.evaluate


# ================================================================
#  Warmup + Cosine Scheduler
# ================================================================

class WarmupCosineScheduler:
    """Linear warmup then cosine decay."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            frac = (epoch + 1) / self.warmup_epochs
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = base_lr * frac
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = self.eta_min + (base_lr - self.eta_min) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )


# ================================================================
#  CLI
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Unified MMA HAR training runner")

    # Pipeline selection
    p.add_argument("--pipeline", type=str, default=None,
                   choices=list(PIPELINES.keys()),
                   help="Predefined pipeline name")

    # Custom model / dataset (override or standalone)
    p.add_argument("--model_module", type=str, default=None,
                   help="Dotted module path for model class")
    p.add_argument("--model_class", type=str, default=None,
                   help="Model class name")
    p.add_argument("--dataset_module", type=str, default=None,
                   help="Dotted module path for dataset class")
    p.add_argument("--dataset_class", type=str, default=None,
                   help="Dataset class name")

    # Kwargs (JSON)
    p.add_argument("--model_kwargs", type=str, default="",
                   help='Model constructor kwargs as JSON, e.g. \'{"fusion":"gated"}\'')
    p.add_argument("--dataset_kwargs", type=str, default="",
                   help="Dataset constructor kwargs as JSON")

    # Data
    p.add_argument("--data_root", type=str, required=True,
                   help="Root path to dataset")

    # IO modes (auto-detected from pipeline, or set manually)
    p.add_argument("--input_mode", type=str, default=None,
                   choices=["unpack", "list"],
                   help="unpack: model(*inputs)  |  list: model(inputs)")
    p.add_argument("--output_mode", type=str, default=None,
                   help="How to extract logits (handled by trainer module)")
    p.add_argument("--trainer_module", type=str, default=None,
                   help="Dotted path to trainer module (e.g. train.mumu_train)")
    p.add_argument("--trainer_kwargs", type=str, default="",
                   help="Trainer-specific constructor kwargs as JSON")

    # Training hyperparams
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adam", "adamw"])
    p.add_argument("--scheduler", type=str, default="cosine",
                   choices=["cosine", "cosine_warmup", "step", "none"])
    p.add_argument("--warmup_epochs", type=int, default=10,
                   help="Warmup epochs (for cosine_warmup scheduler)")
    p.add_argument("--step_size", type=int, default=30,
                   help="Step size for StepLR scheduler")
    p.add_argument("--step_gamma", type=float, default=0.1)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--mixup_alpha", type=float, default=0.0,
                   help="Mixup alpha (0 = disabled)")

    # Corruption / modality dropout
    p.add_argument("--corruption_mode", type=str, default="none",
                   choices=["none", "full", "consecutive", "mixed"],
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

    # Sensor corruption augmentation (Centaur-style, training-time)
    p.add_argument("--sensor_corrupt_p", type=float, default=0.0,
                   help="Probability of applying sensor corruption per sample (0 = disabled)")
    p.add_argument("--sensor_corrupt_sigma_max", type=float, default=0.25,
                   help="Max Gaussian noise sigma for corruption augmentation")
    p.add_argument("--sensor_corrupt_sigma_min", type=float, default=0.02,
                   help="Min Gaussian noise sigma for corruption augmentation")
    p.add_argument("--sensor_corrupt_noise_only", action="store_true",
                   help="Only apply Gaussian noise (Mode 1), skip temporal missing")
    p.add_argument("--sensor_corrupt_mode_weights", type=float, nargs=4,
                   default=None, metavar=("M1", "M2", "M3", "M4"),
                   help="Relative weights for corruption Modes 1-4 (default: equal)")
    p.add_argument("--sensor_corrupt_rgbd_s_corr_max", type=float, default=6.0,
                   help="Max RGBD s_corr for consecutive missing augmentation")
    p.add_argument("--sensor_corrupt_imu_s_corr_max", type=float, default=45.0,
                   help="Max IMU s_corr for consecutive missing augmentation")

    # Centaur-style Cleaning DAE (frozen, applied before model)
    p.add_argument("--cleaning_dae_ckpt", type=str, default=None,
                   help="Path to trained CleaningDAE checkpoint (.pt). "
                        "Data is passed through the frozen DAE before "
                        "the HAR model during both training and evaluation.")

    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience (0 = disabled)")
    p.add_argument("--early_stop_metric", type=str, default="acc",
                   choices=["acc", "f1"],
                   help="Metric for early stopping and best-model selection")

    # Infrastructure
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--save_name", type=str, default="best.pt")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile", action="store_true",
                   help="Use torch.compile for faster training")
    p.add_argument("--swa", action="store_true",
                   help="Enable Stochastic Weight Averaging (last swa_epochs)")
    p.add_argument("--swa_start", type=float, default=0.75,
                   help="Start SWA at this fraction of total epochs")
    p.add_argument("--swa_lr", type=float, default=None,
                   help="SWA learning rate (default: 0.05 * lr)")
    p.add_argument("--vis_dir", type=str, default="",
                   help="Directory for training visualisations (empty = off)")
    p.add_argument("--tb_dir", type=str, default="",
                   help="TensorBoard log directory (empty = off)")

    return p.parse_args()


# ================================================================
#  Main
# ================================================================

def main():
    args = parse_args()

    # -- Seed --
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # -- Resolve pipeline config --
    cfg = {}
    if args.pipeline:
        cfg = PIPELINES[args.pipeline].copy()

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

    # -- Merge kwargs: defaults < user JSON --
    model_kwargs = cfg.get("default_model_kwargs", {}).copy()
    model_kwargs.update(parse_json_kwargs(args.model_kwargs))

    user_ds_kw = parse_json_kwargs(args.dataset_kwargs)

    # -- Corruption / modality dropout --
    modality_dropout_kwargs = _build_modality_dropout_kwargs(args)
    if modality_dropout_kwargs:
        print(f"Corruption: mode={modality_dropout_kwargs['mode']}, "
              f"p_full={modality_dropout_kwargs['p_full']}, "
              f"p_consec={modality_dropout_kwargs['p_consecutive']}")

    # -- Sensor corruption augmentation --
    sensor_corruption_kwargs = None
    if args.sensor_corrupt_p > 0:
        sensor_corruption_kwargs = {
            "p_apply": args.sensor_corrupt_p,
            "sigma_range": (args.sensor_corrupt_sigma_min,
                            args.sensor_corrupt_sigma_max),
            "rgbd_s_corr_range": (1.0, args.sensor_corrupt_rgbd_s_corr_max),
            "imu_s_corr_range": (10.0, args.sensor_corrupt_imu_s_corr_max),
        }
        if args.sensor_corrupt_noise_only:
            sensor_corruption_kwargs["mode_weights"] = (1, 0, 0, 0)
        elif args.sensor_corrupt_mode_weights is not None:
            sensor_corruption_kwargs["mode_weights"] = tuple(args.sensor_corrupt_mode_weights)
        noise_tag = " [noise-only]" if args.sensor_corrupt_noise_only else ""
        print(f"SensorCorruption: p={args.sensor_corrupt_p}, "
              f"σ=[{args.sensor_corrupt_sigma_min}, {args.sensor_corrupt_sigma_max}]"
              f"{noise_tag}")

    # -- Datasets --
    train_ds, test_ds = create_datasets(
        DatasetClass, data_key, args.data_root,
        cfg.get("default_dataset_kwargs", {}),
        user_ds_kw, norm_keys,
        modality_dropout_kwargs=modality_dropout_kwargs,
        sensor_corruption_kwargs=sensor_corruption_kwargs,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Datasets: train={len(train_ds)}, test={len(test_ds)}")

    # -- Model --
    device = torch.device(args.device)
    model = ModelClass(**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_class_name}  params={n_params:,}")

    # -- torch.compile --
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("torch.compile enabled")

    # -- CUDA optimisations --
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # -- Load trainer module --
    get_criterion, do_train_epoch, do_evaluate = load_trainer(trainer_module_path)
    print(f"Trainer: {trainer_module_path}")

    # Pipeline config dict passed to trainer functions
    pipeline_cfg = {
        "input_mode": input_mode,
        "output_mode": output_mode,
    }

    # -- Load frozen Cleaning DAE (if provided) --
    if args.cleaning_dae_ckpt:
        from model.cleaning_dae import CleaningDAE
        dae_ckpt = torch.load(args.cleaning_dae_ckpt, map_location=device,
                              weights_only=False)
        cleaning_dae = CleaningDAE(
            enable_imu=dae_ckpt.get("enable_imu", True),
            enable_rgbd=dae_ckpt.get("enable_rgbd", True),
            imu_latent_dim=dae_ckpt.get("latent_dim", 64),
        ).to(device)
        cleaning_dae.load_state_dict(dae_ckpt["model_state"])
        cleaning_dae.eval()
        for p in cleaning_dae.parameters():
            p.requires_grad_(False)
        pipeline_cfg["cleaning_dae"] = cleaning_dae
        print(f"Cleaning DAE: {args.cleaning_dae_ckpt} "
              f"(imu={dae_ckpt.get('enable_imu')}, "
              f"rgbd={dae_ckpt.get('enable_rgbd')}, frozen)")

    # -- Optimizer --
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )

    # -- Scheduler --
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == "cosine_warmup":
        scheduler = WarmupCosineScheduler(
            optimizer, args.warmup_epochs, args.epochs,
        )
    elif args.scheduler == "step":
        scheduler = StepLR(
            optimizer, step_size=args.step_size, gamma=args.step_gamma,
        )

    # -- Loss (via trainer module) --
    trainer_kwargs = cfg.get("default_trainer_kwargs", {}).copy()
    trainer_kwargs.update(parse_json_kwargs(args.trainer_kwargs))
    criterion = get_criterion(args, **trainer_kwargs)

    # Merge trainer kwargs into pipeline_cfg so custom trainers can access them
    pipeline_cfg.update(trainer_kwargs)

    # -- Resume --
    start_epoch = 0
    best_metric = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0)
        best_metric = ckpt.get("best_metric", 0.0)
        print(f"Resumed from epoch {start_epoch}, best_metric={best_metric:.4f}")

    # -- Logging --
    os.makedirs(args.save_dir, exist_ok=True)
    logger = TrainingLogger() if args.vis_dir else None

    # -- TensorBoard --
    tb_writer = None
    if args.tb_dir:
        tb_log_dir = os.path.join(args.tb_dir, args.pipeline or "custom")
        tb_writer = SummaryWriter(log_dir=tb_log_dir)

    # -- Training loop --
    no_improve = 0

    # -- SWA setup --
    swa_model = None
    swa_scheduler = None
    swa_start_epoch = int(args.swa_start * args.epochs) if args.swa else args.epochs + 1
    if args.swa:
        swa_model = AveragedModel(model)
        swa_lr = args.swa_lr if args.swa_lr else args.lr * 0.05
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
        print(f"SWA enabled: start_epoch={swa_start_epoch}, swa_lr={swa_lr:.2e}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Set epoch info for modality dropout schedule
        base_model = model.module if hasattr(model, 'module') else model
        if hasattr(base_model, 'current_epoch'):
            base_model.current_epoch = epoch
            base_model.total_epochs = args.epochs

        tr_loss, tr_acc = do_train_epoch(
            model, train_loader, optimizer, criterion, device,
            pipeline_cfg, args,
        )
        metrics = do_evaluate(
            model, test_loader, criterion, device, pipeline_cfg, args,
        )
        val_loss = metrics["loss"]
        val_acc = metrics["acc"]
        val_f1 = metrics["f1"]

        # Scheduler step
        if epoch >= swa_start_epoch and swa_scheduler is not None:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        elif isinstance(scheduler, WarmupCosineScheduler):
            scheduler.step(epoch)
        elif scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # Best model
        current_metric = val_acc if args.early_stop_metric == "acc" else val_f1
        is_best = current_metric > best_metric
        if is_best:
            best_metric = current_metric
            no_improve = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_metric": best_metric,
                "pipeline": args.pipeline,
                "model_kwargs": model_kwargs,
                "args": vars(args),
            }, os.path.join(args.save_dir, args.save_name))
        else:
            no_improve += 1

        print(
            f"Epoch {epoch+1}/{args.epochs}  "
            f"TrLoss={tr_loss:.4f} TrAcc={tr_acc:.4f}  "
            f"ValLoss={val_loss:.4f} ValAcc={val_acc:.4f} F1={val_f1:.4f}  "
            f"LR={current_lr:.2e}  Best={best_metric:.4f}  "
            f"Time={elapsed:.1f}s"
        )

        if logger:
            logger.log(
                epoch + 1,
                train_loss=tr_loss, train_acc=tr_acc,
                val_loss=val_loss, val_acc=val_acc,
                lr=current_lr, val_f1=val_f1,
            )

        if tb_writer:
            tb_writer.add_scalar("Loss/train", tr_loss, epoch + 1)
            tb_writer.add_scalar("Loss/val", val_loss, epoch + 1)
            tb_writer.add_scalar("Accuracy/train", tr_acc, epoch + 1)
            tb_writer.add_scalar("Accuracy/val", val_acc, epoch + 1)
            tb_writer.add_scalar("F1/val", val_f1, epoch + 1)
            tb_writer.add_scalar("LR", current_lr, epoch + 1)
            if hasattr(base_model, 'get_md_prob'):
                tb_writer.add_scalar("MD_prob", base_model.get_md_prob(), epoch + 1)

        # Early stopping
        if args.patience > 0 and no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch+1} (patience={args.patience})")
            break

    # -- Final evaluation on best model --
    best_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

    final = do_evaluate(
        model, test_loader, criterion, device, pipeline_cfg, args,
    )
    print(f"\n{'='*50}")
    print(f"Best model  Acc={final['acc']:.4f}  F1={final['f1']:.4f}")
    print(f"{'='*50}")

    # -- SWA: update BN and evaluate --
    if swa_model is not None and args.swa:
        # Custom BN update: handles multi-input models (skel+imu, rgbd+imu, etc.)
        momenta = {}
        for module in swa_model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.reset_running_stats()
                momenta[module] = module.momentum
                module.momentum = None  # cumulative mean
        if momenta:
            swa_model.train()
            with torch.no_grad():
                for batch in train_loader:
                    batch = [b.to(device) for b in batch]
                    inputs = batch[:-1]
                    if pipeline_cfg.get("input_mode") == "list":
                        swa_model(inputs)
                    else:
                        swa_model(*inputs)
            for module, mom in momenta.items():
                module.momentum = mom
            swa_model.eval()
        swa_final = do_evaluate(
            swa_model, test_loader, criterion, device, pipeline_cfg, args,
        )
        print(f"SWA model   Acc={swa_final['acc']:.4f}  F1={swa_final['f1']:.4f}")
        if swa_final["acc"] >= final["acc"]:
            print("SWA model is better — saving as checkpoint")
            torch.save({
                "epoch": args.epochs,
                "model_state": swa_model.module.state_dict(),
                "best_metric": swa_final["acc"],
                "pipeline": args.pipeline,
                "model_kwargs": model_kwargs,
                "args": vars(args),
            }, os.path.join(args.save_dir, args.save_name))
            final = swa_final
        print(f"{'='*50}")

    # -- TensorBoard: log hparams --
    if tb_writer:
        hparam_dict = {
            "pipeline": args.pipeline or "custom",
            "lr": args.lr,
            "batch_size": args.batch_size,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "mixup_alpha": args.mixup_alpha,
        }
        tb_writer.add_hparams(
            hparam_dict,
            {"hparam/best_acc": final["acc"], "hparam/best_f1": final["f1"]},
        )
        tb_writer.close()

    # -- Visualisations --
    if logger and args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)
        logger.plot_curves(save_dir=args.vis_dir)
        logger.save_json(os.path.join(args.vis_dir, "training_history.json"))

    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)
        try:
            from datasets import ACTION_NAMES
            class_names = ACTION_NAMES
        except Exception:
            class_names = None
        plot_confusion_matrix(
            final["labels"], final["preds"],
            class_names=class_names,
            save_dir=args.vis_dir,
        )
        plot_per_class_accuracy(
            final["labels"], final["preds"],
            class_names=class_names,
            save_dir=args.vis_dir,
        )


if __name__ == "__main__":
    main()
