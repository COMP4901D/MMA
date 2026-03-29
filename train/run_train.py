"""
Unified Training Runner
========================
Supports all MMA pipelines (utdmad, mmtsa, rgbd_imu) and baselines (mumu).
Models and datasets are dynamically imported based on pipeline config or
custom module/class paths.  Model-specific parameters are passed via JSON kwargs.

Usage:
  python train/run_train.py --pipeline utdmad --data_root datasets/UTD-MHAD/Inertial
  python train/run_train.py --pipeline mmtsa  --data_root datasets/UTD-MHAD \\
         --model_kwargs '{"fusion":"gated"}'
  python train/run_train.py --pipeline rgbd_imu --data_root datasets/UTD-MHAD \\
         --model_kwargs '{"fusion":"attention"}'

  # Custom model (not in registry):
  python train/run_train.py \\
         --model_module model.mma_utdmad --model_class MomentumMambaHAR \\
         --dataset_module datasets.utd_inertial --dataset_class UTDMADInertialDataset \\
         --data_root datasets/UTD-MHAD/Inertial --input_mode unpack --output_mode logits
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
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

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
        "default_model_kwargs": {
            "num_modalities": 1, "feature_dim": 128,
            "input_dim": 6, "num_activities": 27, "num_activity_groups": 5,
        },
        "default_dataset_kwargs": {"max_len": 256},
        "norm_keys": ["mean", "std"],
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


def create_datasets(DatasetClass, data_key, data_root,
                    default_ds_kwargs, user_ds_kwargs, norm_keys):
    """Create train and test datasets with proper normalisation propagation."""
    base_kw = {}
    base_kw.update(default_ds_kwargs)
    base_kw.update(user_ds_kwargs)

    # -- Train --
    train_kw = {data_key: data_root, "subjects": TRAIN_SUBJECTS}
    train_kw.update(base_kw)
    if _accepts_kwarg(DatasetClass, "augment"):
        train_kw.setdefault("augment", True)
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
#  Forward / Loss Adapters
# ================================================================

def model_forward(model, batch, device, input_mode):
    """Run forward pass.  Returns (output, labels)."""
    batch = [b.to(device) for b in batch]
    labels = batch[-1]
    inputs = batch[:-1]
    if input_mode == "list":
        output = model(inputs)
    else:  # "unpack"
        output = model(*inputs)
    return output, labels


def extract_logits(output, output_mode):
    """Extract classification logits from model output."""
    if output_mode == "tuple_first":
        return output[0]
    if output_mode == "mumu":
        return output[1]  # y_target
    return output  # "logits"


# ================================================================
#  Mixup
# ================================================================

def mixup_forward(model, batch, device, input_mode, output_mode, criterion, alpha):
    """Apply mixup to all inputs, return (loss, logits, labels)."""
    batch = [b.to(device) for b in batch]
    labels = batch[-1]
    inputs = batch[:-1]

    lam = np.random.beta(alpha, alpha)
    B = labels.size(0)
    perm = torch.randperm(B, device=device)

    mixed = [lam * x + (1 - lam) * x[perm] for x in inputs]
    if input_mode == "list":
        output = model(mixed)
    else:
        output = model(*mixed)

    logits = extract_logits(output, output_mode)
    loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[perm])
    return loss, logits, labels


# ================================================================
#  Training / Evaluation
# ================================================================

def train_one_epoch(model, loader, optimizer, criterion, device,
                    input_mode, output_mode,
                    clip_norm=1.0, mixup_alpha=0.0):
    """Generic training epoch."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        optimizer.zero_grad()

        if mixup_alpha > 0:
            loss, logits, labels = mixup_forward(
                model, batch, device, input_mode, output_mode,
                criterion, mixup_alpha,
            )
        else:
            output, labels = model_forward(model, batch, device, input_mode)
            logits = extract_logits(output, output_mode)
            loss = criterion(logits, labels)

        loss.backward()
        if clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.detach().argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, input_mode, output_mode):
    """Generic evaluation. Returns dict with loss, acc, prec, rec, f1, preds, labels."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        output, labels = model_forward(model, batch, device, input_mode)
        logits = extract_logits(output, output_mode)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    n = len(labels)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0,
    )
    return {
        "loss": total_loss / n,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "preds": preds,
        "labels": labels,
    }


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
                   choices=["logits", "tuple_first", "mumu"],
                   help="How to extract logits from model output")

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

    # -- Datasets --
    train_ds, test_ds = create_datasets(
        DatasetClass, data_key, args.data_root,
        cfg.get("default_dataset_kwargs", {}),
        user_ds_kw, norm_keys,
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

    # -- Loss --
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

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

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            input_mode, output_mode,
            clip_norm=args.clip_grad, mixup_alpha=args.mixup_alpha,
        )
        metrics = evaluate(
            model, test_loader, criterion, device, input_mode, output_mode,
        )
        val_loss = metrics["loss"]
        val_acc = metrics["acc"]
        val_f1 = metrics["f1"]

        # Scheduler step
        if isinstance(scheduler, WarmupCosineScheduler):
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

        # Early stopping
        if args.patience > 0 and no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch+1} (patience={args.patience})")
            break

    # -- Final evaluation on best model --
    best_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

    final = evaluate(
        model, test_loader, criterion, device, input_mode, output_mode,
    )
    print(f"\n{'='*50}")
    print(f"Best model  Acc={final['acc']:.4f}  F1={final['f1']:.4f}")
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
