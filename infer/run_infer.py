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
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
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


def create_test_dataset(DatasetClass, data_key, data_root,
                        default_ds_kwargs, user_ds_kwargs, norm_keys):
    """Create train (for norm stats) and test datasets."""
    base_kw = {}
    base_kw.update(default_ds_kwargs)
    base_kw.update(user_ds_kwargs)

    # Build train set first if we need normalisation stats
    test_kw = {data_key: data_root, "subjects": TEST_SUBJECTS}
    test_kw.update(base_kw)
    if _accepts_kwarg(DatasetClass, "augment"):
        test_kw["augment"] = False

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
#  Forward Adapters
# ================================================================

def model_forward(model, batch, device, input_mode):
    """Run forward pass. Returns (output, labels)."""
    batch = [b.to(device) for b in batch]
    labels = batch[-1]
    inputs = batch[:-1]
    if input_mode == "list":
        output = model(inputs)
    else:
        output = model(*inputs)
    return output, labels


def extract_logits(output, output_mode):
    if output_mode == "tuple_first":
        return output[0]
    if output_mode == "mumu":
        return output[1]
    return output


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
    p.add_argument("--checkpoint", type=str, required=True)

    p.add_argument("--input_mode", type=str, default=None,
                   choices=["unpack", "list"])
    p.add_argument("--output_mode", type=str, default=None,
                   choices=["logits", "tuple_first", "mumu"])

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=str, default="preds.npz",
                   help="Path to save predictions (.npz)")
    p.add_argument("--vis_dir", type=str, default="",
                   help="Directory for evaluation plots (empty = off)")

    return p.parse_args()


# ================================================================
#  Main
# ================================================================

def main():
    args = parse_args()
    device = torch.device(args.device)

    # -- Load checkpoint --
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
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

    if not (model_module and model_class_name):
        raise ValueError("Specify --pipeline or --model_module / --model_class")
    if not (dataset_module and dataset_class_name):
        raise ValueError("Specify --pipeline or --dataset_module / --dataset_class")

    # -- Import classes --
    ModelClass = import_class(model_module, model_class_name)
    DatasetClass = import_class(dataset_module, dataset_class_name)

    # -- Merge kwargs: checkpoint saved < registry defaults < CLI --
    model_kwargs = ckpt.get("model_kwargs", {}).copy()
    model_kwargs.update(cfg.get("default_model_kwargs", {}))
    model_kwargs.update(parse_json_kwargs(args.model_kwargs))

    user_ds_kw = parse_json_kwargs(args.dataset_kwargs)

    # -- Model --
    model = ModelClass(**model_kwargs).to(device)

    state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_class_name}  params={n_params:,}")
    print(f"Checkpoint: {args.checkpoint}")

    # -- Dataset --
    test_ds = create_test_dataset(
        DatasetClass, data_key, args.data_root,
        cfg.get("default_dataset_kwargs", {}),
        user_ds_kw, norm_keys,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Test samples: {len(test_ds)}")

    # -- Inference --
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_loader:
            output, labels = model_forward(model, batch, device, input_mode)
            logits = extract_logits(output, output_mode)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    n = len(labels)

    # -- Metrics --
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0,
    )
    print(f"\nTest Results ({n} samples):")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  Loss     : {total_loss / n:.4f}")

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
