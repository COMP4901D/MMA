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
    return {
        "mode": args.corruption_mode,
        "p_full": args.corruption_p_full,
        "p_consecutive": args.corruption_p_consecutive,
        "consecutive_len_range": tuple(args.corruption_block_range),
        "modalities": args.corruption_modalities,
        "both_drop_prob": args.corruption_both_drop_prob,
    }


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

    p.add_argument("--output", type=str, default="preds.npz",
                   help="Path to save predictions (.npz)")
    p.add_argument("--compile", action="store_true",
                   help="Use torch.compile for faster inference")
    p.add_argument("--vis_dir", type=str, default="",
                   help="Directory for evaluation plots (empty = off)")

    return p.parse_args()


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

    # -- Merge kwargs: checkpoint saved < registry defaults < CLI --
    model_kwargs = ckpt.get("model_kwargs", {}).copy()
    model_kwargs.update(cfg.get("default_model_kwargs", {}))
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
    model.load_state_dict(state)
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
