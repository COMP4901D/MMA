"""
MMA: Momentum Mamba for UTD-MAD HAR (Pure IMU)

Entry point — model and dataset are imported from modular packages.

Usage:
    python mma_utdmad.py --data_dir datasets/UTD-MHAD/Inertial
    python mma_utdmad.py --data_dir datasets/UTD-MHAD/Inertial --momentum_mode complex
    python mma_utdmad.py --data_dir datasets/UTD-MHAD/Inertial --momentum_mode none
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
)

from model.mma_utdmad import MomentumMambaHAR
from datasets.utd_inertial import UTDMADInertialDataset
from vis.training import TrainingLogger
from vis.evaluation import (
    plot_confusion_matrix, plot_per_class_metrics, plot_per_class_accuracy,
)


# ================================================================
#  Training / Evaluation
# ================================================================

def train_one_epoch(model, loader, optimizer, criterion, device,
                    clip_norm=1.0):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        running_loss += criterion(logits, yb).item() * xb.size(0)
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(yb.cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return running_loss / len(loader.dataset), acc, prec, rec, f1


# ================================================================
#  Main
# ================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Data ---
    print("=== Loading UTD-MAD Inertial Data ===")
    train_ds = UTDMADInertialDataset(
        args.data_dir, subjects=[1, 3, 5, 7], max_len=args.max_len,
    )
    test_ds = UTDMADInertialDataset(
        args.data_dir, subjects=[2, 4, 6, 8], max_len=args.max_len,
        mean=train_ds.mean, std=train_ds.std,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # --- Model ---
    use_momentum = args.momentum_mode != "none"
    mode_str = args.momentum_mode if use_momentum else "(vanilla Mamba)"
    print(f"\n=== Building Model  [momentum: {mode_str}] ===")

    model = MomentumMambaHAR(
        in_channels=6,
        num_classes=27,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dropout=args.dropout,
        use_momentum=use_momentum,
        momentum_mode=args.momentum_mode if use_momentum else "real",
        alpha_init=args.alpha,
        beta_init=args.beta,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {n_params:,}  ({n_params/1e3:.1f} K)")

    # --- Optimiser ---
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # --- Training loop ---
    print(f"\n=== Training (max {args.epochs} epochs, patience {args.patience}) ===")
    best_acc, wait = 0.0, 0
    best_state = None
    logger = TrainingLogger()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            clip_norm=args.clip_norm,
        )
        te_loss, te_acc, prec, rec, f1 = evaluate(
            model, test_loader, criterion, device,
        )
        scheduler.step()
        elapsed = time.time() - t0

        mom_str = ""
        if use_momentum:
            with torch.no_grad():
                blk0 = model.backbone[0].ssm
                beta_val = torch.sigmoid(blk0.beta_logit).item()
                alpha_val = blk0.alpha.item()
                mom_str = f"  α={alpha_val:.3f} β={beta_val:.3f}"

        log_kw = dict(
            train_loss=tr_loss, train_acc=tr_acc,
            val_loss=te_loss, val_acc=te_acc,
            val_f1=f1, lr=optimizer.param_groups[0]["lr"],
        )
        if use_momentum:
            log_kw["alpha"] = alpha_val
            log_kw["beta"] = beta_val
        logger.log(epoch, **log_kw)

        print(
            f"  Epoch {epoch:3d}/{args.epochs}  "
            f"TrainLoss {tr_loss:.4f}  TrainAcc {tr_acc:.4f}  |  "
            f"TestLoss {te_loss:.4f}  TestAcc {te_acc:.4f}  "
            f"F1 {f1:.4f}{mom_str}  ({elapsed:.1f}s)"
        )

        if te_acc > best_acc:
            best_acc = te_acc
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    # --- Final evaluation ---
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    _, te_acc, prec, rec, f1 = evaluate(model, test_loader, criterion, device)

    print("\n" + "=" * 50)
    print("  BEST TEST RESULTS")
    print("=" * 50)
    print(f"  Accuracy:  {te_acc:.4f}  ({te_acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print("=" * 50)

    save_path = f"mma_{args.momentum_mode}_best.pt"
    torch.save(best_state, save_path)
    print(f"\n  Model saved to {save_path}")

    # --- Visualizations ---
    if args.vis_dir:
        print(f"\n  Generating visualizations → {args.vis_dir}")
        logger.save_json(os.path.join(args.vis_dir, "history.json"))
        logger.plot_curves(save_dir=args.vis_dir)
        logger.plot_loss(save_dir=args.vis_dir)
        logger.plot_accuracy(save_dir=args.vis_dir)
        logger.plot_lr(save_dir=args.vis_dir)
        if use_momentum:
            logger.plot_momentum_params(save_dir=args.vis_dir)

        # Re-run evaluate to get preds for plots
        _, _, _, _, _, = evaluate(model, test_loader, criterion, device)
        # Collect preds/labels
        all_p, all_l = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in test_loader:
                logits = model(xb.to(device))
                all_p.append(logits.argmax(1).cpu().numpy())
                all_l.append(yb.numpy())
        preds = np.concatenate(all_p)
        labels = np.concatenate(all_l)
        plot_confusion_matrix(labels, preds, save_dir=args.vis_dir)
        plot_per_class_metrics(labels, preds, save_dir=args.vis_dir)
        plot_per_class_accuracy(labels, preds, save_dir=args.vis_dir)
        print(f"  Done. Plots saved to {args.vis_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="MMA — Momentum Mamba for UTD-MAD HAR"
    )

    # Data
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory with UTD-MAD *_inertial.mat files")
    p.add_argument("--max_len", type=int, default=256)

    # Architecture
    p.add_argument("--d_model",  type=int,   default=128)
    p.add_argument("--n_layers", type=int,   default=2)
    p.add_argument("--d_state",  type=int,   default=64)
    p.add_argument("--d_conv",   type=int,   default=4)
    p.add_argument("--expand",   type=int,   default=2)
    p.add_argument("--dropout",  type=float, default=0.1)

    # Momentum
    p.add_argument("--momentum_mode", type=str, default="real",
                   choices=["none", "real", "complex"])
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--beta",  type=float, default=0.99)

    # Training
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_norm",    type=float, default=1.0)
    p.add_argument("--patience",     type=int,   default=15)
    p.add_argument("--num_workers",  type=int,   default=2)

    # Visualization
    p.add_argument("--vis_dir", type=str, default=None,
                   help="Directory for saving vis plots (None = skip)")

    args = p.parse_args()
    main(args)