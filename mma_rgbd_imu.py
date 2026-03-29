"""
MMA-Multimodal: Momentum Mamba for HAR  (RGB-D + IMU)

Entry point — model and dataset are imported from modular packages.

Usage:
    python mma_rgbd_imu.py --data_dir datasets/UTD-MHAD --fusion attention
    python mma_rgbd_imu.py --data_dir datasets/UTD-MHAD --fusion gated
    python mma_rgbd_imu.py --data_dir datasets/UTD-MHAD --fusion concat
    python mma_rgbd_imu.py --data_dir datasets/UTD-MHAD --momentum_mode none
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
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

from model.mma_rgbd_imu import MultimodalMMA
from datasets.utd_rgbd_imu import UTDMADRGBDIMUDataset
from datasets import ACTION_NAMES
from vis.training import TrainingLogger
from vis.evaluation import (
    plot_confusion_matrix, plot_per_class_metrics, plot_per_class_accuracy,
)


# ================================================================
#  Training / Evaluation
# ================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, clip_norm=1.0):
    model.train()
    tot_loss, correct, total = 0.0, 0, 0
    for rgbd, imu, yb in loader:
        rgbd, imu, yb = rgbd.to(device), imu.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(rgbd, imu)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        tot_loss += loss.item() * rgbd.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += rgbd.size(0)
    return tot_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    tot_loss = 0.0
    all_pred, all_true = [], []
    for rgbd, imu, yb in loader:
        rgbd, imu, yb = rgbd.to(device), imu.to(device), yb.to(device)
        logits = model(rgbd, imu)
        tot_loss += criterion(logits, yb).item() * rgbd.size(0)
        all_pred.append(logits.argmax(1).cpu())
        all_true.append(yb.cpu())

    preds = torch.cat(all_pred).numpy()
    labels = torch.cat(all_true).numpy()
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0,
    )
    return tot_loss / len(loader.dataset), acc, prec, rec, f1, preds, labels


# ================================================================
#  Main
# ================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Data ---
    print("=" * 64)
    print("  Loading UTD-MHAD  (RGB-D + IMU)")
    print("=" * 64)

    use_gaf_imu = args.imu_encoder == "convnextv2"

    train_ds = UTDMADRGBDIMUDataset(
        args.data_dir,
        subjects=[1, 3, 5, 7],
        n_frames=args.n_frames,
        frame_size=args.frame_size,
        max_imu_len=args.max_imu_len,
        imu_as_gaf=use_gaf_imu,
        gaf_size=args.gaf_size,
    )
    test_ds = UTDMADRGBDIMUDataset(
        args.data_dir,
        subjects=[2, 4, 6, 8],
        n_frames=args.n_frames,
        frame_size=args.frame_size,
        max_imu_len=args.max_imu_len,
        iner_mean=train_ds.iner_mean,
        iner_std=train_ds.iner_std,
        imu_as_gaf=use_gaf_imu,
        gaf_size=args.gaf_size,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # --- Model ---
    use_mom = args.momentum_mode != "none"
    print(
        f"\n  Model: MultimodalMMA (RGB-D + IMU)"
        f"  [momentum={args.momentum_mode}, fusion={args.fusion}]"
    )

    model = MultimodalMMA(
        num_classes=27,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dropout=args.dropout,
        use_momentum=use_mom,
        momentum_mode=args.momentum_mode if use_mom else "real",
        alpha_init=args.alpha,
        beta_init=args.beta,
        fusion=args.fusion,
        n_heads=args.n_heads,
        encoder=args.encoder,
        convnext_model=args.convnext_model,
        freeze_stages=args.freeze_stages,
        imu_encoder=args.imu_encoder,
        gaf_size=args.gaf_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}  ({n_params / 1e6:.2f} M)")

    rgbd_mem = len(train_ds) * args.n_frames * 4 * args.frame_size ** 2 * 4
    rgbd_mem += len(test_ds) * args.n_frames * 4 * args.frame_size ** 2 * 4
    print(f"  RGBD in RAM: ~{rgbd_mem / 1e9:.2f} GB\n")

    # --- Optimiser ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --- Train ---
    print(
        f"  Training: max {args.epochs} epochs, "
        f"patience {args.patience}, bs {args.batch_size}"
    )
    print("-" * 94)

    best_acc, best_f1, wait, best_state = 0.0, 0.0, 0, None
    logger = TrainingLogger()

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            clip_norm=args.clip_norm,
        )
        te_loss, te_acc, prec, rec, f1, _, _ = evaluate(
            model, test_loader, criterion, device,
        )
        scheduler.step()
        dt = time.time() - t0

        mom = ""
        if use_mom:
            with torch.no_grad():
                ssm0 = model.rgbd_enc.blocks[0].ssm
                mom = (
                    f"  α={ssm0.alpha.item():.3f}"
                    f" β={torch.sigmoid(ssm0.beta_logit).item():.3f}"
                )

        log_kw = dict(
            train_loss=tr_loss, train_acc=tr_acc,
            val_loss=te_loss, val_acc=te_acc,
            val_f1=f1, lr=optimizer.param_groups[0]["lr"],
        )
        if use_mom:
            with torch.no_grad():
                log_kw["alpha"] = model.rgbd_enc.blocks[0].ssm.alpha.item()
                log_kw["beta"] = torch.sigmoid(
                    model.rgbd_enc.blocks[0].ssm.beta_logit
                ).item()
        logger.log(ep, **log_kw)

        tag = " *" if te_acc > best_acc else ""
        print(
            f"  Ep {ep:3d}/{args.epochs}"
            f"  TrL {tr_loss:.4f}  TrA {tr_acc:.4f}"
            f"  | TeL {te_loss:.4f}  TeA {te_acc:.4f}  F1 {f1:.4f}"
            f"{mom}  ({dt:.1f}s){tag}"
        )

        if te_acc > best_acc:
            best_acc, best_f1, wait = te_acc, f1, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print(f"\n  Early stopping at epoch {ep}.")
                break

    # --- Final ---
    if best_state:
        model.load_state_dict(best_state)
    model.to(device)
    _, te_acc, prec, rec, f1, preds, labels = evaluate(
        model, test_loader, criterion, device,
    )

    print("\n" + "=" * 64)
    print("  FINAL  MULTIMODAL  RESULTS  (RGB-D + IMU)")
    print("=" * 64)
    print(f"  Fusion:      {args.fusion}")
    print(f"  Momentum:    {args.momentum_mode}")
    print(f"  Accuracy:    {te_acc:.4f}  ({te_acc * 100:.2f}%)")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  F1-score:    {f1:.4f}")
    print("=" * 64)

    names = [f"{i:02d}_{n}" for i, n in enumerate(ACTION_NAMES)]
    print("\n  Per-class Report:\n")
    print(classification_report(labels, preds, target_names=names, zero_division=0))

    save_tag = f"mma_rgbd_imu_{args.fusion}_{args.momentum_mode}"
    torch.save(best_state, f"{save_tag}_best.pt")
    print(f"  Saved → {save_tag}_best.pt")

    # --- Visualizations ---
    if args.vis_dir:
        print(f"\n  Generating visualizations → {args.vis_dir}")
        logger.save_json(os.path.join(args.vis_dir, "history.json"))
        logger.plot_curves(save_dir=args.vis_dir)
        logger.plot_loss(save_dir=args.vis_dir)
        logger.plot_accuracy(save_dir=args.vis_dir)
        logger.plot_lr(save_dir=args.vis_dir)
        if use_mom:
            logger.plot_momentum_params(save_dir=args.vis_dir)
        plot_confusion_matrix(
            labels, preds,
            class_names=ACTION_NAMES, save_dir=args.vis_dir,
        )
        plot_per_class_metrics(
            labels, preds,
            class_names=ACTION_NAMES, save_dir=args.vis_dir,
        )
        plot_per_class_accuracy(
            labels, preds,
            class_names=ACTION_NAMES, save_dir=args.vis_dir,
        )
        print(f"  Done. Plots saved to {args.vis_dir}")


# ================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MMA — RGB-D + IMU Multimodal HAR")

    # Data
    p.add_argument("--data_dir", type=str, required=True,
                   help="UTD-MHAD root (contains Inertial/, Depth/, RGB-part*/)")
    p.add_argument("--n_frames", type=int, default=16)
    p.add_argument("--frame_size", type=int, default=112)
    p.add_argument("--max_imu_len", type=int, default=128)

    # Architecture
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--d_state", type=int, default=64)
    p.add_argument("--d_conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--n_heads", type=int, default=4)

    # Encoder backbone
    p.add_argument("--encoder", type=str, default="default",
                   choices=["default", "convnextv2"])
    p.add_argument("--imu_encoder", type=str, default="conv1d",
                   choices=["conv1d", "convnextv2"])
    p.add_argument("--convnext_model", type=str, default="convnextv2_atto")
    p.add_argument("--freeze_stages", type=int, default=3)
    p.add_argument("--gaf_size", type=int, default=64)

    # Momentum
    p.add_argument("--momentum_mode", type=str, default="real",
                   choices=["none", "real", "complex"])
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--beta", type=float, default=0.99)

    # Fusion
    p.add_argument("--fusion", type=str, default="attention",
                   choices=["concat", "attention", "gated"])

    # Training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=15)

    # Visualization
    p.add_argument("--vis_dir", type=str, default=None,
                   help="Directory for saving vis plots (None = skip)")

    args = p.parse_args()
    main(args)