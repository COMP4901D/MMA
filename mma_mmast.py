#!/usr/bin/env python3
"""
MMA-MMTSA  –  Momentum Multimodal Attention + GAF + Temporal Segments

Entry point — model and dataset are imported from modular packages.

Usage:
    python mma_mmast.py --data_root ./datasets/UTD-MHAD
    python mma_mmast.py --use_segments 0
    python mma_mmast.py --use_momentum 0
    python mma_mmast.py --fusion gated
"""

import random, time, argparse, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, accuracy_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix)

from model.mma_mmtsa import MMA_MMTSA
from datasets.utd_depth_imu import UTD_MHAD_Dataset
from datasets import ACTION_NAMES
from vis.training import TrainingLogger
from vis.evaluation import (
    plot_confusion_matrix, plot_per_class_metrics,
    plot_per_class_accuracy, plot_prediction_confidence,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args():
    p = argparse.ArgumentParser(description="MMA-MMTSA training on UTD-MHAD")

    # paths
    p.add_argument("--data_root",    type=str,   default="./UTD-MHAD")
    p.add_argument("--save_path",    type=str,   default="mma_mmtsa_best.pt")

    # data / GAF
    p.add_argument("--gaf_size",     type=int,   default=64)
    p.add_argument("--frame_size",   type=int,   default=64)
    p.add_argument("--depth_channels", type=int, default=3,
                   help="1=single frame, 3=mean+motion+max")

    # model
    p.add_argument("--feature_dim",  type=int,   default=256)
    p.add_argument("--momentum",     type=float, default=0.990)
    p.add_argument("--dropout",      type=float, default=0.40)
    p.add_argument("--num_classes",  type=int,   default=27)
    p.add_argument("--fusion",       type=str,   default="attention",
                   choices=["attention", "gated", "concat"])
    p.add_argument("--label_smooth", type=float, default=0.1)
    p.add_argument("--mixup_alpha",  type=float, default=0.2)

    # encoder backbone
    p.add_argument("--encoder",      type=str,   default="lightcnn",
                   choices=["lightcnn", "convnextv2"])
    p.add_argument("--convnext_model", type=str, default="convnextv2_atto")
    p.add_argument("--freeze_stages", type=int,  default=3)

    # ablation flags  (1 = on, 0 = off)
    p.add_argument("--use_gaf",      type=int,   default=1)
    p.add_argument("--use_segments", type=int,   default=1)
    p.add_argument("--use_momentum", type=int,   default=1)

    # training
    p.add_argument("--epochs",       type=int,   default=120)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--patience",     type=int,   default=25)
    p.add_argument("--warmup_epochs",type=int,   default=5)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--num_segments", type=int,   default=3)

    # Visualization
    p.add_argument("--vis_dir", type=str, default=None,
                   help="Directory for saving vis plots (None = skip)")

    return p.parse_args()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Training / Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def mixup_data(x1, x2, y, alpha=0.4):
    if alpha <= 0:
        return x1, x2, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    B = x1.size(0)
    idx = torch.randperm(B, device=x1.device)
    x1_mix = lam * x1 + (1 - lam) * x1[idx]
    x2_mix = lam * x2 + (1 - lam) * x2[idx]
    return x1_mix, x2_mix, y, y[idx], lam


def train_epoch(model, loader, opt, crit, device, mixup_alpha=0.4):
    model.train()
    tot_loss, correct, total = 0.0, 0, 0
    for depth, imu, labels in loader:
        depth, imu, labels = depth.to(device), imu.to(device), labels.to(device)

        depth_m, imu_m, y_a, y_b, lam = mixup_data(depth, imu, labels, mixup_alpha)

        opt.zero_grad()
        logits, _ = model(depth_m, imu_m)
        loss = lam * crit(logits, y_a) + (1 - lam) * crit(logits, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        tot_loss += loss.item() * labels.size(0)
        correct  += (logits.argmax(1) == labels).sum().item()
        total    += labels.size(0)
    return tot_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval()
    tot_loss, preds, gts = 0.0, [], []
    for depth, imu, labels in loader:
        depth, imu, labels = depth.to(device), imu.to(device), labels.to(device)
        logits, _ = model(depth, imu)
        tot_loss += crit(logits, labels).item() * labels.size(0)
        preds.extend(logits.argmax(1).cpu().numpy())
        gts.extend(labels.cpu().numpy())

    preds, gts = np.array(preds), np.array(gts)
    n = len(gts)
    return dict(
        loss = tot_loss / n,
        acc  = accuracy_score(gts, preds),
        prec = precision_score(gts, preds, average="macro", zero_division=0),
        rec  = recall_score(gts, preds, average="macro", zero_division=0),
        f1   = f1_score(gts, preds, average="macro", zero_division=0),
        preds=preds, gts=gts,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_seg = bool(args.use_segments)
    use_gaf = bool(args.use_gaf)
    use_mom = bool(args.use_momentum)

    tag = (f"fusion={args.fusion}  segments={use_seg}  "
           f"gaf={use_gaf}  momentum={use_mom}")

    print("=" * 70)
    print(f"  MMA-MMTSA  Training")
    print(f"  {tag}")
    print(f"  Device: {device}")
    print("=" * 70)

    # --- Data ---
    print("\n── Training set ──")
    train_ds = UTD_MHAD_Dataset(
        args.data_root, subjects=[1, 3, 5, 7],
        num_segments=args.num_segments, gaf_size=args.gaf_size,
        frame_size=args.frame_size, use_gaf=use_gaf,
        use_segments=use_seg, augment=True,
        depth_channels=args.depth_channels,
    )
    print("── Test set ──")
    test_ds = UTD_MHAD_Dataset(
        args.data_root, subjects=[2, 4, 6, 8],
        num_segments=args.num_segments, gaf_size=args.gaf_size,
        frame_size=args.frame_size, use_gaf=use_gaf,
        use_segments=use_seg, augment=False,
        depth_channels=args.depth_channels,
    )

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=False)
    test_loader  = DataLoader(test_ds,  args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    # --- Model ---
    print("\n── Model ──")
    model = MMA_MMTSA(
        num_classes    = args.num_classes,
        feat_dim       = args.feature_dim,
        n_seg          = args.num_segments,
        beta           = args.momentum,
        dropout        = args.dropout,
        fusion         = args.fusion,
        use_segments   = use_seg,
        use_momentum   = use_mom,
        imu_channels   = 6,
        depth_channels = args.depth_channels,
        encoder        = args.encoder,
        convnext_model = args.convnext_model,
        freeze_stages  = args.freeze_stages,
    ).to(device)

    # --- Optimiser & Scheduler (with Warmup) ---
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=args.warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs]
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)

    # --- Training loop ---
    print(f"\n{'Ep':>4}  {'TrLoss':>7} {'TrAcc':>6}  "
          f"{'TeLoss':>7} {'TeAcc':>6} {'TeF1':>6}  "
          f"{'LR':>9}  {'Time':>5}")
    print("-" * 70)

    best_f1, best_ep, wait = 0.0, 0, 0
    logger = TrainingLogger()

    for ep in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer,
                                      criterion, device,
                                      mixup_alpha=args.mixup_alpha)
        te = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        mark = ""
        if te["f1"] > best_f1:
            best_f1, best_ep, wait = te["f1"], ep, 0
            torch.save(dict(
                epoch=ep,
                state_dict=model.state_dict(),
                best_f1=best_f1,
                args=vars(args),
            ), args.save_path)
            mark = " ★"
        else:
            wait += 1

        logger.log(ep,
            train_loss=tr_loss, train_acc=tr_acc,
            val_loss=te["loss"], val_acc=te["acc"], val_f1=te["f1"],
            lr=lr,
        )

        print(f"  {ep:3d}  {tr_loss:7.4f} {tr_acc:6.4f}  "
              f"{te['loss']:7.4f} {te['acc']:6.4f} {te['f1']:6.4f}  "
              f"{lr:9.2e}  {dt:4.1f}s{mark}")

        if wait >= args.patience:
            print(f"\n  ⏹ Early stopping (patience={args.patience})")
            break

    # --- Final evaluation ---
    print("\n" + "=" * 70)
    print(f"  FINAL  MMA-MMTSA  RESULTS   (best epoch {best_ep})")
    print("=" * 70)

    ckpt = torch.load(args.save_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])

    te = evaluate(model, test_loader, criterion, device)

    print(f"  Config:      {tag}")
    print(f"  Accuracy:    {te['acc']:.4f}  ({te['acc']*100:.2f}%)")
    print(f"  Precision:   {te['prec']:.4f}")
    print(f"  Recall:      {te['rec']:.4f}")
    print(f"  F1-score:    {te['f1']:.4f}")
    print("=" * 70)

    present = sorted(set(te["gts"]))
    names   = [f"{c:02d}_{ACTION_NAMES[c]}" for c in present]
    report  = classification_report(
        te["gts"], te["preds"],
        labels=present, target_names=names, digits=2, zero_division=0,
    )
    print("\n  Per-class Report:\n")
    print(report)

    cm = confusion_matrix(te["gts"], te["preds"], labels=present)
    cm_path = args.save_path.replace(".pt", "_cm.npy")
    np.save(cm_path, cm)
    print(f"  Confusion matrix saved → {cm_path}")
    print(f"  Model saved → {args.save_path}")
    print("=" * 70)

    # --- Visualizations ---
    if args.vis_dir:
        print(f"\n  Generating visualizations → {args.vis_dir}")
        logger.save_json(os.path.join(args.vis_dir, "history.json"))
        logger.plot_curves(save_dir=args.vis_dir)
        logger.plot_loss(save_dir=args.vis_dir)
        logger.plot_accuracy(save_dir=args.vis_dir)
        logger.plot_lr(save_dir=args.vis_dir)
        plot_confusion_matrix(
            te["gts"], te["preds"],
            class_names=ACTION_NAMES, save_dir=args.vis_dir,
        )
        plot_per_class_metrics(
            te["gts"], te["preds"],
            class_names=ACTION_NAMES, save_dir=args.vis_dir,
        )
        plot_per_class_accuracy(
            te["gts"], te["preds"],
            class_names=ACTION_NAMES, save_dir=args.vis_dir,
        )
        print(f"  Done. Plots saved to {args.vis_dir}")


if __name__ == "__main__":
    main()