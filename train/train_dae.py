"""
Train Centaur-style Convolutional Denoising Autoencoder (DAE).

The DAE is trained INDEPENDENTLY from the HAR model.  It learns to map
corrupted sensor data back to the original clean data using MSE loss.

Usage:
  # Train both IMU + RGBD DAEs (default)
  python train/train_dae.py --data_root datasets/UTD-MHAD --epochs 200

  # Train IMU-only DAE
  python train/train_dae.py --data_root datasets/UTD-MHAD --modality imu --epochs 200

  # Train RGBD-only DAE
  python train/train_dae.py --data_root datasets/UTD-MHAD --modality rgbd --epochs 200
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.utd_rgbd_imu import UTDMADRGBDIMUDataset
from datasets.transforms import SensorCorruptionAugment
from model.cleaning_dae import CleaningDAE

TRAIN_SUBJECTS = [1, 3, 5, 7]
VAL_SUBJECTS = [2, 4, 6, 8]


def corrupt_batch(rgbd_clean, imu_clean, corruptor, rng):
    """Apply stochastic corruption to a batch of clean data.

    Returns numpy arrays (corrupted copies).
    """
    B = rgbd_clean.shape[0]
    rgbd_corr = rgbd_clean.copy()
    imu_corr = imu_clean.copy()
    for i in range(B):
        rgbd_corr[i], imu_corr[i] = corruptor(rgbd_corr[i], imu_corr[i], rng=rng)
    return rgbd_corr, imu_corr


class DAEDataset(torch.utils.data.Dataset):
    """Wraps UTDMADRGBDIMUDataset to return clean data as numpy arrays.

    We need raw numpy arrays so we can apply corruption ourselves
    (the corruption must produce a (corrupted, clean) pair).
    """

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        rgbd = self.base.rgbd_data[idx].copy()  # (T, 4, H, W) float32 [0,1]
        imu_raw = self.base.imu_data[idx]
        imu = ((imu_raw - self.base.iner_mean) / self.base.iner_std).astype(np.float32)
        imu = UTDMADRGBDIMUDataset._resample_1d(imu, self.base.max_imu_len)
        return rgbd, imu


def collate_dae(batch):
    """Collate clean numpy arrays into a single numpy batch."""
    rgbds, imus = zip(*batch)
    return np.stack(rgbds, axis=0), np.stack(imus, axis=0)


def train_dae(args):
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Load clean training data ----
    print("Loading training data...")
    train_ds_base = UTDMADRGBDIMUDataset(
        data_dir=args.data_root, subjects=TRAIN_SUBJECTS,
        n_frames=16, frame_size=112, max_imu_len=128,
    )
    val_ds_base = UTDMADRGBDIMUDataset(
        data_dir=args.data_root, subjects=VAL_SUBJECTS,
        n_frames=16, frame_size=112, max_imu_len=128,
        iner_mean=train_ds_base.iner_mean,
        iner_std=train_ds_base.iner_std,
    )

    train_ds = DAEDataset(train_ds_base)
    val_ds = DAEDataset(val_ds_base)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_dae,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_dae,
    )

    # ---- Corruption process (Centaur's 4 modes) ----
    corruptor = SensorCorruptionAugment(
        p_apply=1.0,  # Always corrupt (DAE training needs corrupted input)
        sigma_range=(0.02, 0.30),
        rgbd_s_norm=8.0,
        rgbd_s_corr_range=(2.0, 6.0),
        imu_s_norm=60.0,
        imu_s_corr_range=(15.0, 45.0),
        mode_weights=(1, 1, 1, 1),  # Uniform across all 4 modes
    )
    rng = np.random.default_rng(42)

    # ---- Build model ----
    enable_imu = args.modality in ("both", "imu")
    enable_rgbd = args.modality in ("both", "rgbd")

    model = CleaningDAE(
        imu_channels=6, imu_seq_len=128, imu_latent_dim=args.latent_dim,
        rgbd_channels=4,
        enable_imu=enable_imu,
        enable_rgbd=enable_rgbd,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"CleaningDAE ({args.modality}): {n_params:,} params")

    # ---- Optimizer (Centaur uses RMSprop lr=1e-4 momentum=0.1) ----
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=args.lr, momentum=0.1,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0

    print(f"\nTraining for {args.epochs} epochs  "
          f"(lr={args.lr}, batch={args.batch_size}, latent={args.latent_dim})")
    print(f"Modality: {args.modality}")
    print(f"Save dir: {args.save_dir}\n")

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        train_losses = []
        t0 = time.time()

        for rgbd_clean_np, imu_clean_np in train_loader:
            # Apply corruption to get (corrupted, clean) pairs
            rgbd_corr_np, imu_corr_np = corrupt_batch(
                rgbd_clean_np, imu_clean_np, corruptor, rng,
            )

            rgbd_clean = torch.as_tensor(rgbd_clean_np, device=device)
            imu_clean = torch.as_tensor(imu_clean_np, device=device)
            rgbd_corr = torch.as_tensor(rgbd_corr_np, device=device)
            imu_corr = torch.as_tensor(imu_corr_np, device=device)

            # Forward: corrupted -> DAE -> reconstructed
            rgbd_recon, imu_recon = model(rgbd_corr, imu_corr)

            # MSE loss vs clean targets
            loss = 0.0
            if enable_rgbd:
                loss = loss + criterion(rgbd_recon, rgbd_clean)
            if enable_imu:
                loss = loss + criterion(imu_recon, imu_clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        avg_train = np.mean(train_losses)

        # ---- Validate ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for rgbd_clean_np, imu_clean_np in val_loader:
                rgbd_corr_np, imu_corr_np = corrupt_batch(
                    rgbd_clean_np, imu_clean_np, corruptor, rng,
                )

                rgbd_clean = torch.as_tensor(rgbd_clean_np, device=device)
                imu_clean = torch.as_tensor(imu_clean_np, device=device)
                rgbd_corr = torch.as_tensor(rgbd_corr_np, device=device)
                imu_corr = torch.as_tensor(imu_corr_np, device=device)

                rgbd_recon, imu_recon = model(rgbd_corr, imu_corr)

                loss = 0.0
                if enable_rgbd:
                    loss = loss + criterion(rgbd_recon, rgbd_clean)
                if enable_imu:
                    loss = loss + criterion(imu_recon, imu_clean)
                val_losses.append(loss.item())

        avg_val = np.mean(val_losses)
        elapsed = time.time() - t0

        improved = avg_val < best_val_loss
        if improved:
            best_val_loss = avg_val
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": avg_val,
                "modality": args.modality,
                "latent_dim": args.latent_dim,
                "enable_imu": enable_imu,
                "enable_rgbd": enable_rgbd,
            }, os.path.join(args.save_dir, f"dae_{args.modality}_best.pt"))

        if epoch % args.log_every == 0 or improved:
            mark = " *" if improved else ""
            print(f"  Epoch {epoch:>3d}/{args.epochs}  "
                  f"train={avg_train:.6f}  val={avg_val:.6f}  "
                  f"best={best_val_loss:.6f}@{best_epoch}  "
                  f"[{elapsed:.1f}s]{mark}")

    # Save final
    torch.save({
        "epoch": args.epochs,
        "model_state": model.state_dict(),
        "val_loss": avg_val,
        "modality": args.modality,
        "latent_dim": args.latent_dim,
        "enable_imu": enable_imu,
        "enable_rgbd": enable_rgbd,
    }, os.path.join(args.save_dir, f"dae_{args.modality}_last.pt"))

    print(f"\nDone. Best val loss: {best_val_loss:.6f} @ epoch {best_epoch}")
    print(f"  Best:  {args.save_dir}/dae_{args.modality}_best.pt")
    print(f"  Last:  {args.save_dir}/dae_{args.modality}_last.pt")


def parse_args():
    p = argparse.ArgumentParser(description="Train Centaur-style Cleaning DAE")
    p.add_argument("--data_root", type=str, required=True,
                   help="Path to UTD-MHAD dataset root")
    p.add_argument("--modality", type=str, default="both",
                   choices=["both", "imu", "rgbd"],
                   help="Which modality DAE(s) to train")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate (Centaur default: 1e-4)")
    p.add_argument("--latent_dim", type=int, default=64,
                   help="Latent dimension for IMU DAE bottleneck")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=str, default="checkpoints/dae")
    p.add_argument("--log_every", type=int, default=10,
                   help="Print every N epochs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dae(args)
