"""
MMA_SkeletonIMU: Skeleton + IMU dual-branch Momentum Mamba with cross-modal fusion.

Encoder variants:
  - skel_enc="linear":   Flatten(60) → Linear → Mamba  (baseline)
  - skel_enc="spatial":  Joints(60) + Bones(57) + Velocity(60) → Linear → Mamba

  - imu_enc="conv1d":      Conv1D(k=3) → Mamba  (baseline)
  - imu_enc="multiscale":  ParallelConv(k=3,7,15) → Mamba

Fusion modes:
  - "attention"/"gated": IndependentPools → DimGatedFusion
  - "cross_mamba": Concat sequences → shared MambaBlock → pool
  - "concat": Concatenate pooled representations
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm
from .mamba import MomentumMambaBlock


# ================================================================
#  Pooling & Fusion modules
# ================================================================

class AttentionPool(nn.Module):
    """Learnable attention-weighted temporal pooling."""

    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, x):
        """x: (B, T, D) -> (B, D)"""
        w = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        return (w * x).sum(dim=1)


class DimGatedFusion(nn.Module):
    """Per-dimension gated fusion — learns which dimensions to take from each modality."""

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

    def forward(self, fv, fi):
        """fv, fi: (B, D) -> (B, D)"""
        g = self.gate(torch.cat([fv, fi], dim=-1))  # (B, D)
        return g * fv + (1 - g) * fi


class AuxHead(nn.Module):
    """Per-modality auxiliary classification head for multi-task regularization."""

    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.pool = AttentionPool(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x):
        """x: (B, T, D) -> logits (B, C)"""
        return self.head(self.pool(x))


# ================================================================
#  Positional Encoding
# ================================================================

class LearnablePositionalEncoding(nn.Module):
    """Sinusoidal base + learnable residual positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2 + d_model % 2])
        self.register_buffer("pe_base", pe.unsqueeze(0))  # (1, max_len, D)
        self.residual = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        """x: (B, T, D) -> (B, T, D) with positional info added."""
        T = x.size(1)
        return x + self.pe_base[:, :T] + self.residual[:, :T]


# ================================================================
#  Encoders
# ================================================================

class SkeletonEncoder(nn.Module):
    """Linear projection + temporal Mamba for skeleton joint sequences."""

    def __init__(
        self,
        in_dim: int = 60,
        d_model: int = 128,
        n_layers: int = 2,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
        use_momentum: bool = True,
        momentum_mode: str = "real",
        alpha_init: float = 0.6,
        beta_init: float = 0.99,
        center_joints: bool = False,
    ):
        super().__init__()
        self.center_joints = center_joints
        self.frontend = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            MomentumMambaBlock(
                d_model, d_state, d_conv, expand,
                use_momentum, momentum_mode,
                alpha_init, beta_init,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        """x: (B, T, 60) -> (B, T, d_model)"""
        if self.center_joints:
            B, T, _ = x.shape
            joints = x.view(B, T, 20, 3)
            x = (joints - joints[:, :, 0:1, :]).reshape(B, T, 60)
        x = self.frontend(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class IMUEncoder(nn.Module):
    """Conv1D projection + temporal Mamba for 6-axis IMU sequences."""

    def __init__(
        self,
        in_channels: int = 6,
        d_model: int = 128,
        n_layers: int = 2,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
        use_momentum: bool = True,
        momentum_mode: str = "real",
        alpha_init: float = 0.6,
        beta_init: float = 0.99,
    ):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            MomentumMambaBlock(
                d_model, d_state, d_conv, expand,
                use_momentum, momentum_mode,
                alpha_init, beta_init,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        """x: (B, T, 6) -> (B, T, d_model)"""
        x = self.frontend(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class SpatialSkeletonEncoder(nn.Module):
    """Skeleton encoder with bone features and temporal velocity.

    Augments raw joint positions (60-dim) with:
      - Bone vectors: differences between connected joint pairs (19 bones × 3 = 57-dim)
      - Temporal velocity: first-order temporal difference of joint positions (60-dim)
    Total input: 60 + 57 + 60 = 177 dims.
    """

    # UTD-MHAD 20-joint skeleton connectivity (0-indexed)
    BONES = [
        (0, 1), (1, 2), (2, 3),                        # spine
        (2, 4), (4, 5), (5, 6), (6, 7),                # left arm
        (2, 8), (8, 9), (9, 10), (10, 11),             # right arm
        (0, 12), (12, 13), (13, 14), (14, 15),         # left leg
        (0, 16), (16, 17), (17, 18), (18, 19),         # right leg
    ]

    def __init__(
        self,
        in_dim: int = 60,
        d_model: int = 128,
        n_layers: int = 2,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
        use_momentum: bool = True,
        momentum_mode: str = "real",
        alpha_init: float = 0.6,
        beta_init: float = 0.99,
    ):
        super().__init__()
        # 60 (joints) + 57 (bones) + 60 (velocity) = 177
        total_in = in_dim + len(self.BONES) * 3 + in_dim
        self.frontend = nn.Sequential(
            nn.Linear(total_in, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            MomentumMambaBlock(
                d_model, d_state, d_conv, expand,
                use_momentum, momentum_mode,
                alpha_init, beta_init,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        """x: (B, T, 60) -> (B, T, d_model)"""
        B, T, _ = x.shape
        # Bone features: vector from parent to child joint
        joints = x.view(B, T, 20, 3)
        bones = torch.cat(
            [joints[:, :, c] - joints[:, :, p] for p, c in self.BONES],
            dim=-1,
        )  # (B, T, 57)
        # Temporal velocity: first-order diff (pad first frame with zeros)
        vel = torch.zeros_like(x)
        vel[:, 1:] = x[:, 1:] - x[:, :-1]
        # Concatenate all features
        x = torch.cat([x, bones, vel], dim=-1)  # (B, T, 177)
        x = self.frontend(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class MultiScaleIMUEncoder(nn.Module):
    """IMU encoder with parallel multi-scale Conv1D frontend.

    Uses three parallel convolutions with kernel sizes 3, 7, 15 to capture
    short, medium, and long-range temporal patterns simultaneously.
    """

    def __init__(
        self,
        in_channels: int = 6,
        d_model: int = 128,
        n_layers: int = 2,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
        use_momentum: bool = True,
        momentum_mode: str = "real",
        alpha_init: float = 0.6,
        beta_init: float = 0.99,
    ):
        super().__init__()
        k_sizes = [3, 7, 15]
        d_per = d_model // len(k_sizes)
        self.convs = nn.ModuleList()
        for i, k in enumerate(k_sizes):
            d_out = d_per if i < len(k_sizes) - 1 else d_model - d_per * (len(k_sizes) - 1)
            self.convs.append(nn.Conv1d(in_channels, d_out, kernel_size=k, padding=k // 2))
        self.bn = nn.BatchNorm1d(d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            MomentumMambaBlock(
                d_model, d_state, d_conv, expand,
                use_momentum, momentum_mode,
                alpha_init, beta_init,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        """x: (B, T, 6) -> (B, T, d_model)"""
        x_t = x.transpose(1, 2)  # (B, 6, T)
        feats = [conv(x_t) for conv in self.convs]
        x = torch.cat(feats, dim=1)  # (B, D, T)
        x = self.drop(F.relu(self.bn(x)))
        x = x.transpose(1, 2)  # (B, T, D)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ================================================================
#  Main model
# ================================================================

class MMA_SkeletonIMU(nn.Module):
    """
    Dual-branch Momentum Mamba for skeleton + IMU fusion.

    Encoder options (skel_enc / imu_enc):
      - skel_enc="linear": baseline Linear(60→D)
      - skel_enc="spatial": bone features + velocity → Linear(177→D)
      - imu_enc="conv1d": baseline Conv1D(k=3)
      - imu_enc="multiscale": parallel Conv1D(k=3,7,15)

    Fusion modes:
      - "attention"/"gated": Independent pools → DimGatedFusion
      - "cross_mamba": Concat sequences → shared MambaBlock → pool
      - "concat": Concatenate pooled representations
    """

    def __init__(
        self,
        num_classes: int = 27,
        d_model: int = 128,
        n_layers: int = 2,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
        use_momentum: bool = True,
        momentum_mode: str = "real",
        alpha_init: float = 0.6,
        beta_init: float = 0.99,
        fusion: str = "attention",
        n_heads: int = 4,
        modality_dropout: float = 0.1,
        aux_weight: float = 0.0,
        skel_enc: str = "linear",
        imu_enc: str = "conv1d",
        pos_enc: bool = False,
        center_joints: bool = False,
        n_cross_layers: int = 1,
        no_mod_embed: bool = False,
        single_modality: str = "both",
    ):
        super().__init__()
        self.fusion_mode = fusion
        self.modality_dropout = modality_dropout
        self.aux_weight = aux_weight
        self.no_mod_embed = no_mod_embed
        self.single_modality = single_modality

        mamba_kw = dict(
            d_model=d_model, n_layers=n_layers, d_state=d_state,
            d_conv=d_conv, expand=expand, dropout=dropout,
            use_momentum=use_momentum, momentum_mode=momentum_mode,
            alpha_init=alpha_init, beta_init=beta_init,
        )
        cross_kw = dict(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
            use_momentum=use_momentum, momentum_mode=momentum_mode,
            alpha_init=alpha_init, beta_init=beta_init,
        )

        # --- Encoder selection ---
        if skel_enc == "spatial":
            self.skel_enc = SpatialSkeletonEncoder(in_dim=60, **mamba_kw)
        else:
            self.skel_enc = SkeletonEncoder(in_dim=60, center_joints=center_joints, **mamba_kw)

        if imu_enc == "multiscale":
            self.imu_enc = MultiScaleIMUEncoder(in_channels=6, **mamba_kw)
        else:
            self.imu_enc = IMUEncoder(in_channels=6, **mamba_kw)

        # --- Optional positional encoding ---
        self.skel_pe = LearnablePositionalEncoding(d_model, max_len=256) if pos_enc else None
        self.imu_pe = LearnablePositionalEncoding(d_model, max_len=512) if pos_enc else None

        # --- Fusion ---
        # --- Single-modality shortcut ---
        if single_modality in ("skel", "imu"):
            self.pool = AttentionPool(d_model)
            self.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(d_model, num_classes),
            )
        elif fusion == "cross_mamba":
            self.mod_embed_skel = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.mod_embed_imu = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.cross_blocks = nn.ModuleList([
                MomentumMambaBlock(**cross_kw) for _ in range(n_cross_layers)
            ])
            self.pool = AttentionPool(d_model)
            self.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(d_model, num_classes),
            )
        elif fusion in ("attention", "gated"):
            self.skel_pool = AttentionPool(d_model)
            self.imu_pool = AttentionPool(d_model)
            self.fusion_layer = DimGatedFusion(d_model)
            self.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(d_model, num_classes),
            )
        elif fusion == "concat":
            self.skel_pool = AttentionPool(d_model)
            self.imu_pool = AttentionPool(d_model)
            self.fusion_layer = None
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )
        else:
            raise ValueError(f"Unknown fusion mode: {fusion}")

        # Auxiliary per-modality heads
        self._aux_logits = None
        if aux_weight > 0:
            self.aux_skel = AuxHead(d_model, num_classes, dropout)
            self.aux_imu = AuxHead(d_model, num_classes, dropout)

    def forward(self, skel, imu):
        """
        skel: (B, T_s, 60) — skeleton joint coordinates
        imu:  (B, T_i, 6)  — 6-axis IMU data
        -> logits (B, num_classes)
        """
        # --- Single-modality shortcut ---
        if self.single_modality == "skel":
            fs = self.skel_enc(skel)
            if self.skel_pe is not None:
                fs = self.skel_pe(fs)
            return self.head(self.pool(fs))
        elif self.single_modality == "imu":
            fi = self.imu_enc(imu)
            if self.imu_pe is not None:
                fi = self.imu_pe(fi)
            return self.head(self.pool(fi))

        fs = self.skel_enc(skel)   # (B, T_s, D)
        fi = self.imu_enc(imu)     # (B, T_i, D)

        # Optional positional encoding
        if self.skel_pe is not None:
            fs = self.skel_pe(fs)
        if self.imu_pe is not None:
            fi = self.imu_pe(fi)

        # Auxiliary outputs
        self._aux_logits = None
        if self.aux_weight > 0 and self.training:
            self._aux_logits = (self.aux_skel(fs), self.aux_imu(fi))

        if self.fusion_mode == "cross_mamba":
            if not self.no_mod_embed:
                fs = fs + self.mod_embed_skel
                fi = fi + self.mod_embed_imu
            fused = torch.cat([fs, fi], dim=1)
            for blk in self.cross_blocks:
                fused = blk(fused)
            pooled = self.pool(fused)
        else:
            # attention/gated or concat
            ps = self.skel_pool(fs)
            pi = self.imu_pool(fi)
            # Modality dropout
            if self.training and self.modality_dropout > 0:
                r = random.random()
                if r < self.modality_dropout:
                    ps = torch.zeros_like(ps)
                elif r < 2 * self.modality_dropout:
                    pi = torch.zeros_like(pi)
            if self.fusion_layer is not None:
                pooled = self.fusion_layer(ps, pi)
            else:
                pooled = torch.cat([ps, pi], dim=-1)

        return self.head(pooled)
