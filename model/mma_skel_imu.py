"""
MMA_SkeletonIMU: Skeleton + IMU dual-branch Momentum Mamba with cross-modal fusion.

Fusion modes:
  - "attention"/"gated": IndependentPools → DimGatedFusion
  - "cross_mamba": Concat sequences → shared MambaBlock → pool
  - "hybrid": CrossMamba → split → per-modal AttnPool → DimGatedFusion
  - "hybrid_skip": Hybrid + skip connections from pre-cross features
  - "hybrid_bi": Bidirectional cross-mamba → split → pool → gate
  - "concat": Concatenate pooled representations
"""

import random

import torch
import torch.nn as nn

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
    ):
        super().__init__()
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


# ================================================================
#  Main model
# ================================================================

class MMA_SkeletonIMU(nn.Module):
    """
    Dual-branch Momentum Mamba for skeleton + IMU fusion.

    Fusion modes:
      - "attention"/"gated": Independent pools → DimGatedFusion
      - "cross_mamba": Concatenate sequences → shared MambaBlock → pool
      - "hybrid": CrossMamba → split → per-modal AttnPool → DimGatedFusion
      - "hybrid_skip": Hybrid + residual skip from pre-cross features
      - "hybrid_bi": Bidirectional cross-mamba hybrid
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
    ):
        super().__init__()
        self.fusion_mode = fusion
        self.modality_dropout = modality_dropout
        self.aux_weight = aux_weight

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

        self.skel_enc = SkeletonEncoder(in_dim=60, **mamba_kw)
        self.imu_enc = IMUEncoder(in_channels=6, **mamba_kw)

        is_hybrid = fusion.startswith("hybrid")

        if fusion == "cross_mamba" or is_hybrid:
            # Learnable modality embeddings
            self.mod_embed_skel = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.mod_embed_imu = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            # Shared cross-modal Mamba block
            self.cross_block = MomentumMambaBlock(**cross_kw)

        if fusion == "hybrid_bi":
            # Reverse-direction cross-mamba block
            self.cross_block_rev = MomentumMambaBlock(**cross_kw)

        if fusion == "cross_mamba":
            self.pool = AttentionPool(d_model)
            self.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(d_model, num_classes),
            )
        elif is_hybrid:
            self.skel_pool = AttentionPool(d_model)
            self.imu_pool = AttentionPool(d_model)
            self.fusion_layer = DimGatedFusion(d_model)
            if fusion == "hybrid_skip":
                # Pre-cross skip pools + projection to combine pre and post
                self.skel_pool_pre = AttentionPool(d_model)
                self.imu_pool_pre = AttentionPool(d_model)
                self.skip_proj = nn.Sequential(
                    nn.Linear(d_model * 2, d_model), nn.ReLU(),
                )
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

        # Auxiliary per-modality heads (multi-task regularization)
        self._aux_logits = None  # populated during training forward
        if aux_weight > 0:
            self.aux_skel = AuxHead(d_model, num_classes, dropout)
            self.aux_imu = AuxHead(d_model, num_classes, dropout)

    def forward(self, skel, imu):
        """
        skel: (B, T_s, 60) — skeleton joint coordinates
        imu:  (B, T_i, 6)  — 6-axis IMU data
        -> logits (B, num_classes)
        """
        fs = self.skel_enc(skel)   # (B, T_s, D)
        fi = self.imu_enc(imu)     # (B, T_i, D)

        # Auxiliary outputs (computed before fusion, from raw encoder features)
        self._aux_logits = None
        if self.aux_weight > 0 and self.training:
            self._aux_logits = (self.aux_skel(fs), self.aux_imu(fi))

        if self.fusion_mode == "cross_mamba":
            fs = fs + self.mod_embed_skel
            fi = fi + self.mod_embed_imu
            fused = torch.cat([fs, fi], dim=1)
            fused = self.cross_block(fused)
            pooled = self.pool(fused)

        elif self.fusion_mode == "hybrid":
            T_s = fs.size(1)
            fs = fs + self.mod_embed_skel
            fi = fi + self.mod_embed_imu
            fused = self.cross_block(torch.cat([fs, fi], dim=1))
            # Split back into per-modality sequences
            fs_cross, fi_cross = fused[:, :T_s], fused[:, T_s:]
            ps = self.skel_pool(fs_cross)
            pi = self.imu_pool(fi_cross)
            pooled = self.fusion_layer(ps, pi)

        elif self.fusion_mode == "hybrid_skip":
            T_s = fs.size(1)
            # Pre-cross pooling (skip connection)
            pre_skel = self.skel_pool_pre(fs)
            pre_imu = self.imu_pool_pre(fi)
            pre_fused = self.fusion_layer(pre_skel, pre_imu)  # reuse same gate
            # Cross-mamba
            fs = fs + self.mod_embed_skel
            fi = fi + self.mod_embed_imu
            fused = self.cross_block(torch.cat([fs, fi], dim=1))
            fs_cross, fi_cross = fused[:, :T_s], fused[:, T_s:]
            ps = self.skel_pool(fs_cross)
            pi = self.imu_pool(fi_cross)
            post_fused = self.fusion_layer(ps, pi)
            # Combine multi-scale: pre-cross + post-cross
            pooled = self.skip_proj(torch.cat([pre_fused, post_fused], dim=-1))

        elif self.fusion_mode == "hybrid_bi":
            T_s = fs.size(1)
            fs_e = fs + self.mod_embed_skel
            fi_e = fi + self.mod_embed_imu
            cat_seq = torch.cat([fs_e, fi_e], dim=1)
            # Forward pass
            fwd = self.cross_block(cat_seq)
            # Reverse pass (flip time, process, flip back)
            rev = self.cross_block_rev(cat_seq.flip(1)).flip(1)
            # Average bidirectional
            fused = (fwd + rev) * 0.5
            fs_cross, fi_cross = fused[:, :T_s], fused[:, T_s:]
            ps = self.skel_pool(fs_cross)
            pi = self.imu_pool(fi_cross)
            pooled = self.fusion_layer(ps, pi)

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
            else:  # concat
                pooled = torch.cat([ps, pi], dim=-1)

        return self.head(pooled)
