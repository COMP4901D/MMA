"""
MultimodalMMA: RGB-D + IMU multimodal HAR with Momentum Mamba.

Architecture:
  RGB-D (N x 4 x H x W) -> RGBDEncoder (spatial+temporal) --+
                                                              |-> Fusion -> Head -> 27 classes
  IMU   (T x 6)          -> IMUEncoder (Conv1D+Mamba)      --+

Fusion modes (ported from skel_imu best practices):
  - "attention"/"gated": AttentionPool per branch -> DimGatedFusion
  - "cross_mamba": Modality embeddings -> concat -> shared MambaBlock -> AttentionPool
  - "concat": AttentionPool per branch -> MLP
"""

import torch
import torch.nn as nn

from .encoders import RGBDEncoder, IMUEncoder
from .mamba import MomentumMambaBlock
from .mma_skel_imu import AttentionPool, DimGatedFusion, AuxHead


class MultimodalMMA(nn.Module):
    """RGBD+IMU multimodal model with cross_mamba fusion and auxiliary loss."""

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
        fusion: str = "cross_mamba",
        n_heads: int = 4,
        encoder: str = "default",
        convnext_model: str = "convnextv2_atto",
        freeze_stages: int = 3,
        freeze: str = "all",
        temporal_velocity: bool = False,
        in_channels: int = 4,
        imu_encoder: str = "conv1d",
        gaf_size: int = 64,
        aux_weight: float = 0.0,
    ):
        super().__init__()
        self.fusion_mode = fusion
        self.aux_weight = aux_weight
        self._aux_logits = None

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

        # Map encoder name
        enc_name = encoder
        if enc_name == "convnextv2":
            enc_name = "convnextv2"
        elif enc_name in ("pretrained", "pretrained_cnn", "resnet18"):
            enc_name = "pretrained_cnn"
        else:
            enc_name = "spatial_cnn"

        self.rgbd_enc = RGBDEncoder(
            **mamba_kw,
            encoder=enc_name,
            convnext_model=convnext_model,
            freeze_stages=freeze_stages,
            freeze=freeze,
            temporal_velocity=temporal_velocity,
            in_channels=in_channels,
        )
        self.imu_enc = IMUEncoder(
            in_channels=6, **mamba_kw,
            encoder=imu_encoder,
            convnext_model=convnext_model,
            freeze_stages=freeze_stages,
            gaf_size=gaf_size,
        )

        # --- Fusion ---
        if fusion == "cross_mamba":
            self.mod_embed_rgbd = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.mod_embed_imu = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.cross_block = MomentumMambaBlock(**cross_kw)
            self.pool = AttentionPool(d_model)
            self.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(d_model, num_classes),
            )
        elif fusion in ("attention", "gated"):
            self.rgbd_pool = AttentionPool(d_model)
            self.imu_pool = AttentionPool(d_model)
            self.fusion_layer = DimGatedFusion(d_model)
            self.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(d_model, num_classes),
            )
        elif fusion == "concat":
            self.rgbd_pool = AttentionPool(d_model)
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
            raise ValueError(f"Unknown fusion: {fusion}")

        # Auxiliary per-modality heads
        if aux_weight > 0:
            self.aux_rgbd = AuxHead(d_model, num_classes, dropout)
            self.aux_imu = AuxHead(d_model, num_classes, dropout)

    def forward(self, rgbd, imu):
        """
        rgbd: (B, N, 4, H, W)
        imu:  (B, T, 6)
        -> logits (B, num_classes)
        """
        fv = self.rgbd_enc(rgbd)  # (B, N, D)
        fi = self.imu_enc(imu)    # (B, T, D)

        # Auxiliary outputs
        self._aux_logits = None
        if self.aux_weight > 0 and self.training:
            self._aux_logits = (self.aux_rgbd(fv), self.aux_imu(fi))

        if self.fusion_mode == "cross_mamba":
            fv = fv + self.mod_embed_rgbd
            fi = fi + self.mod_embed_imu
            fused = torch.cat([fv, fi], dim=1)
            fused = self.cross_block(fused)
            pooled = self.pool(fused)
        elif self.fusion_mode in ("attention", "gated"):
            pv = self.rgbd_pool(fv)
            pi = self.imu_pool(fi)
            pooled = self.fusion_layer(pv, pi)
        else:  # concat
            pv = self.rgbd_pool(fv)
            pi = self.imu_pool(fi)
            pooled = torch.cat([pv, pi], dim=-1)

        return self.head(pooled)
