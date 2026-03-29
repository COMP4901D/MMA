"""
MultimodalMMA: RGB-D + IMU multimodal HAR with Momentum Mamba.

Architecture:
  RGB-D (N x 4 x H x W) -> RGBDEncoder (spatial+temporal) --+
                                                              |-> Fusion -> Head -> 27 classes
  IMU   (T x 6)          -> IMUEncoder (1D or GAF+2D)     --+
"""

import torch
import torch.nn as nn

from .encoders import RGBDEncoder, IMUEncoder
from .fusion import CrossModalAttentionFusion, GatedFusionSimple


class MultimodalMMA(nn.Module):
    """Full multimodal model with configurable encoders and fusion."""

    def __init__(
        self,
        num_classes: int = 27,
        d_model: int = 128,
        n_layers: int = 2,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_momentum: bool = True,
        momentum_mode: str = "real",
        alpha_init: float = 0.6,
        beta_init: float = 0.99,
        fusion: str = "attention",
        n_heads: int = 4,
        encoder: str = "default",
        convnext_model: str = "convnextv2_atto",
        freeze_stages: int = 3,
        imu_encoder: str = "conv1d",
        gaf_size: int = 64,
    ):
        super().__init__()
        self.fusion_mode = fusion

        mamba_kw = dict(
            d_model=d_model, n_layers=n_layers, d_state=d_state,
            d_conv=d_conv, expand=expand, dropout=dropout,
            use_momentum=use_momentum, momentum_mode=momentum_mode,
            alpha_init=alpha_init, beta_init=beta_init,
        )

        self.rgbd_enc = RGBDEncoder(
            **mamba_kw,
            encoder="convnextv2" if encoder == "convnextv2" else "spatial_cnn",
            convnext_model=convnext_model,
            freeze_stages=freeze_stages,
        )
        self.imu_enc = IMUEncoder(
            in_channels=6, **mamba_kw,
            encoder=imu_encoder,
            convnext_model=convnext_model,
            freeze_stages=freeze_stages,
            gaf_size=gaf_size,
        )

        if fusion == "attention":
            self.fusion = CrossModalAttentionFusion(d_model, n_heads, dropout)
            self.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(d_model, num_classes),
            )
        elif fusion == "gated":
            self.fusion = GatedFusionSimple(d_model)
            self.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(d_model, num_classes),
            )
        elif fusion == "concat":
            self.fusion = None
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

    def forward(self, rgbd, imu):
        """
        rgbd: (B, N, 4, H, W)
        imu:  (B, T, 6) or (B, 6, H, W) depending on encoder
        -> logits (B, num_classes)
        """
        fv = self.rgbd_enc(rgbd)
        fi = self.imu_enc(imu)

        if self.fusion_mode == "attention":
            pooled = self.fusion(fv, fi)
        elif self.fusion_mode == "gated":
            pooled = self.fusion(fv.mean(1), fi.mean(1))
        else:  # concat
            pooled = torch.cat([fv.mean(1), fi.mean(1)], dim=-1)

        return self.head(pooled)
