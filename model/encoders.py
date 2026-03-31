"""
Modality-specific encoders.

RGBDEncoder: Spatial 2D CNN (per-frame) + MomentumMamba (temporal)
IMUEncoder:  Conv1D frontend  or  ConvNeXtV2 (GAF images) + MomentumMamba
"""

import torch
import torch.nn as nn

from .layers import RMSNorm
from .mamba import MomentumMambaBlock


class RGBDEncoder(nn.Module):
    """SpatialCNN/ConvNeXtV2/PretrainedCNN per frame -> MomentumMamba temporal -> RMSNorm

    New options:
      encoder="pretrained_cnn": ResNet18 pretrained on ImageNet (best for small datasets)
      freeze="all"/"partial"/"none": freeze strategy for pretrained backbone
      temporal_velocity=True: compute feature-level frame differences (mirrors skeleton velocity)
    """

    def __init__(
        self,
        d_model=128,
        n_layers=2,
        d_state=64,
        d_conv=4,
        expand=2,
        dropout=0.1,
        use_momentum=True,
        momentum_mode="real",
        alpha_init=0.6,
        beta_init=0.99,
        encoder="spatial_cnn",
        convnext_model="convnextv2_atto",
        freeze_stages=3,
        freeze="all",
        temporal_velocity=False,
        in_channels=4,
    ):
        super().__init__()
        self.temporal_velocity = temporal_velocity

        if encoder == "convnextv2":
            from .backbones import ConvNeXtV2Encoder
            self.spatial = ConvNeXtV2Encoder(
                in_channels=in_channels, feat_dim=d_model,
                model_name=convnext_model, freeze_stages=freeze_stages,
            )
        elif encoder == "pretrained_cnn":
            from .backbones import PretrainedCNN
            self.spatial = PretrainedCNN(
                in_channels=in_channels, d_model=d_model, freeze=freeze,
            )
        else:
            from .backbones import SpatialCNN
            self.spatial = SpatialCNN(in_channels=in_channels, d_model=d_model)

        # Temporal velocity: concat [feat, vel] -> project back to d_model
        if temporal_velocity:
            self.vel_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
            )

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            MomentumMambaBlock(
                d_model, d_state, d_conv, expand,
                use_momentum, momentum_mode, alpha_init, beta_init,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        """
        x: (B, N_frames, 4, H, W) -> (B, N_frames, d_model)
        """
        B, N, C, H, W = x.shape
        feat = self.spatial(x.reshape(B * N, C, H, W))
        feat = feat.reshape(B, N, -1)  # (B, N, d_model)

        # Feature-level temporal velocity (mirrors skeleton velocity features)
        if self.temporal_velocity:
            vel = torch.zeros_like(feat)
            vel[:, 1:] = feat[:, 1:] - feat[:, :-1]
            feat = self.vel_proj(torch.cat([feat, vel], dim=-1))

        feat = self.drop(feat)
        for blk in self.blocks:
            feat = blk(feat)
        return self.norm(feat)


class IMUEncoder(nn.Module):
    """Conv1D / ConvNeXtV2 (GAF) -> MomentumMamba -> RMSNorm"""

    def __init__(
        self,
        in_channels=6,
        d_model=128,
        n_layers=2,
        d_state=64,
        d_conv=4,
        expand=2,
        dropout=0.1,
        use_momentum=True,
        momentum_mode="real",
        alpha_init=0.6,
        beta_init=0.99,
        encoder="conv1d",
        convnext_model="convnextv2_atto",
        freeze_stages=3,
        gaf_size=64,
    ):
        super().__init__()
        self.encoder_type = encoder
        self.gaf_size = gaf_size

        if encoder == "convnextv2":
            from .backbones import ConvNeXtV2Encoder
            self.spatial = ConvNeXtV2Encoder(
                in_channels=in_channels, feat_dim=d_model,
                model_name=convnext_model, freeze_stages=freeze_stages,
            )
        else:
            self.frontend = nn.Sequential(
                nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.blocks = nn.ModuleList([
            MomentumMambaBlock(
                d_model, d_state, d_conv, expand,
                use_momentum, momentum_mode, alpha_init, beta_init,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        """
        conv1d mode:      x: (B, T, 6) -> (B, T, d_model)
        convnextv2 mode:  x: (B, 6, H, W) -> (B, 1, d_model)
        """
        if self.encoder_type == "convnextv2":
            feat = self.spatial(x).unsqueeze(1)
            for blk in self.blocks:
                feat = blk(feat)
            return self.norm(feat)
        else:
            x = self.frontend(x.transpose(1, 2)).transpose(1, 2)
            for blk in self.blocks:
                x = blk(x)
            return self.norm(x)
