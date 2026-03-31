"""
MuMu RGBD+IMU: Multimodal MuMu baseline for RGBD+IMU HAR.

Wraps the original MuMu architecture with a ResNet18 frontend for RGBD
frame-level feature extraction. Both modality feature sequences are fed
into MuMu's Bi-LSTM UFE → SM-Fusion → GM-Fusion pipeline.

Architecture:
  RGBD (B,T,4,H,W) -> ResNet18(freeze=partial) -> (B,T,D_cnn) -+
                                                                 |-> MuMu(num_modalities=2)
  IMU  (B,T,6)      ------------------------------------------- +

Supports None inputs for missing-modality evaluation.
"""

import torch
import torch.nn as nn
from model.backbones.pretrained_cnn import PretrainedCNN
from .MuMu import MuMu


class MuMuRGBDIMU(nn.Module):
    """Multimodal MuMu for RGBD + IMU action recognition."""

    def __init__(
        self,
        num_activities: int = 27,
        num_activity_groups: int = 5,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        cnn_d_model: int = 128,
        freeze: str = "partial",
        in_channels: int = 4,
    ):
        super().__init__()
        self.cnn_d_model = cnn_d_model

        # ResNet18 for per-frame RGBD feature extraction
        self.rgbd_cnn = PretrainedCNN(
            in_channels=in_channels, d_model=cnn_d_model, freeze=freeze,
        )

        # MuMu with 2 modalities: RGBD features (cnn_d_model) + IMU (6)
        self.mumu = MuMu(
            num_modalities=2,
            feature_dim=feature_dim,
            num_activity_groups=num_activity_groups,
            num_activities=num_activities,
            input_dim=[cnn_d_model, 6],
            hidden_dim=hidden_dim,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
        )

    def forward(self, rgbd, imu):
        """
        rgbd: (B, T, 4, H, W) or None
        imu:  (B, T_imu, 6) or None
        Returns: (y_aux, y_target, alpha, attn_weights)
        """
        if rgbd is not None:
            B, T, C, H, W = rgbd.shape
            rgbd_feat = self.rgbd_cnn(rgbd.reshape(B * T, C, H, W))
            rgbd_feat = rgbd_feat.reshape(B, T, -1)  # (B, T, cnn_d_model)
        else:
            # Use zero features matching IMU batch size
            B = imu.shape[0]
            rgbd_feat = torch.zeros(
                B, 1, self.cnn_d_model, device=imu.device, dtype=imu.dtype
            )

        if imu is None:
            B = rgbd.shape[0]
            imu = torch.zeros(B, 1, 6, device=rgbd.device, dtype=rgbd.dtype)

        return self.mumu([rgbd_feat, imu])
