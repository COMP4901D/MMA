"""MomentumMambaHAR: Pure IMU pipeline for HAR using Momentum Mamba."""

import torch.nn as nn

from .layers import RMSNorm
from .mamba import MomentumMambaBlock


class MomentumMambaHAR(nn.Module):
    """
    Conv1D Front-End -> N x MomentumMambaBlock -> GlobalAvgPool + Linear

    Input:  (B, L, 6)  — 6-axis IMU window
    Output: (B, num_classes)  — logits
    """

    def __init__(
        self,
        in_channels: int = 6,
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
    ):
        super().__init__()

        self.frontend = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.backbone = nn.ModuleList([
            MomentumMambaBlock(
                d_model, d_state, d_conv, expand,
                use_momentum, momentum_mode,
                alpha_init, beta_init,
            )
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """x: (B, L, 6) -> logits (B, num_classes)"""
        x = self.frontend(x.transpose(1, 2)).transpose(1, 2)
        for block in self.backbone:
            x = block(x)
        x = self.norm(x).mean(dim=1)
        return self.head(self.drop(x))
